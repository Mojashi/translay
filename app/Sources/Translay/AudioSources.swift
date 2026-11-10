import AVFoundation
import CoreMedia
import ScreenCaptureKit

/// 音源: 音を AVAudioPCMBuffer の流れとして渡す。形式（サンプルレート・チャンネル）は問わない —
/// the recogniser resamples to 16 kHz mono itself.
protocol AudioSource: AnyObject {
    var name: String { get }
    func sounds() async throws -> AsyncStream<AVAudioPCMBuffer>
    func stop() async
}

enum AudioSourceError: Error, CustomStringConvertible {
    case microphoneDenied
    case noDisplay
    case unreadableFile(URL)

    var description: String {
        switch self {
        case .microphoneDenied: return "マイクへのアクセスが許可されていません（システム設定 > プライバシーとセキュリティ > マイク）"
        case .noDisplay: return "画面収録の対象ディスプレイが見つかりません（画面収録の許可を確認してください）"
        case .unreadableFile(let url): return "音声ファイルを読めません: \(url.path)"
        }
    }
}

extension AVAudioPCMBuffer {
    /// Tap and capture buffers are reused by their producers; keep our own copy.
    func copied() -> AVAudioPCMBuffer {
        let copy = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength)!
        copy.frameLength = frameLength
        let src = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: audioBufferList))
        let dst = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        for (s, d) in zip(src, dst) {
            if let sData = s.mData, let dData = d.mData {
                memcpy(dData, sData, Int(min(s.mDataByteSize, d.mDataByteSize)))
            }
        }
        return copy
    }

    var duration: Double { Double(frameLength) / format.sampleRate }
}

// MARK: - マイク

final class MicrophoneSource: AudioSource {
    let name = "マイク"
    private let engine = AVAudioEngine()

    func sounds() async throws -> AsyncStream<AVAudioPCMBuffer> {
        guard await AVCaptureDevice.requestAccess(for: .audio) else { throw AudioSourceError.microphoneDenied }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        let (stream, continuation) = AsyncStream.makeStream(of: AVAudioPCMBuffer.self, bufferingPolicy: .unbounded)
        // ~100 ms taps: small enough that the 560 ms recogniser chunk fills promptly.
        let frames = AVAudioFrameCount(format.sampleRate / 10)
        input.installTap(onBus: 0, bufferSize: frames, format: format) { buffer, _ in
            continuation.yield(buffer.copied())
        }
        continuation.onTermination = { [engine] _ in
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        try engine.start()
        return stream
    }

    func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

// MARK: - システム音声（ScreenCaptureKit）

final class SystemAudioSource: NSObject, AudioSource, SCStreamOutput, SCStreamDelegate {
    let name = "システム音声"
    private var stream: SCStream?
    private var continuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private let queue = DispatchQueue(label: "translay.system-audio")

    func sounds() async throws -> AsyncStream<AVAudioPCMBuffer> {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else { throw AudioSourceError.noDisplay }
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 1
        // Audio is all we want; keep the mandatory video side as cheap as possible.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let (output, continuation) = AsyncStream.makeStream(of: AVAudioPCMBuffer.self, bufferingPolicy: .unbounded)
        self.continuation = continuation
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
        return output
    }

    func stop() async {
        try? await stream?.stopCapture()
        stream = nil
        continuation?.finish()
        continuation = nil
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, let buffer = Self.pcmBuffer(from: sampleBuffer) else { return }
        continuation?.yield(buffer)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("system audio stopped: \(error)\n".data(using: .utf8)!)
        continuation?.finish()
    }

    static func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = sampleBuffer.formatDescription else { return nil }
        let format = AVAudioFormat(cmAudioFormatDescription: description)
        let frames = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        buffer.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        return status == noErr ? buffer : nil
    }
}

// MARK: - 音声ファイル（検証用）

/// Plays a file into the pipeline in 100 ms buffers, optionally at real-time pace, so the whole
/// system can be measured end to end without a microphone.
final class FileSource: AudioSource {
    let name: String
    private let url: URL
    private let realtime: Bool
    private var task: Task<Void, Never>?

    init(url: URL, realtime: Bool) {
        self.url = url
        self.realtime = realtime
        self.name = "ファイル(\(url.lastPathComponent))"
    }

    func sounds() async throws -> AsyncStream<AVAudioPCMBuffer> {
        guard let file = try? AVAudioFile(forReading: url) else { throw AudioSourceError.unreadableFile(url) }
        let (stream, continuation) = AsyncStream.makeStream(of: AVAudioPCMBuffer.self, bufferingPolicy: .unbounded)
        let realtime = self.realtime
        task = Task.detached {
            let format = file.processingFormat
            let frames = AVAudioFrameCount(format.sampleRate / 10)
            let start = ContinuousClock.now
            var played = 0.0
            while !Task.isCancelled {
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { break }
                do { try file.read(into: buffer, frameCount: frames) } catch { break }
                if buffer.frameLength == 0 { break }
                if realtime {
                    let due = start + .milliseconds(Int(played * 1000))
                    try? await Task.sleep(until: due, clock: .continuous)
                }
                played += buffer.duration
                continuation.yield(buffer)
            }
            continuation.finish()
        }
        return stream
    }

    func stop() async {
        task?.cancel()
    }
}
