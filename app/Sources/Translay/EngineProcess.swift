import AVFoundation
import Foundation
import TranslayCore

/// エンジン: 聞き取り（Qwen3-ASR）と翻訳（PLaMo 3）を担う常駐 Python プロセス（translay_mt.engine）。
///
/// The app only streams audio in and renders what comes out. All GPU work happens in the engine,
/// one job after another, so recognition and translation never compete.
final class EngineProcess: @unchecked Sendable {
    struct Launch {
        let uv: URL                 // absolute: GUI apps do not inherit the shell's PATH
        let project: URL            // the translator/ uv project
        let recogniser: String      // "whisper" or "qwen"
        let asrModel: URL
        let aneEncoder: URL?        // Whisper only: Core ML encoder for the Neural Engine
        let translationModel: URL
        let targets: [String]
        let termsFile: URL
        let language: String        // "English" etc., or "auto"
        var trace: Bool
    }

    /// エンジンから届く出来事
    enum Event {
        case status(String)
        case ready
        case speaking(LineID, String, language: String?)
        case finished(LineID, String, language: String?)
        case translation(Rendering)
        case pass([String: Any])
        case exited(Int32)
    }

    let events: AsyncStream<Event>
    private let process = Process()
    private let stdin = Pipe()
    private let writeLock = NSLock()

    init(_ launch: Launch) {
        let stdout = Pipe()
        process.executableURL = launch.uv
        process.arguments = ["run", "--project", launch.project.path, "python", "-m", "translay_mt.engine",
                             "--recogniser", launch.recogniser, "--asr-model", launch.asrModel.path, "--mt-model", launch.translationModel.path,
                             "--targets", launch.targets.joined(separator: ","),
                             "--terms-file", launch.termsFile.path, "--language", launch.language]
            + (launch.aneEncoder.map { ["--whisper-ane-encoder", $0.path] } ?? [])
            + (launch.trace ? ["--trace"] : [])
        process.currentDirectoryURL = launch.project
        // Captions are interactive: without this the child inherits the launching task's QoS and can
        // be parked on efficiency cores, where MLX's per-token dispatch loop runs several× slower.
        process.qualityOfService = .userInteractive
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = Self.logFile("engine.log")

        let lines = stdout.fileHandleForReading.bytes.lines
        let proc = process
        events = AsyncStream { continuation in
            Task {
                do {
                    for try await line in lines {
                        if let event = Self.parse(line) { continuation.yield(event) }
                    }
                } catch {}
                proc.waitUntilExit()
                continuation.yield(.exited(proc.terminationStatus))
                continuation.finish()
            }
        }
        do {
            try process.run()
        } catch {
            fatalError("エンジンを起動できません: \(launch.uv.path) — \(error)")
        }
    }

    /// 音を渡す（16 kHz モノラルに変換済みのもの）
    func hear(_ pcm16k: [Float]) {
        var ints = [Int16](repeating: 0, count: pcm16k.count)
        for i in pcm16k.indices {
            ints[i] = Int16(max(-1, min(1, pcm16k[i])) * 32767)
        }
        let data = ints.withUnsafeBufferPointer { Data(buffer: $0) }
        send(["type": "audio", "pcm16": data.base64EncodedString()])
    }

    /// 音源が止まった（話し途中の文も確定させる）
    func endOfAudio() {
        send(["type": "end"])
    }

    /// 単語登録を反映する
    func register(terms: [String]) {
        send(["type": "terms", "terms": terms])
    }

    /// 訳す先の言語を選び直す（PLaMo 3 が訳せる言語の中から。空なら訳さない）
    func translate(into languages: [String]) {
        send(["type": "targets", "targets": languages])
    }

    /// 認識言語を変える（"English" など、または "auto"）
    func recognise(language: String) {
        send(["type": "language", "language": language])
    }

    func terminate() {
        process.terminate()
    }

    private func send(_ message: [String: Any]) {
        guard process.isRunning, var data = try? JSONSerialization.data(withJSONObject: message) else { return }
        data.append(0x0A)
        writeLock.lock()
        defer { writeLock.unlock() }
        try? stdin.fileHandleForWriting.write(contentsOf: data)
    }

    /// ~/Library/Logs/Translay/<name>, opened for appending (the engine's stderr goes there).
    static func logFile(_ name: String) -> FileHandle {
        let dir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Logs/Translay", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { fatalError("cannot open log \(url.path)") }
        handle.seekToEndOfFile()
        handle.write("---- \(Date()) ----\n".data(using: .utf8)!)
        return handle
    }

    private static func parse(_ line: String) -> Event? {
        guard let data = line.data(using: .utf8),
              let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = o["type"] as? String else { return nil }
        let line = LineID(o["line"] as? Int ?? 0)
        switch type {
        case "ready": return .ready
        case "status": return .status(o["text"] as? String ?? "")
        case "speaking": return .speaking(line, o["en"] as? String ?? "", language: o["lang"] as? String)
        case "finished": return .finished(line, o["en"] as? String ?? "", language: o["lang"] as? String)
        case "translation":
            return .translation(Rendering(line: line, translations: o["texts"] as? [String: String] ?? [:],
                                          isFinal: o["final"] as? Bool ?? false))
        case "pass": return .pass(o)
        default: return nil
        }
    }
}

/// 音源の音（任意の形式）を、エンジンが聞く 16 kHz モノラルに変換する。
final class Resampler {
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                                       interleaved: false)!
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    func convert(_ buffer: AVAudioPCMBuffer) -> [Float] {
        if sourceFormat != buffer.format {
            sourceFormat = buffer.format
            converter = AVAudioConverter(from: buffer.format, to: target)
        }
        guard let converter else { return [] }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16_000 / buffer.format.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return [] }
        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let channel = out.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(out.frameLength)))
    }
}
