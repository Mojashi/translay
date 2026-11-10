// 文字起こしエンジンの比較用: 音声ファイルを各エンジンで認識し、全文を JSON で出す。
//
//     translay-asr-eval --engine nemotron560|parakeet-v2|parakeet-v3 --models <dir> <wav>...
//
// Prints one JSON line per file: {"engine", "file", "text", "seconds"}.
import AVFoundation
import FluidAudio
import Foundation

func argument(_ flag: String) -> String {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { fatalError("missing \(flag) <value>") }
    return args[i + 1]
}

func audioFiles() -> [URL] {
    var files: [URL] = []
    var skip = false
    for (i, a) in CommandLine.arguments.enumerated() where i > 0 {
        if skip { skip = false; continue }
        if a.hasPrefix("--") { skip = true; continue }
        files.append(URL(fileURLWithPath: a))
    }
    guard !files.isEmpty else { fatalError("no audio files given") }
    return files
}

func emit(_ engine: String, _ file: URL, _ text: String, _ seconds: Double) {
    let object: [String: Any] = ["engine": engine, "file": file.lastPathComponent, "text": text, "seconds": seconds]
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
    fflush(stdout)
}

func readBuffer(_ url: URL) throws -> AVAudioPCMBuffer {
    let file = try AVAudioFile(forReading: url)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    return buffer
}

func run() async throws {
    let engine = argument("--engine")
    let models = URL(fileURLWithPath: argument("--models"))
    switch engine {
    case "nemotron560":
        let asr = StreamingNemotronAsrManager(requestedChunkSize: .ms560)
        try await asr.loadModels(to: models)
        for url in audioFiles() {
            let t0 = Date()
            _ = try await asr.process(audioBuffer: try readBuffer(url))
            let text = try await asr.finish()
            emit(engine, url, text, Date().timeIntervalSince(t0))
            await asr.reset()
        }
    case "parakeet-v2", "parakeet-v3":
        let version: AsrModelVersion = engine == "parakeet-v2" ? .v2 : .v3
        let loaded = try await AsrModels.downloadAndLoad(to: models, version: version)
        let asr = AsrManager(config: .default, models: loaded)
        for url in audioFiles() {
            let t0 = Date()
            var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
            let result = try await asr.transcribe(url, decoderState: &state)
            emit(engine, url, result.text, Date().timeIntervalSince(t0))
        }
    default:
        fatalError("unknown engine \(engine)")
    }
}

Task.detached {
    do {
        try await run()
        exit(0)
    } catch {
        FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
dispatchMain()
