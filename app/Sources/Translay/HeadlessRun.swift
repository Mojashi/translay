import Foundation
import os
import TranslayCore

/// 画面なしで音声ファイルを字幕化し、遅れと精度を記録する（検証・計測用）。
///
///     translay --file talk.wav [--realtime] [--trace] [--reference script.txt] --targets es,ja
///              --language English --uv … --engine-project …
///              --whisper-model … --whisper-ane-encoder … --translation-model … --terms-file …
///
/// Reports: recogniser lag behind real time (wall clock − audio heard, per pass), pass time, and
/// for each finished sentence the time from "finished" to its final translation. With
/// --reference, the finished English text is printed for scoring.
enum HeadlessRun {
    static func run(arguments: [String]) async throws {
        guard let i = arguments.firstIndex(of: "--file"), i + 1 < arguments.count else {
            fatalError("--file <audio> が必要です")
        }
        let url = URL(fileURLWithPath: arguments[i + 1])
        let realtime = arguments.contains("--realtime")
        let configuration = Configuration.fromArguments(arguments)
        let engine = EngineProcess(configuration.engineLaunch(
            language: configuration.recognitionLanguage, trace: true))
        let board = await CaptionBoard(languages: configuration.targetLanguages, keep: 1000)
        let clock = Clocked()

        let started = ContinuousClock.now
        let captioning = LiveCaptioning(engine: engine, board: board) { line in clock.mark(line) }
        await MainActor.run {
            board.onFinalTranslation = { line in
                let latency = clock.since(line.id)
                clock.finished(english: line.english, latency: latency)
                log("[\(line.id)] SRC: \(line.english)   (finished→final \(latency.map { "\($0) ms" } ?? "?"))")
                for language in board.languages {
                    log("[\(line.id)] \(language.uppercased()): \(line.translations[language] ?? "")")
                }
            }
        }
        // Wait for the engine to be ready before playing, so the measurement starts warm.
        var readyEvents = engine.events.makeAsyncIterator()
        while let event = await readyEvents.next() {
            if case .ready = event { break }
            if case .exited(let status) = event { throw NSError(domain: "engine", code: Int(status)) }
        }
        log("engine ready in \(ms(ContinuousClock.now - started)) ms")
        let shown = Task { await captioning.showPasses(clock) }
        let playStart = ContinuousClock.now
        clock.startPlayback()
        try await captioning.run(FileSource(url: url, realtime: realtime))
        try await Task.sleep(nanoseconds: 4_000_000_000)        // let the last sentence finish
        shown.cancel()
        let wall = ms(ContinuousClock.now - playStart)

        let latencies = clock.latencies.sorted()
        let lags = clock.lags.sorted()
        let passes = clock.passTimes.sorted()
        func q(_ xs: [Int], _ p: Double) -> Int { xs.isEmpty ? -1 : xs[min(xs.count - 1, Int(Double(xs.count) * p))] }
        log("passes=\(passes.count) pass median=\(q(passes, 0.5)) ms p90=\(q(passes, 0.9)) ms")
        log("recogniser lag behind real time: median=\(q(lags, 0.5)) ms p90=\(q(lags, 0.9)) ms max=\(lags.last ?? -1) ms")
        log("lines=\(latencies.count) finished→final median=\(q(latencies, 0.5)) ms p90=\(q(latencies, 0.9)) ms")
        log("wall \(wall) ms (realtime=\(realtime))")
        log("FINISHED_ENGLISH: \(clock.englishText)")
        engine.terminate()
    }

    private static func ms(_ d: Duration) -> Int {
        Int(d.components.seconds * 1000 + d.components.attoseconds / 1_000_000_000_000_000)
    }

    static func log(_ s: String) {
        print(s)
        fflush(stdout)
    }

    /// Wall-clock marks and collected measurements.
    final class Clocked: Sendable {
        private struct State {
            var marks: [LineID: ContinuousClock.Instant] = [:]
            var latencies: [Int] = []
            var lags: [Int] = []
            var passTimes: [Int] = []
            var english: [String] = []
            var playback: ContinuousClock.Instant?
        }
        private let state = OSAllocatedUnfairLock(initialState: State())

        func startPlayback() { state.withLock { $0.playback = .now } }
        func mark(_ id: LineID) { state.withLock { $0.marks[id] = .now } }
        func since(_ id: LineID) -> Int? {
            state.withLock { s in s.marks[id].map { HeadlessRun.ms(ContinuousClock.now - $0) } }
        }
        func finished(english: String, latency: Int?) {
            state.withLock { s in
                s.english.append(english)
                if let latency { s.latencies.append(latency) }
            }
        }
        func pass(ms: Int, heardSeconds: Double) {
            state.withLock { s in
                s.passTimes.append(ms)
                if let start = s.playback {
                    s.lags.append(HeadlessRun.ms(ContinuousClock.now - start) - Int(heardSeconds * 1000))
                }
            }
        }
        var latencies: [Int] { state.withLock { $0.latencies } }
        var lags: [Int] { state.withLock { $0.lags } }
        var passTimes: [Int] { state.withLock { $0.passTimes } }
        var englishText: String { state.withLock { $0.english.joined(separator: " ") } }
    }
}

extension LiveCaptioning {
    /// 計測用: エンジンの各パスの時間と、実時間からの遅れを記録する（showEvents の代わりに使う）
    func showPasses(_ clock: HeadlessRun.Clocked) async {
        await showEvents { event in
            if case .pass(let o) = event {
                clock.pass(ms: Int(o["ms"] as? Double ?? 0), heardSeconds: o["heard"] as? Double ?? 0)
            }
        }
    }
}
