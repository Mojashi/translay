import AVFoundation
import TranslayCore

/// ライブ字幕: 音源の音をエンジンに渡し、エンジンからの出来事を字幕に出す。記録係がいれば残す。
///
/// Recognition, sentence cutting and translation all live in the engine; this type only connects
/// the source, the engine, the board and the recorder.
final class LiveCaptioning {
    private let engine: EngineProcess
    private let board: CaptionBoard
    private var recorder: SessionRecorder?
    private let onFinished: (LineID) -> Void
    /// エンジンの準備が終わったか（それまでは準備の段階と経過秒数を字幕板に出し続ける）
    private(set) var isReady = false
    private var startupStage = "エンジンを起動中"
    var listeningMessage = ""

    init(engine: EngineProcess, board: CaptionBoard, onFinished: @escaping (LineID) -> Void = { _ in }) {
        self.engine = engine
        self.board = board
        self.onFinished = onFinished
    }

    /// エンジンからの出来事を字幕に出し続ける（アプリの生存期間ずっと）。observe は計測用の覗き窓。
    func showEvents(observe: (EngineProcess.Event) -> Void = { _ in }) async {
        let started = ContinuousClock.now
        let ticker = Task { [weak self] in
            while let self, !self.isReady, !Task.isCancelled {
                let seconds = (ContinuousClock.now - started).components.seconds
                await self.board.say("\(self.startupStage)… \(seconds)秒")
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
        defer { ticker.cancel() }
        for await event in engine.events {
            observe(event)
            switch event {
            case .status(let stage):
                startupStage = stage
                if isReady { await board.say(stage) }        // after start-up: e.g. the ANE restarting
            case .ready:
                isReady = true
                ticker.cancel()
                await board.say(listeningMessage)
            case .speaking(let line, let text, let language):
                await recorder?.note(.speaking(text), line: line)
                await board.show(line, english: text, language: language, finished: false)
            case .finished(let line, let text, let language):
                onFinished(line)
                await recorder?.note(.finished(text), line: line)
                await board.show(line, english: text, language: language, finished: true)
            case .translation(let rendering):
                await recorder?.note(rendering)
                await board.render(rendering)
            case .pass:
                break
            case .exited(let status):
                await board.say("エンジンが終了しました (exit \(status))")
            }
        }
    }

    /// 音源が止まるまで、音をエンジンに渡し続ける
    func run(_ source: AudioSource, recordingTo recorder: SessionRecorder? = nil) async throws {
        self.recorder = recorder
        let resampler = Resampler()
        let audioLog = EngineProcess.logFile("app.log")
        var sent = 0, peak: Float = 0, windowStart = ContinuousClock.now
        // However the source ends — exhausted, stopped, or failed — the sentence in progress is
        // finished and the recording closed.
        defer {
            engine.endOfAudio()
            Task { await recorder?.close() }
        }
        for await sound in try await source.sounds() {
            try Task.checkCancellation()
            try await recorder?.record(sound)
            let pcm = resampler.convert(sound)
            engine.hear(pcm)
            // Once a second: how much audio went to the engine, and how loud it was.
            sent += pcm.count
            peak = max(peak, pcm.reduce(0) { max($0, abs($1)) })
            if ContinuousClock.now - windowStart >= .seconds(1) {
                let line = String(format: "%@ sent %.2fs (in %d frames @%.0f Hz) peak=%.3f\n",
                                  "\(Date())", Double(sent) / 16_000, Int(sound.frameLength),
                                  sound.format.sampleRate, peak)
                audioLog.write(line.data(using: .utf8)!)
                sent = 0
                peak = 0
                windowStart = .now
            }
        }
    }
}
