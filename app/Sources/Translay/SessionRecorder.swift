import AVFoundation
import Foundation
import TranslayCore

/// 記録係: 聞いた音声と、字幕に出した出来事（英文・訳）を時系列でファイルに残す。
///
/// One directory per session:
///   audio.caf     — the source audio exactly as captured (native rate/format)
///   events.jsonl  — {"t": <seconds of audio>, "kind": "speaking"|"finished"|"draft"|"final", ...}
/// The audio clock `t` lets events be lined up with the recording when replaying offline.
actor SessionRecorder {
    let directory: URL
    private var audio: AVAudioFile?
    private let events: FileHandle
    private var heard = 0.0

    static func start(in root: URL) throws -> SessionRecorder {
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd-HHmmss"
        let directory = root.appendingPathComponent(stamp.string(from: Date()), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let eventsURL = directory.appendingPathComponent("events.jsonl")
        FileManager.default.createFile(atPath: eventsURL.path, contents: nil)
        return SessionRecorder(directory: directory, events: try FileHandle(forWritingTo: eventsURL))
    }

    private init(directory: URL, events: FileHandle) {
        self.directory = directory
        self.events = events
    }

    /// 聞いた音を残す
    func record(_ sound: AVAudioPCMBuffer) throws {
        if audio == nil {
            audio = try AVAudioFile(forWriting: directory.appendingPathComponent("audio.caf"),
                                    settings: sound.format.settings,
                                    commonFormat: sound.format.commonFormat,
                                    interleaved: sound.format.isInterleaved)
        }
        try audio?.write(from: sound)
        heard += sound.duration
    }

    /// 英文の出来事を残す
    func note(_ utterance: Utterance, line: LineID) {
        switch utterance {
        case .speaking(let english): write(["kind": "speaking", "line": line.value, "en": english])
        case .finished(let english): write(["kind": "finished", "line": line.value, "en": english])
        }
    }

    /// 訳の出来事を残す
    func note(_ rendering: Rendering) {
        write(["kind": rendering.isFinal ? "final" : "draft", "line": rendering.line.value,
               "translations": rendering.translations])
    }

    func close() {
        audio = nil          // AVAudioFile finalises on deinit
        try? events.close()
    }

    private func write(_ fields: [String: Any]) {
        var object = fields
        object["t"] = (heard * 1000).rounded() / 1000
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return }
        events.write(data)
        events.write(Data([0x0A]))
    }
}
