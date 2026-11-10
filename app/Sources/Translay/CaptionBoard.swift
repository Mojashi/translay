import Foundation
import TranslayCore

/// 字幕板: 直近の数行（英語の原文と各言語の訳）を持つ。表示はこれを見るだけ。
@MainActor
final class CaptionBoard: ObservableObject {
    struct Line: Identifiable, Equatable {
        let id: LineID
        var english: String             // the spoken text, in whatever language it was spoken
        var spokenLanguage: String?     // its language code when detected ("en", "ja", …)
        var englishFinished: Bool
        var translations: Translations
        var translationsFinal: Bool
    }

    @Published private(set) var lines: [Line] = []
    @Published private(set) var status: String = ""
    @Published var languages: [String]  // 訳を出す言語（表示順）
    private let keep: Int
    /// Called whenever a line's translation becomes final (used by the headless runner for logs).
    var onFinalTranslation: ((Line) -> Void)?

    init(languages: [String], keep: Int = 2) {
        self.languages = languages
        self.keep = keep
    }

    /// 英語の文を表示する（話し途中 or 言い終えた）
    func show(_ id: LineID, english: String, language: String? = nil, finished: Bool) {
        if let i = lines.firstIndex(where: { $0.id == id }) {
            lines[i].english = english
            lines[i].spokenLanguage = language
            lines[i].englishFinished = finished
        } else {
            lines.append(Line(id: id, english: english, spokenLanguage: language, englishFinished: finished,
                              translations: [:], translationsFinal: false))
            if lines.count > keep { lines.removeFirst(lines.count - keep) }
        }
    }

    /// 訳を表示する（下書き or 確定）
    func render(_ rendering: Rendering) {
        guard let i = lines.firstIndex(where: { $0.id == rendering.line }) else { return }
        if lines[i].translationsFinal { return }    // a late draft must not overwrite the final
        lines[i].translations = rendering.translations
        lines[i].translationsFinal = rendering.isFinal
        if rendering.isFinal { onFinalTranslation?(lines[i]) }
    }

    func say(_ message: String) {
        status = message
    }

    func clear() {
        lines.removeAll()
    }
}
