/// 字幕の語彙: 行、英文の出来事、各言語の訳。

/// 訳文: 言語コード（"es", "ja" …）ごとの訳。
public typealias Translations = [String: String]

/// 字幕の1行（1文）。
public struct LineID: Hashable, Comparable, Sendable, CustomStringConvertible {
    public let value: Int
    public init(_ value: Int) { self.value = value }
    public static func < (a: LineID, b: LineID) -> Bool { a.value < b.value }
    public var description: String { "#\(value)" }
}

/// 英文の出来事: 話し途中か、言い終えたか。
public enum Utterance: Equatable, Sendable {
    case speaking(String)
    case finished(String)
}

/// ある行の訳（下書き or 確定）。
public struct Rendering: Sendable {
    public let line: LineID
    public let translations: Translations
    public let isFinal: Bool

    public init(line: LineID, translations: Translations, isFinal: Bool) {
        self.line = line
        self.translations = translations
        self.isFinal = isFinal
    }
}
