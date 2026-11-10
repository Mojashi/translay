import Foundation

/// 起動設定。どれも必須で、欠けていたら起動時に止める（空文字や既定値で動き続けない）。
///
/// The .app gets these from Info.plist (written by scripts/build-app.sh); the headless runner
/// takes the same settings as command-line flags.
struct Configuration {
    let uv: URL
    let engineProject: URL
    let whisperModel: URL               // Whisper（確定した区間を捨てながら聞き直す）
    let whisperANEEncoder: URL          // Whisper のエンコーダの Core ML 版（Neural Engine で動かす）
    let translationModel: URL
    let targetLanguages: [String]       // 最初にオンにする翻訳先。例: ["en", "ja"]
    let termsFile: URL                  // 単語登録（1行1語）
    let recognitionLanguage: String     // 起動時の認識言語: "English" など、または "auto"

    private static let keys: [(plist: String, flag: String)] = [
        ("TranslayUV", "--uv"),
        ("TranslayEngineProject", "--engine-project"),
        ("TranslayTranslationModel", "--translation-model"),
        ("TranslayTargetLanguages", "--targets"),
        ("TranslayTermsFile", "--terms-file"),
        ("TranslayRecognitionLanguage", "--language"),
        ("TranslayWhisperModel", "--whisper-model"),
        ("TranslayWhisperANEEncoder", "--whisper-ane-encoder"),
    ]

    static func fromBundle(_ bundle: Bundle = .main) -> Configuration {
        build { key in
            guard let value = bundle.object(forInfoDictionaryKey: key.plist) as? String, !value.isEmpty else {
                fatalError("Info.plist に \(key.plist) がありません。scripts/build-app.sh でビルドしてください")
            }
            return value
        }
    }

    static func fromArguments(_ args: [String]) -> Configuration {
        build { key in
            guard let i = args.firstIndex(of: key.flag), i + 1 < args.count else {
                fatalError("引数 \(key.flag) <値> が必要です")
            }
            return args[i + 1]
        }
    }

    private static func build(_ value: ((plist: String, flag: String)) -> String) -> Configuration {
        let v = keys.map(value)
        let languages = v[3].split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !languages.isEmpty else { fatalError("翻訳先の言語が1つもありません: \"\(v[3])\"") }
        let configuration = Configuration(
            uv: URL(fileURLWithPath: v[0]), engineProject: URL(fileURLWithPath: v[1]),
            whisperModel: URL(fileURLWithPath: v[6]), whisperANEEncoder: URL(fileURLWithPath: v[7]),
            translationModel: URL(fileURLWithPath: v[2]),
            targetLanguages: languages, termsFile: URL(fileURLWithPath: v[4]),
            recognitionLanguage: v[5].trimmingCharacters(in: .whitespaces))
        configuration.validate()
        return configuration
    }

    private func validate() {
        for (what, url) in [("uv", uv), ("engine project", engineProject),
                            ("Whisper model", whisperModel), ("Whisper ANE encoder", whisperANEEncoder),
                            ("translation model", translationModel)] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                fatalError("\(what) が見つかりません: \(url.path)")
            }
        }
        if !FileManager.default.fileExists(atPath: termsFile.path) {
            try? FileManager.default.createDirectory(at: termsFile.deletingLastPathComponent(),
                                                     withIntermediateDirectories: true)
            let header = "# 単語登録: 1行に1語。聞き取りが寄せられます（例: Redis, orchestrator）。# で始まる行は無視。\n"
            guard FileManager.default.createFile(atPath: termsFile.path, contents: header.data(using: .utf8)) else {
                fatalError("単語登録ファイルを作れません: \(termsFile.path)")
            }
        }
    }

    /// Whisper's encoder runs on the Neural Engine (faster, frees the GPU). The engine keeps it in a
    /// supervised child process, because an ANE request was seen to hang in live use.
    func engineLaunch(language: String, trace: Bool) -> EngineProcess.Launch {
        .init(uv: uv, project: engineProject, recogniser: "whisper", asrModel: whisperModel,
              aneEncoder: whisperANEEncoder, translationModel: translationModel, targets: targetLanguages, termsFile: termsFile,
              language: language, trace: trace)
    }

    /// 単語登録ファイルの中身（1行1語、# 行と空行は除く）
    func registeredTerms() -> [String] {
        let text = (try? String(contentsOf: termsFile, encoding: .utf8)) ?? ""
        return text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }
}
