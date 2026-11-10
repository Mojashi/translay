import AppKit

/// メニューバー常駐アプリ: エンジンを立ち上げ、選んだ音源の字幕を出し続ける。単語登録はファイルで編集する。
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    enum SourceChoice: String, CaseIterable {
        case microphone = "マイク"
        case systemAudio = "システム音声"

        func make() -> AudioSource {
            switch self {
            case .microphone: return MicrophoneSource()
            case .systemAudio: return SystemAudioSource()
            }
        }
    }

    private let configuration: Configuration
    private let board: CaptionBoard
    private var panel: OverlayPanel?
    private var statusItem: NSStatusItem?

    private var captioning: LiveCaptioning?
    private var engine: EngineProcess?
    private var termsWatch: Timer?
    private var termsStamp: Date?
    private var source: AudioSource?
    private var running: Task<Void, Never>?
    private var choice: SourceChoice = .microphone
    private var recording = false
    private lazy var recognitionLanguage = configuration.recognitionLanguage
    /// 翻訳先に選べる言語: PLaMo 3 Translate が訳せる 53 言語（メニューと字幕の表示順）。
    private static let targetChoices = [
        "ja", "en", "zh-hans", "zh-hant", "ko", "es", "fr", "de", "it", "ru", "ar", "hi", "id", "ms",
        "th", "vi", "tl", "nl", "pl", "fa", "he", "hu", "bg", "hr", "sr", "sk", "sl", "lt", "lv", "et", "be", "ca",
        "eu", "hy", "ka", "az", "kk", "uz", "bn", "gu", "mr", "ml", "ne", "pa", "si", "ta", "te", "ur", "km", "my",
        "jv", "ceb", "sw",
    ]
    /// 翻訳先のうちオンにしている言語（次回起動にも引き継ぐ）。
    private lazy var enabledTargets: Set<String> = {
        guard let stored = UserDefaults.standard.stringArray(forKey: Self.enabledTargetsKey) else {
            return Set(configuration.targetLanguages)      // never chosen yet: what the app was built with
        }
        return Set(stored).intersection(Self.targetChoices)
    }()
    private static let enabledTargetsKey = "TranslayEnabledTargets"
    private var targetsInOrder: [String] { Self.targetChoices.filter(enabledTargets.contains) }
    /// 認識言語の選択肢: 自動判定と、よく使う 30 言語（値はエンジンに渡す英語名）。
    private static let languageChoices: [(title: String, value: String)] = [
        ("自動判定", "auto"),
        ("英語", "English"), ("日本語", "Japanese"), ("中国語", "Chinese"), ("広東語", "Cantonese"),
        ("韓国語", "Korean"), ("スペイン語", "Spanish"), ("フランス語", "French"), ("ドイツ語", "German"),
        ("イタリア語", "Italian"), ("ポルトガル語", "Portuguese"), ("ロシア語", "Russian"),
        ("アラビア語", "Arabic"), ("ヒンディー語", "Hindi"), ("インドネシア語", "Indonesian"),
        ("マレー語", "Malay"), ("タイ語", "Thai"), ("ベトナム語", "Vietnamese"), ("フィリピン語", "Filipino"),
        ("トルコ語", "Turkish"), ("ペルシア語", "Persian"), ("オランダ語", "Dutch"), ("スウェーデン語", "Swedish"),
        ("デンマーク語", "Danish"), ("フィンランド語", "Finnish"), ("ポーランド語", "Polish"),
        ("チェコ語", "Czech"), ("ギリシャ語", "Greek"), ("ルーマニア語", "Romanian"),
        ("ハンガリー語", "Hungarian"), ("マケドニア語", "Macedonian"),
    ]
    private let sessionsDirectory = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Translay/sessions", isDirectory: true)

    init(configuration: Configuration) {
        self.configuration = configuration
        self.board = CaptionBoard(languages: configuration.targetLanguages)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        board.languages = targetsInOrder
        panel = OverlayPanel(board: board)
        panel?.orderFrontRegardless()
        buildMenu()
        board.say("モデルを読み込み中…")
        Task { await loadModelsAndStart() }
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine?.terminate()
    }

    /// エンジンを立ち上げ、すぐ聞き始める（モデルが載るまでの音はエンジンが貯めておく）。
    private func loadModelsAndStart() async {
        let engine = EngineProcess(configuration.engineLaunch(
            language: recognitionLanguage, trace: false))
        self.engine = engine
        engine.translate(into: targetsInOrder)
        let captioning = LiveCaptioning(engine: engine, board: board)
        self.captioning = captioning
        Task { await captioning.showEvents() }
        if termsWatch == nil { watchTerms() }
        startCaptioning()
    }

    // MARK: 単語登録

    /// 単語登録ファイルが保存されたらエンジンに反映する（エディタは保存時に置き換えることがあるので更新時刻で見る）
    private func watchTerms() {
        termsStamp = modificationDate(configuration.termsFile)
        termsWatch = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let stamp = self.modificationDate(self.configuration.termsFile)
                guard stamp != self.termsStamp else { return }
                self.termsStamp = stamp
                let terms = self.configuration.registeredTerms()
                self.engine?.register(terms: terms)
                self.board.say("単語登録を反映しました（\(terms.count)語）")
            }
        }
    }

    private func modificationDate(_ url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }

    // MARK: 字幕の開始・停止

    private func startCaptioning() {
        guard let captioning, running == nil else { return }
        let source = choice.make()
        self.source = source
        let recorder: SessionRecorder?
        do {
            recorder = recording ? try SessionRecorder.start(in: sessionsDirectory) : nil
        } catch {
            board.say("記録を開始できません: \(error)")
            return
        }
        // Until the engine is ready the board shows its start-up stage instead.
        captioning.listeningMessage = "\(source.name) を聞いています" + (recording ? "（記録中）" : "")
        if captioning.isReady { board.say(captioning.listeningMessage) }
        running = Task { [board] in
            do {
                try await captioning.run(source, recordingTo: recorder)
            } catch is CancellationError {
                await recorder?.close()
            } catch {
                await MainActor.run { board.say("\(source.name): \(error)") }
            }
        }
        refreshMenu()
    }

    private func stopCaptioning() async {
        running?.cancel()
        await source?.stop()
        _ = await running?.value
        running = nil
        source = nil
        refreshMenu()
    }

    // MARK: メニュー

    private func buildMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "訳"
        refreshMenu()
    }

    private func refreshMenu() {
        let menu = NSMenu()
        let header = NSMenuItem(title: "音源", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        for option in SourceChoice.allCases {
            let item = NSMenuItem(title: option.rawValue, action: #selector(chooseSource(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = option.rawValue
            item.state = option == choice ? .on : .off
            menu.addItem(item)
        }
        menu.addItem(.separator())
        let current = Self.languageChoices.first {
            $0.value.caseInsensitiveCompare(recognitionLanguage) == .orderedSame
        }?.title ?? recognitionLanguage
        let languageItem = NSMenuItem(title: "認識言語: \(current)", action: nil, keyEquivalent: "")
        let languages = NSMenu()
        for choice in Self.languageChoices {
            let item = NSMenuItem(title: choice.title, action: #selector(chooseLanguage(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = choice.value
            item.state = choice.value.caseInsensitiveCompare(recognitionLanguage) == .orderedSame ? .on : .off
            languages.addItem(item)
            if choice.value == "auto" { languages.addItem(.separator()) }
        }
        languageItem.submenu = languages
        menu.addItem(languageItem)
        let targetsItem = NSMenuItem(title: "翻訳先: " + (targetsInOrder.isEmpty ? "なし（原文のみ）"
                                                        : targetsInOrder.map(Self.languageName).joined(separator: "・")),
                                     action: nil, keyEquivalent: "")
        let targets = NSMenu()
        for code in Self.targetChoices {
            let item = NSMenuItem(title: Self.languageName(code), action: #selector(toggleTarget(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = code
            item.state = enabledTargets.contains(code) ? .on : .off
            targets.addItem(item)
        }
        targetsItem.submenu = targets
        menu.addItem(targetsItem)
        menu.addItem(.separator())
        let toggle = NSMenuItem(title: running == nil ? "字幕を開始" : "字幕を停止",
                                action: #selector(toggleCaptioning), keyEquivalent: "")
        toggle.target = self
        toggle.isEnabled = captioning != nil
        menu.addItem(toggle)
        let clear = NSMenuItem(title: "字幕を消去", action: #selector(clearBoard), keyEquivalent: "")
        clear.target = self
        menu.addItem(clear)
        menu.addItem(.separator())
        let record = NSMenuItem(title: "記録する（音声と字幕を保存）", action: #selector(toggleRecording),
                                keyEquivalent: "")
        record.target = self
        record.state = recording ? .on : .off
        menu.addItem(record)
        let reveal = NSMenuItem(title: "記録フォルダを開く", action: #selector(revealSessions), keyEquivalent: "")
        reveal.target = self
        menu.addItem(reveal)
        menu.addItem(.separator())
        let terms = NSMenuItem(title: "単語登録を編集…", action: #selector(editTerms), keyEquivalent: "")
        terms.target = self
        menu.addItem(terms)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "終了", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)
        statusItem?.menu = menu
    }

    @objc private func chooseSource(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let next = SourceChoice(rawValue: raw),
              next != choice else { return }
        choice = next
        Task {
            let wasRunning = running != nil
            await stopCaptioning()
            if wasRunning { startCaptioning() } else { refreshMenu() }
        }
    }

    @objc private func toggleCaptioning() {
        if running == nil {
            startCaptioning()
        } else {
            Task {
                await stopCaptioning()
                board.say("停止中")
            }
        }
    }

    @objc private func clearBoard() {
        board.clear()
    }

    /// 記録のオン・オフ（字幕の途中なら、次の区切りからではなく今すぐ新しいセッションで録り直す）
    @objc private func toggleRecording() {
        recording.toggle()
        Task {
            let wasRunning = running != nil
            await stopCaptioning()
            if wasRunning { startCaptioning() } else { refreshMenu() }
        }
    }

    @objc private func chooseLanguage(_ sender: NSMenuItem) {
        guard let value = sender.representedObject as? String else { return }
        recognitionLanguage = value
        engine?.recognise(language: value)
        board.say("認識言語: \(sender.title)")
        refreshMenu()
    }

    /// 翻訳先の言語をオン・オフする（エンジンは動かしたまま、次の文から効く）
    @objc private func toggleTarget(_ sender: NSMenuItem) {
        guard let code = sender.representedObject as? String else { return }
        enabledTargets.formSymmetricDifference([code])
        UserDefaults.standard.set(targetsInOrder, forKey: Self.enabledTargetsKey)
        engine?.translate(into: targetsInOrder)
        board.languages = targetsInOrder
        refreshMenu()
    }

    private static func languageName(_ code: String) -> String {
        Locale(identifier: "ja").localizedString(forIdentifier: code) ?? code
    }

    @objc private func editTerms() {
        NSWorkspace.shared.open(configuration.termsFile)
    }

    @objc private func revealSessions() {
        try? FileManager.default.createDirectory(at: sessionsDirectory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(sessionsDirectory)
    }
}
