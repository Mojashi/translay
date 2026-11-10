import AppKit

// translay                      … メニューバー常駐の字幕アプリ（設定は Info.plist）
// translay --file <audio> …     … 画面なしでファイルを字幕化して遅延を測る（設定は引数）
let arguments = CommandLine.arguments

if arguments.contains("--file") {
    // The caption board lives on the main actor, so the main queue must keep running.
    Task.detached {
        do {
            try await HeadlessRun.run(arguments: arguments)
            exit(0)
        } catch {
            HeadlessRun.log("error: \(error)")
            exit(1)
        }
    }
    dispatchMain()
} else {
    let app = NSApplication.shared
    let delegate = MainActor.assumeIsolated { AppDelegate(configuration: Configuration.fromBundle()) }
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
}
