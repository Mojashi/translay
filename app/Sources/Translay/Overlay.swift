import AppKit
import SwiftUI

/// 字幕パネル: 全スペース・フルスクリーンの上に浮かび、フォーカスを奪わない半透明の帯。
final class OverlayPanel: NSPanel {
    init(board: CaptionBoard) {
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let width = min(1100, screen.width * 0.8)
        let height: CGFloat = 300
        let frame = NSRect(x: screen.midX - width / 2, y: screen.minY + 40, width: width, height: height)
        super.init(contentRect: frame, styleMask: [.borderless, .nonactivatingPanel, .resizable],
                   backing: .buffered, defer: false)
        level = .statusBar
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        isMovableByWindowBackground = true
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        hidesOnDeactivate = false
        contentView = NSHostingView(rootView: CaptionView(board: board))
    }

    override var canBecomeKey: Bool { false }
}

struct CaptionView: View {
    @ObservedObject var board: CaptionBoard

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(board.lines) { line in
                VStack(alignment: .leading, spacing: 2) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text((line.spokenLanguage ?? "").uppercased())
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.35))
                            .frame(width: 22, alignment: .leading)
                        Text(line.english)
                            .font(.system(size: 15))
                            .foregroundStyle(.white.opacity(line.englishFinished ? 0.65 : 0.85))
                            .lineLimit(2)
                            .truncationMode(.head)     // too long to fit: keep the newest words, drop the start
                    }
                    // A target in the language that was spoken would only repeat the original line.
                    ForEach(board.languages.filter { $0 != line.spokenLanguage }, id: \.self) { language in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(language.uppercased())
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.45))
                                .frame(width: 22, alignment: .leading)
                            Text(line.translations[language] ?? " ")
                                .font(.system(size: 21, weight: .semibold))
                                .foregroundStyle(.white.opacity(line.translationsFinal ? 1.0 : 0.6))
                                .lineLimit(3)
                                .truncationMode(.head)     // too long to fit: keep the newest words, drop the start
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if !board.status.isEmpty {
                Text(board.status)
                    .font(.system(size: 12))
                    .foregroundStyle(.yellow.opacity(0.9))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
        .background(RoundedRectangle(cornerRadius: 14).fill(.black.opacity(0.62)))
        .animation(.easeOut(duration: 0.12), value: board.lines)
    }
}
