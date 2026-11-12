#!/usr/bin/env bash
# Build Translay.app (menu-bar caption overlay).
#
#     scripts/build-app.sh en,ja English   # translation targets switched on at first launch; spoken language
#     scripts/build-app.sh en,ja auto      # … detect the spoken language
#
# Recognition is Whisper large-v3-turbo with its encoder on the Neural Engine, or Qwen3-ASR 1.7B on
# the GPU; the menu switches between them.
#
# Paths the app needs at run time are written into Info.plist; the app refuses to start if any is
# missing. Output: app/build/Translay.app
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

TARGETS="${1:?usage: scripts/build-app.sh <targets, e.g. en,ja> <English|auto|…>}"
LANGUAGE="${2:?usage: scripts/build-app.sh <targets, e.g. en,ja> <English|auto|…>}"
UV="$(command -v uv)" || { echo "uv not found on PATH" >&2; exit 1; }
WHISPER_MODEL="$ROOT/models/whisper-large-v3-turbo"
WHISPER_ANE_ENCODER="$ROOT/models/whisperkit/openai_whisper-large-v3-v20240930/AudioEncoder.mlmodelc"
QWEN_MODEL="$ROOT/models/Qwen3-ASR-1.7B-4bit"
MT_MODEL="$ROOT/models/plamo-3-translate-0.8b-MLX-4bit"
TERMS_FILE="$HOME/Library/Application Support/Translay/terms.txt"
for p in "$WHISPER_MODEL" "$WHISPER_ANE_ENCODER" "$QWEN_MODEL" "$MT_MODEL" "$ROOT/translator"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

(cd app && swift build -c release --product translay)

APP="$ROOT/app/build/Translay.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp app/.build/release/translay "$APP/Contents/MacOS/translay"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>com.mojashi.translay</string>
  <key>CFBundleName</key><string>Translay</string>
  <key>CFBundleExecutable</key><string>translay</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>話している英語を文字起こしして翻訳するためにマイクを使います。</string>
  <key>TranslayUV</key><string>$UV</string>
  <key>TranslayEngineProject</key><string>$ROOT/translator</string>
  <key>TranslayWhisperModel</key><string>$WHISPER_MODEL</string>
  <key>TranslayWhisperANEEncoder</key><string>$WHISPER_ANE_ENCODER</string>
  <key>TranslayQwenModel</key><string>$QWEN_MODEL</string>
  <key>TranslayTranslationModel</key><string>$MT_MODEL</string>
  <key>TranslayTargetLanguages</key><string>$TARGETS</string>
  <key>TranslayTermsFile</key><string>$TERMS_FILE</string>
  <key>TranslayRecognitionLanguage</key><string>$LANGUAGE</string>
</dict>
</plist>
PLIST

# A fixed local certificate, so macOS keeps the permissions across rebuilds (see signing-identity.sh).
KEYCHAIN="$("$ROOT/scripts/signing-identity.sh")"
# By its SHA-1: a self-signed certificate is "not trusted", and codesign only resolves trusted ones by name.
IDENTITY="$(security find-identity -p codesigning "$KEYCHAIN" | awk '/Translay Local Signing/ {print $2; exit}')"
# codesign finds the private key only through the keychain search list: add ours just while signing.
ORIGINAL_KEYCHAINS=$(security list-keychains -d user | tr -d '"')
security list-keychains -d user -s $ORIGINAL_KEYCHAINS "$KEYCHAIN"
codesign --force --deep --keychain "$KEYCHAIN" --sign "$IDENTITY" "$APP" || SIGN_FAILED=1
security list-keychains -d user -s $ORIGINAL_KEYCHAINS
[ -z "${SIGN_FAILED:-}" ] || { echo "codesign failed" >&2; exit 1; }

echo "built $APP (targets: $TARGETS, recognition: $LANGUAGE)"
