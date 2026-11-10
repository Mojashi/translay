#!/bin/bash
# この Mac 専用の署名用証明書（自己署名）を、専用のキーチェーンに一度だけ作る。
#
# Why: ad-hoc signing ("codesign -s -") makes the app's identity its code hash, so every rebuild is a
# different app to macOS and the Screen Recording / microphone permissions never carry over. Signed
# with a fixed certificate, the designated requirement becomes "this bundle id + this certificate",
# which stays the same across builds.
#
# Prints the keychain path. The keychain has an empty password: it holds only this local-use key.
set -euo pipefail
NAME="Translay Local Signing"
KEYCHAIN="$HOME/Library/Keychains/translay-signing.keychain-db"
if [ ! -f "$KEYCHAIN" ]; then
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
  /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj "/CN=$NAME" \
    -keyout "$WORK/key.pem" -out "$WORK/cert.pem" \
    -addext "keyUsage=critical,digitalSignature" -addext "extendedKeyUsage=critical,codeSigning" >/dev/null 2>&1
  /usr/bin/openssl pkcs12 -export -inkey "$WORK/key.pem" -in "$WORK/cert.pem" -name "$NAME" \
    -passout pass:translay -out "$WORK/id.p12"
  security create-keychain -p "" "$KEYCHAIN"
  security set-keychain-settings "$KEYCHAIN"            # never auto-lock
  security unlock-keychain -p "" "$KEYCHAIN"
  security import "$WORK/id.p12" -k "$KEYCHAIN" -P translay -T /usr/bin/codesign >/dev/null
  security set-key-partition-list -S apple-tool:,apple: -s -k "" "$KEYCHAIN" >/dev/null
fi
security unlock-keychain -p "" "$KEYCHAIN"
echo "$KEYCHAIN"
