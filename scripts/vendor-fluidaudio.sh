#!/usr/bin/env bash
# Fetch FluidAudio v0.18.2 into vendor/ and patch its manifest for the Command Line Tools' Swift 6.0
# (upstream lists visionOS v2 / tvOS platforms that this PackageDescription cannot parse).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d vendor/FluidAudio ]; then
  git clone --depth 1 --branch v0.18.2 https://github.com/FluidInference/FluidAudio.git vendor/FluidAudio
fi
sed -i '' -e '/\.visionOS(\.v2)/d' -e '/\.tvOS(\.v17)/d' vendor/FluidAudio/Package.swift
echo "vendor/FluidAudio ready"
