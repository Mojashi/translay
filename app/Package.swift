// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Translay",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "translay", targets: ["Translay"]),
    ],
    dependencies: [
        // Only the ASR comparison tool uses it (bench). Patched local checkout, see
        // scripts/vendor-fluidaudio.sh: upstream names visionOS v2 / tvOS, which CLT Swift 6.0 cannot parse.
        .package(path: "../vendor/FluidAudio"),
    ],
    targets: [
        .target(
            name: "TranslayCore",
            path: "Sources/TranslayCore",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "Translay",
            dependencies: ["TranslayCore"],
            path: "Sources/Translay",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "TranslayASREval",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            path: "Sources/TranslayASREval",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
