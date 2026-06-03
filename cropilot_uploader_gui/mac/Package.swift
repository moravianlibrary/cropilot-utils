// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "CropilotUploader",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "CropilotUploader", targets: ["CropilotUploader"])
    ],
    targets: [
        .executableTarget(
            name: "CropilotUploader",
            path: "Sources"
        )
    ]
)
