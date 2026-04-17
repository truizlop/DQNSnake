// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SnakeEnv",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SnakeEnv", targets: ["SnakeEnv"]),
        .executable(name: "snake-env-cli", targets: ["SnakeEnvCLI"])
    ],
    targets: [
        .target(name: "SnakeEnv"),
        .executableTarget(
            name: "SnakeEnvCLI",
            dependencies: ["SnakeEnv"]
        ),
        .testTarget(name: "SnakeEnvTests", dependencies: ["SnakeEnv"])
    ]
)
