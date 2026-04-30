// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SnakeEnv",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SnakeEnv", targets: ["SnakeEnv"]),
        .executable(name: "snake-env-cli", targets: ["SnakeEnvCLI"]),
        .executable(name: "snake-dqn-train", targets: ["SnakeDQNTrain"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0")
    ],
    targets: [
        .target(name: "SnakeEnv"),
        .executableTarget(
            name: "SnakeEnvCLI",
            dependencies: ["SnakeEnv"]
        ),
        .executableTarget(
            name: "SnakeDQNTrain",
            dependencies: [
                "SnakeEnv",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ]
        ),
        .testTarget(name: "SnakeEnvTests", dependencies: ["SnakeEnv"]),
        .testTarget(name: "SnakeDQNTrainTests", dependencies: ["SnakeDQNTrain"])
    ]
)
