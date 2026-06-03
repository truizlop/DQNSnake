import Cmlx
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import SnakeEnv

@main
struct SnakeDQNTrainApp {
    static func main() async {
        let env = ProcessInfo.processInfo.environment
        let pythonDir = env["SNAKE_PYTHON_DIR"]
        let config = SnakeDQNTrainEnvironmentParser.parseConfig(
            env: env,
            base: DQNHyperparameterBaseline.snakeV1
        )
        let mlxDevice = (env["SNAKE_MLX_DEVICE"] ?? "cpu").lowercased()
        let deviceType = mlxDevice == "gpu" ? MLX_GPU : MLX_CPU
        let device = mlx_device_new_type(deviceType, 0)
        mlx_set_default_device(device)
        if let trainingSeed = config.seed {
            MLXRandom.seed(UInt64(bitPattern: Int64(trainingSeed)))
        }

        var trainer = DQNTrainer(
            env: SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir, seed: config.seed),
            config: config
        )

        do {
            if config.evalOnly {
                _ = try await trainer.runEvaluationOnly()
            } else {
                try await trainer.run()
                print("snake-dqn-train skeleton run completed.")
            }
        } catch {
            fputs("snake-dqn-train error: \(error)\n", stderr)
            exit(1)
        }
    }
}
