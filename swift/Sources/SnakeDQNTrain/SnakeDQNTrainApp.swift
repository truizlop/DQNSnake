import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import SnakeEnv

@main
struct SnakeDQNTrainApp {
    static func main() async {
        let pythonDir = ProcessInfo.processInfo.environment["SNAKE_PYTHON_DIR"]
        let steps = Int(ProcessInfo.processInfo.environment["SNAKE_STEPS"] ?? "20") ?? 20
        let maxEpisodeSteps = Int(ProcessInfo.processInfo.environment["SNAKE_MAX_EPISODE_STEPS"] ?? "2000") ?? 2_000

        // DQN training skeleton wired to SnakeEnv. Algorithm internals are TODOs.
        var trainer = DQNTrainer(
            env: SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir),
            config: DQNTrainingConfig(
                totalEnvironmentSteps: steps,
                maxStepsPerEpisode: maxEpisodeSteps
            )
        )

        do {
            try await trainer.run()
            print("snake-dqn-train skeleton run completed.")
        } catch {
            fputs("snake-dqn-train error: \(error)\n", stderr)
            exit(1)
        }
    }
}
