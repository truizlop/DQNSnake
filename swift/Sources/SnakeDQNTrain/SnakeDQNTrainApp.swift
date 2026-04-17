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
        let enableMLX = (ProcessInfo.processInfo.environment["SNAKE_ENABLE_MLX"] ?? "0") == "1"

        // This target is intentionally a scaffold: wiring, shape contracts, and
        // conversion points are here so DQN implementation can be added incrementally.
        let trainer = DQNTrainerScaffold(
            env: SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir),
            enableMLX: enableMLX
        )

        do {
            try await trainer.smokeRollout(steps: steps)
            print("snake-dqn-train scaffold completed.")
        } catch {
            fputs("snake-dqn-train error: \(error)\n", stderr)
            exit(1)
        }
    }
}
