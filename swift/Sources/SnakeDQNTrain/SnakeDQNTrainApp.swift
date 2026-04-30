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
        let resumeCheckpointPath = ProcessInfo.processInfo.environment["SNAKE_RESUME_CHECKPOINT"]
        let enableTensorBoard = (ProcessInfo.processInfo.environment["SNAKE_TB_ENABLE"] ?? "1") == "1"
        let launchTensorBoard = (ProcessInfo.processInfo.environment["SNAKE_TB_LAUNCH"] ?? "1") == "1"
        let tensorBoardLogDir = ProcessInfo.processInfo.environment["SNAKE_TB_LOGDIR"] ?? "runs/snake_dqn"
        let tensorBoardPort = Int(ProcessInfo.processInfo.environment["SNAKE_TB_PORT"] ?? "6006") ?? 6006
        let evalEveryEpisodes = Int(ProcessInfo.processInfo.environment["SNAKE_EVAL_EVERY_EPISODES"] ?? "0") ?? 0
        let evalEpisodes = Int(ProcessInfo.processInfo.environment["SNAKE_EVAL_EPISODES"] ?? "5") ?? 5
        let enableStructuredLogs = (ProcessInfo.processInfo.environment["SNAKE_OBS_ENABLE"] ?? "1") == "1"
        let structuredLogPath = ProcessInfo.processInfo.environment["SNAKE_OBS_LOG_PATH"] ?? "runs/snake_dqn/observability.jsonl"

        var trainer = DQNTrainer(
            env: SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir),
            config: DQNTrainingConfig(
                totalEnvironmentSteps: steps,
                maxStepsPerEpisode: maxEpisodeSteps,
                resumeCheckpointPath: resumeCheckpointPath,
                evalEveryEpisodes: evalEveryEpisodes,
                evalEpisodes: evalEpisodes,
                enableTensorBoard: enableTensorBoard,
                launchTensorBoard: launchTensorBoard,
                tensorBoardLogDir: tensorBoardLogDir,
                tensorBoardPort: tensorBoardPort,
                enableStructuredLogs: enableStructuredLogs,
                structuredLogPath: structuredLogPath
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
