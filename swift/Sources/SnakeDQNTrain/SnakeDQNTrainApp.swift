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
        var config = DQNHyperparameterBaseline.snakeV1

        let steps = Int(ProcessInfo.processInfo.environment["SNAKE_STEPS"] ?? "") ?? config.totalEnvironmentSteps
        let maxEpisodeSteps =
            Int(ProcessInfo.processInfo.environment["SNAKE_MAX_EPISODE_STEPS"] ?? "") ?? config.maxStepsPerEpisode
        let resumeCheckpointPath = ProcessInfo.processInfo.environment["SNAKE_RESUME_CHECKPOINT"]
        let enableTensorBoard =
            (ProcessInfo.processInfo.environment["SNAKE_TB_ENABLE"] ?? (config.enableTensorBoard ? "1" : "0")) == "1"
        let launchTensorBoard =
            (ProcessInfo.processInfo.environment["SNAKE_TB_LAUNCH"] ?? (config.launchTensorBoard ? "1" : "0")) == "1"
        let tensorBoardLogDir = ProcessInfo.processInfo.environment["SNAKE_TB_LOGDIR"] ?? config.tensorBoardLogDir
        let tensorBoardPort = Int(ProcessInfo.processInfo.environment["SNAKE_TB_PORT"] ?? "") ?? config.tensorBoardPort
        let evalEveryEpisodes =
            Int(ProcessInfo.processInfo.environment["SNAKE_EVAL_EVERY_EPISODES"] ?? "") ?? config.evalEveryEpisodes
        let evalEpisodes = Int(ProcessInfo.processInfo.environment["SNAKE_EVAL_EPISODES"] ?? "") ?? config.evalEpisodes
        let enableStructuredLogs =
            (ProcessInfo.processInfo.environment["SNAKE_OBS_ENABLE"] ?? (config.enableStructuredLogs ? "1" : "0"))
            == "1"
        let structuredLogPath = ProcessInfo.processInfo.environment["SNAKE_OBS_LOG_PATH"] ?? config.structuredLogPath
        let trainingSeed = ProcessInfo.processInfo.environment["SNAKE_SEED"].flatMap(Int.init)
        if let trainingSeed {
            MLXRandom.seed(UInt64(bitPattern: Int64(trainingSeed)))
        }

        config.totalEnvironmentSteps = steps
        config.maxStepsPerEpisode = maxEpisodeSteps
        config.resumeCheckpointPath = resumeCheckpointPath
        config.evalEveryEpisodes = evalEveryEpisodes
        config.evalEpisodes = evalEpisodes
        config.enableTensorBoard = enableTensorBoard
        config.launchTensorBoard = launchTensorBoard
        config.tensorBoardLogDir = tensorBoardLogDir
        config.tensorBoardPort = tensorBoardPort
        config.enableStructuredLogs = enableStructuredLogs
        config.structuredLogPath = structuredLogPath
        config.seed = trainingSeed

        var trainer = DQNTrainer(
            env: SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir, seed: trainingSeed),
            config: config
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
