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
        let pythonDir = ProcessInfo.processInfo.environment["SNAKE_PYTHON_DIR"]
        var config = DQNHyperparameterBaseline.snakeV1

        let steps = Int(ProcessInfo.processInfo.environment["SNAKE_STEPS"] ?? "") ?? config.totalEnvironmentSteps
        let replaySamplingStrategy = ReplaySamplingStrategy(
            envValue: ProcessInfo.processInfo.environment["SNAKE_REPLAY_SAMPLING_STRATEGY"] ?? ""
        ) ?? config.replaySamplingStrategy
        let prioritizedReplayAlpha =
            Float(ProcessInfo.processInfo.environment["SNAKE_PER_ALPHA"] ?? "") ?? config.prioritizedReplayAlpha
        let prioritizedReplayBetaStart =
            Float(ProcessInfo.processInfo.environment["SNAKE_PER_BETA_START"] ?? "")
            ?? config.prioritizedReplayBetaStart
        let prioritizedReplayBetaAnnealSteps =
            Int(ProcessInfo.processInfo.environment["SNAKE_PER_BETA_ANNEAL_STEPS"] ?? "")
            ?? config.prioritizedReplayBetaAnnealSteps
        let prioritizedReplayEpsilon =
            Float(ProcessInfo.processInfo.environment["SNAKE_PER_EPSILON"] ?? "") ?? config.prioritizedReplayEpsilon
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
        let warmupSteps = Int(ProcessInfo.processInfo.environment["SNAKE_WARMUP_STEPS"] ?? "") ?? config.warmupSteps
        let trainEvery = Int(ProcessInfo.processInfo.environment["SNAKE_TRAIN_EVERY"] ?? "") ?? config.trainEvery
        let targetSyncEvery =
            Int(ProcessInfo.processInfo.environment["SNAKE_TARGET_SYNC_EVERY"] ?? "") ?? config.targetSyncEvery
        let batchSize = Int(ProcessInfo.processInfo.environment["SNAKE_BATCH_SIZE"] ?? "") ?? config.batchSize
        let checkpointEverySteps =
            Int(ProcessInfo.processInfo.environment["SNAKE_CHECKPOINT_EVERY_STEPS"] ?? "")
            ?? config.checkpointEverySteps
        let checkpointDirectory =
            ProcessInfo.processInfo.environment["SNAKE_CHECKPOINT_DIR"] ?? config.checkpointDirectory
        let enableStructuredLogs =
            (ProcessInfo.processInfo.environment["SNAKE_OBS_ENABLE"] ?? (config.enableStructuredLogs ? "1" : "0"))
            == "1"
        let structuredLogPath = ProcessInfo.processInfo.environment["SNAKE_OBS_LOG_PATH"] ?? config.structuredLogPath
        let resourceTelemetryEverySteps =
            Int(ProcessInfo.processInfo.environment["SNAKE_RESOURCE_TELEMETRY_EVERY_STEPS"] ?? "")
            ?? config.resourceTelemetryEverySteps
        let enableBestEpisodeGIFCapture =
            (ProcessInfo.processInfo.environment["SNAKE_BEST_GIF_ENABLE"]
                ?? (config.enableBestEpisodeGIFCapture ? "1" : "0")) == "1"
        let bestEpisodeGIFDirectory =
            ProcessInfo.processInfo.environment["SNAKE_BEST_GIF_DIR"] ?? config.bestEpisodeGIFDirectory
        let bestEpisodeGIFScale =
            Int(ProcessInfo.processInfo.environment["SNAKE_BEST_GIF_SCALE"] ?? "") ?? config.bestEpisodeGIFScale
        let bestEpisodeGIFFrameDurationMs =
            Int(ProcessInfo.processInfo.environment["SNAKE_BEST_GIF_FRAME_MS"] ?? "")
            ?? config.bestEpisodeGIFFrameDurationMs
        let mlxDevice = (ProcessInfo.processInfo.environment["SNAKE_MLX_DEVICE"] ?? "cpu").lowercased()
        let trainingSeed = ProcessInfo.processInfo.environment["SNAKE_SEED"].flatMap(Int.init)
        let dqnAlgorithm = DQNAlgorithm(
            rawValue: (ProcessInfo.processInfo.environment["SNAKE_DQN_ALGORITHM"] ?? config.dqnAlgorithm.rawValue)
                .lowercased()
        ) ?? config.dqnAlgorithm
        let deviceType = mlxDevice == "gpu" ? MLX_GPU : MLX_CPU
        let device = mlx_device_new_type(deviceType, 0)
        mlx_set_default_device(device)
        if let trainingSeed {
            MLXRandom.seed(UInt64(bitPattern: Int64(trainingSeed)))
        }

        config.totalEnvironmentSteps = steps
        config.replaySamplingStrategy = replaySamplingStrategy
        config.prioritizedReplayAlpha = prioritizedReplayAlpha
        config.prioritizedReplayBetaStart = prioritizedReplayBetaStart
        config.prioritizedReplayBetaAnnealSteps = prioritizedReplayBetaAnnealSteps
        config.prioritizedReplayEpsilon = prioritizedReplayEpsilon
        config.maxStepsPerEpisode = maxEpisodeSteps
        config.resumeCheckpointPath = resumeCheckpointPath
        config.evalEveryEpisodes = evalEveryEpisodes
        config.evalEpisodes = evalEpisodes
        config.warmupSteps = warmupSteps
        config.trainEvery = trainEvery
        config.targetSyncEvery = targetSyncEvery
        config.batchSize = batchSize
        config.checkpointEverySteps = checkpointEverySteps
        config.checkpointDirectory = checkpointDirectory
        config.enableTensorBoard = enableTensorBoard
        config.launchTensorBoard = launchTensorBoard
        config.tensorBoardLogDir = tensorBoardLogDir
        config.tensorBoardPort = tensorBoardPort
        config.enableStructuredLogs = enableStructuredLogs
        config.structuredLogPath = structuredLogPath
        config.resourceTelemetryEverySteps = resourceTelemetryEverySteps
        config.enableBestEpisodeGIFCapture = enableBestEpisodeGIFCapture
        config.bestEpisodeGIFDirectory = bestEpisodeGIFDirectory
        config.bestEpisodeGIFScale = bestEpisodeGIFScale
        config.bestEpisodeGIFFrameDurationMs = bestEpisodeGIFFrameDurationMs
        config.seed = trainingSeed
        config.dqnAlgorithm = dqnAlgorithm

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
