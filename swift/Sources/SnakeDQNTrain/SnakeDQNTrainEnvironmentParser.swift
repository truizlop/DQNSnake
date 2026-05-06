import Foundation

enum SnakeDQNTrainEnvironmentParser {
    static func parseConfig(
        env: [String: String],
        base: DQNTrainingConfig
    ) -> DQNTrainingConfig {
        var config = base

        config.totalEnvironmentSteps = Int(env["SNAKE_STEPS"] ?? "") ?? config.totalEnvironmentSteps
        config.replaySamplingStrategy = ReplaySamplingStrategy(
            envValue: env["SNAKE_REPLAY_SAMPLING_STRATEGY"] ?? ""
        ) ?? config.replaySamplingStrategy
        config.prioritizedReplayAlpha = Float(env["SNAKE_PER_ALPHA"] ?? "") ?? config.prioritizedReplayAlpha
        config.prioritizedReplayBetaStart =
            Float(env["SNAKE_PER_BETA_START"] ?? "") ?? config.prioritizedReplayBetaStart
        config.prioritizedReplayBetaAnnealSteps =
            Int(env["SNAKE_PER_BETA_ANNEAL_STEPS"] ?? "") ?? config.prioritizedReplayBetaAnnealSteps
        config.prioritizedReplayEpsilon = Float(env["SNAKE_PER_EPSILON"] ?? "") ?? config.prioritizedReplayEpsilon
        config.maxStepsPerEpisode = Int(env["SNAKE_MAX_EPISODE_STEPS"] ?? "") ?? config.maxStepsPerEpisode
        config.resumeCheckpointPath = env["SNAKE_RESUME_CHECKPOINT"]
        config.enableTensorBoard = (env["SNAKE_TB_ENABLE"] ?? (config.enableTensorBoard ? "1" : "0")) == "1"
        config.launchTensorBoard = (env["SNAKE_TB_LAUNCH"] ?? (config.launchTensorBoard ? "1" : "0")) == "1"
        config.tensorBoardLogDir = env["SNAKE_TB_LOGDIR"] ?? config.tensorBoardLogDir
        config.tensorBoardPort = Int(env["SNAKE_TB_PORT"] ?? "") ?? config.tensorBoardPort
        config.evalEveryEpisodes = Int(env["SNAKE_EVAL_EVERY_EPISODES"] ?? "") ?? config.evalEveryEpisodes
        config.evalEpisodes = Int(env["SNAKE_EVAL_EPISODES"] ?? "") ?? config.evalEpisodes
        config.warmupSteps = Int(env["SNAKE_WARMUP_STEPS"] ?? "") ?? config.warmupSteps
        config.trainEvery = Int(env["SNAKE_TRAIN_EVERY"] ?? "") ?? config.trainEvery
        config.epsilonStart = Float(env["SNAKE_EPSILON_START"] ?? "") ?? config.epsilonStart
        config.epsilonEnd = Float(env["SNAKE_EPSILON_END"] ?? "") ?? config.epsilonEnd
        config.epsilonDecaySteps = Int(env["SNAKE_EPSILON_DECAY_STEPS"] ?? "") ?? config.epsilonDecaySteps
        config.targetSyncEvery = Int(env["SNAKE_TARGET_SYNC_EVERY"] ?? "") ?? config.targetSyncEvery
        config.batchSize = Int(env["SNAKE_BATCH_SIZE"] ?? "") ?? config.batchSize
        config.checkpointEverySteps = Int(env["SNAKE_CHECKPOINT_EVERY_STEPS"] ?? "") ?? config.checkpointEverySteps
        config.checkpointDirectory = env["SNAKE_CHECKPOINT_DIR"] ?? config.checkpointDirectory
        config.enableStructuredLogs = (env["SNAKE_OBS_ENABLE"] ?? (config.enableStructuredLogs ? "1" : "0")) == "1"
        config.structuredLogPath = env["SNAKE_OBS_LOG_PATH"] ?? config.structuredLogPath
        config.resourceTelemetryEverySteps =
            Int(env["SNAKE_RESOURCE_TELEMETRY_EVERY_STEPS"] ?? "") ?? config.resourceTelemetryEverySteps
        config.enableBestEpisodeGIFCapture =
            (env["SNAKE_BEST_GIF_ENABLE"] ?? (config.enableBestEpisodeGIFCapture ? "1" : "0")) == "1"
        config.bestEpisodeGIFDirectory = env["SNAKE_BEST_GIF_DIR"] ?? config.bestEpisodeGIFDirectory
        config.bestEpisodeGIFScale = Int(env["SNAKE_BEST_GIF_SCALE"] ?? "") ?? config.bestEpisodeGIFScale
        config.bestEpisodeGIFFrameDurationMs =
            Int(env["SNAKE_BEST_GIF_FRAME_MS"] ?? "") ?? config.bestEpisodeGIFFrameDurationMs
        config.seed = env["SNAKE_SEED"].flatMap(Int.init)
        config.dqnAlgorithm = DQNAlgorithm(
            rawValue: (env["SNAKE_DQN_ALGORITHM"] ?? config.dqnAlgorithm.rawValue).lowercased()
        ) ?? config.dqnAlgorithm
        return config
    }
}
