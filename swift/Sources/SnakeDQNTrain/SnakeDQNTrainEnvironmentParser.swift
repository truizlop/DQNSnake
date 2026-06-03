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
        let explicitResumeCheckpointPath = env["SNAKE_RESUME_CHECKPOINT"]
        config.resumeCheckpointPath = explicitResumeCheckpointPath
        config.enableTensorBoard = (env["SNAKE_TB_ENABLE"] ?? (config.enableTensorBoard ? "1" : "0")) == "1"
        config.launchTensorBoard = (env["SNAKE_TB_LAUNCH"] ?? (config.launchTensorBoard ? "1" : "0")) == "1"
        config.tensorBoardLogDir = env["SNAKE_TB_LOGDIR"] ?? config.tensorBoardLogDir
        config.tensorBoardPort = Int(env["SNAKE_TB_PORT"] ?? "") ?? config.tensorBoardPort
        config.evalEveryEpisodes = Int(env["SNAKE_EVAL_EVERY_EPISODES"] ?? "") ?? config.evalEveryEpisodes
        config.evalEpisodes = Int(env["SNAKE_EVAL_EPISODES"] ?? "") ?? config.evalEpisodes
        config.evalRollingWindow = Int(env["SNAKE_EVAL_ROLLING_WINDOW"] ?? "") ?? config.evalRollingWindow
        config.evalFixedSeeds = parseIntList(env["SNAKE_EVAL_SEEDS"])
        config.warmupSteps = Int(env["SNAKE_WARMUP_STEPS"] ?? "") ?? config.warmupSteps
        config.trainEvery = Int(env["SNAKE_TRAIN_EVERY"] ?? "") ?? config.trainEvery
        config.gradientClipNorm = Float(env["SNAKE_GRAD_CLIP_NORM"] ?? "") ?? config.gradientClipNorm
        config.learningRate = Float(env["SNAKE_LEARNING_RATE"] ?? "") ?? config.learningRate
        config.learningRateFinal = env["SNAKE_LEARNING_RATE_FINAL"].flatMap(Float.init) ?? config.learningRateFinal
        config.learningRateDecayStartStep =
            env["SNAKE_LEARNING_RATE_DECAY_START_STEP"].flatMap(Int.init) ?? config.learningRateDecayStartStep
        config.learningRateDecayEndStep =
            env["SNAKE_LEARNING_RATE_DECAY_END_STEP"].flatMap(Int.init) ?? config.learningRateDecayEndStep
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
        config.evalOnly = (env["SNAKE_EVAL_ONLY"] ?? (config.evalOnly ? "1" : "0")) == "1"
        config.playOnly = (env["SNAKE_PLAY_ONLY"] ?? (config.playOnly ? "1" : "0")) == "1"
        config.playEpisodes = Int(env["SNAKE_PLAY_EPISODES"] ?? "") ?? config.playEpisodes
        config.playRender = (env["SNAKE_PLAY_RENDER"] ?? (config.playRender ? "1" : "0")) == "1"
        config.playGIFDirectory = env["SNAKE_PLAY_GIF_DIR"] ?? config.playGIFDirectory
        if config.playOnly {
            config.resumeCheckpointPath = explicitResumeCheckpointPath ?? "checkpoints/model_best.safetensors"
            config.enableTensorBoard = false
            config.launchTensorBoard = false
            config.enableStructuredLogs = false
            config.enableBestEpisodeGIFCapture = false
        }
        config.dqnAlgorithm = DQNAlgorithm(
            rawValue: (env["SNAKE_DQN_ALGORITHM"] ?? config.dqnAlgorithm.rawValue).lowercased()
        ) ?? config.dqnAlgorithm
        return config
    }

    private static func parseIntList(_ raw: String?) -> [Int]? {
        guard let raw, !raw.isEmpty else { return nil }
        let values = raw
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        return values.isEmpty ? nil : values
    }
}
