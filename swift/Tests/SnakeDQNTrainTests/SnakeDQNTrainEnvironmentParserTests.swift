import Testing
@testable import SnakeDQNTrain

@Test func environmentParserAppliesOverrides() {
    let base = DQNHyperparameterBaseline.snakeV1
    let parsed = SnakeDQNTrainEnvironmentParser.parseConfig(
        env: [
            "SNAKE_STEPS": "123",
            "SNAKE_REPLAY_SAMPLING_STRATEGY": "prioritized",
            "SNAKE_PER_ALPHA": "0.7",
            "SNAKE_PER_BETA_START": "0.5",
            "SNAKE_PER_BETA_ANNEAL_STEPS": "200000",
            "SNAKE_PER_EPSILON": "0.002",
            "SNAKE_DQN_ALGORITHM": "single",
            "SNAKE_LEARNING_RATE": "0.0003",
            "SNAKE_LEARNING_RATE_FINAL": "0.0001",
            "SNAKE_LEARNING_RATE_DECAY_START_STEP": "15000000",
            "SNAKE_LEARNING_RATE_DECAY_END_STEP": "25000000",
            "SNAKE_EPSILON_END": "0.03",
            "SNAKE_EPSILON_DECAY_STEPS": "300000",
            "SNAKE_EVAL_ROLLING_WINDOW": "30",
            "SNAKE_EVAL_SEEDS": "11, 22,33",
            "SNAKE_EVAL_ONLY": "1",
            "SNAKE_PLAY_ONLY": "1",
            "SNAKE_PLAY_EPISODES": "2",
            "SNAKE_PLAY_RENDER": "0",
            "SNAKE_PLAY_GIF_DIR": "runs/custom_play",
            "SNAKE_SEED": "99",
        ],
        base: base
    )

    #expect(parsed.totalEnvironmentSteps == 123)
    #expect(parsed.replaySamplingStrategy == .prioritized)
    #expect(parsed.prioritizedReplayAlpha == 0.7)
    #expect(parsed.prioritizedReplayBetaStart == 0.5)
    #expect(parsed.prioritizedReplayBetaAnnealSteps == 200_000)
    #expect(parsed.prioritizedReplayEpsilon == 0.002)
    #expect(parsed.dqnAlgorithm == .single)
    #expect(parsed.learningRate == 0.0003)
    #expect(parsed.learningRateFinal == 0.0001)
    #expect(parsed.learningRateDecayStartStep == 15_000_000)
    #expect(parsed.learningRateDecayEndStep == 25_000_000)
    #expect(parsed.epsilonEnd == 0.03)
    #expect(parsed.epsilonDecaySteps == 300_000)
    #expect(parsed.evalRollingWindow == 30)
    #expect(parsed.evalFixedSeeds == [11, 22, 33])
    #expect(parsed.evalOnly)
    #expect(parsed.playOnly)
    #expect(parsed.playEpisodes == 2)
    #expect(parsed.playRender == false)
    #expect(parsed.playGIFDirectory == "runs/custom_play")
    #expect(parsed.resumeCheckpointPath == "checkpoints/model_best.safetensors")
    #expect(parsed.enableTensorBoard == false)
    #expect(parsed.launchTensorBoard == false)
    #expect(parsed.enableStructuredLogs == false)
    #expect(parsed.enableBestEpisodeGIFCapture == false)
    #expect(parsed.seed == 99)
}

@Test func environmentParserPreservesExplicitPlayCheckpointOverride() {
    let parsed = SnakeDQNTrainEnvironmentParser.parseConfig(
        env: [
            "SNAKE_PLAY_ONLY": "1",
            "SNAKE_RESUME_CHECKPOINT": "checkpoints/custom.safetensors",
        ],
        base: DQNHyperparameterBaseline.snakeV1
    )

    #expect(parsed.resumeCheckpointPath == "checkpoints/custom.safetensors")
}

@Test func environmentParserFallsBackOnInvalidValues() {
    let base = DQNHyperparameterBaseline.snakeV1
    let parsed = SnakeDQNTrainEnvironmentParser.parseConfig(
        env: [
            "SNAKE_REPLAY_SAMPLING_STRATEGY": "invalid",
            "SNAKE_DQN_ALGORITHM": "invalid",
            "SNAKE_PER_ALPHA": "not-a-number",
            "SNAKE_SEED": "bad",
        ],
        base: base
    )

    #expect(parsed.replaySamplingStrategy == base.replaySamplingStrategy)
    #expect(parsed.dqnAlgorithm == base.dqnAlgorithm)
    #expect(parsed.prioritizedReplayAlpha == base.prioritizedReplayAlpha)
    #expect(parsed.seed == nil)
}
