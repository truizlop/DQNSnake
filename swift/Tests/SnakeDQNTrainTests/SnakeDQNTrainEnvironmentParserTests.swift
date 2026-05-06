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
            "SNAKE_EPSILON_END": "0.03",
            "SNAKE_EPSILON_DECAY_STEPS": "300000",
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
    #expect(parsed.epsilonEnd == 0.03)
    #expect(parsed.epsilonDecaySteps == 300_000)
    #expect(parsed.seed == 99)
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
