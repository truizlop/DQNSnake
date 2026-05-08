import Testing
@testable import SnakeDQNTrain

@Test func trainingConfigDefaultsUsePracticalDecaySchedules() {
    let config = DQNTrainingConfig()
    #expect(config.epsilonDecaySteps == 100_000)
    #expect(config.prioritizedReplayBetaAnnealSteps == 200_000)
}
