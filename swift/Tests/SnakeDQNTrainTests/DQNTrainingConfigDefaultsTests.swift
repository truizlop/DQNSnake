import Testing
@testable import SnakeDQNTrain

@Test func trainingConfigDefaultsUsePracticalDecaySchedules() {
    let config = DQNTrainingConfig()
    #expect(config.epsilonDecaySteps == 100_000)
    #expect(config.prioritizedReplayBetaAnnealSteps == 200_000)
}

@Test func trainingConfigDefaultsUseConvenientPlaySettings() {
    let config = DQNTrainingConfig()
    #expect(config.playEpisodes == 1)
    #expect(config.playRender)
    #expect(config.playGIFDirectory == "runs/model_best_play")
}
