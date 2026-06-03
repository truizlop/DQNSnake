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
    #expect(config.playActivationInspectionEnabled == false)
    #expect(config.playActivationDirectory == "runs/model_best_play/activations")
    #expect(config.playActivationStepMode == false)
    #expect(config.playActivationExportEverySteps == 1)
    #expect(config.playActivationDashboardEnabled == false)
    #expect(config.playStepIntervalSeconds == 2)
}
