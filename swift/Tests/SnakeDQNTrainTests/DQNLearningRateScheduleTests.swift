import Testing
@testable import SnakeDQNTrain

@Test func learningRateScheduleKeepsInitialRateWithoutDecayConfig() {
    let rate = DQNLearningRateSchedule.learningRate(
        at: 15_000_000,
        initial: 2.5e-4,
        final: nil,
        decayStartStep: nil,
        decayEndStep: nil
    )

    #expect(rate == 2.5e-4)
}

@Test func learningRateScheduleLinearlyInterpolatesBetweenStartAndEndSteps() {
    let rate = DQNLearningRateSchedule.learningRate(
        at: 20_000_000,
        initial: 2.5e-4,
        final: 1.25e-4,
        decayStartStep: 15_000_000,
        decayEndStep: 25_000_000
    )

    #expect(abs(rate - 1.875e-4) < 0.0000001)
}

@Test func learningRateScheduleClampsBeforeAndAfterDecayWindow() {
    let before = DQNLearningRateSchedule.learningRate(
        at: 14_000_000,
        initial: 2.5e-4,
        final: 1.25e-4,
        decayStartStep: 15_000_000,
        decayEndStep: 25_000_000
    )
    let after = DQNLearningRateSchedule.learningRate(
        at: 26_000_000,
        initial: 2.5e-4,
        final: 1.25e-4,
        decayStartStep: 15_000_000,
        decayEndStep: 25_000_000
    )

    #expect(before == 2.5e-4)
    #expect(after == 1.25e-4)
}
