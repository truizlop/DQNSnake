import Testing
@testable import SnakeDQNTrain

@Test func betaScheduleUsesBoundariesAndAnnealsToOne() {
    #expect(abs(PrioritizedReplayBetaSchedule.beta(globalStep: 0, start: 0.4, annealSteps: 100) - 0.4) < 1e-5)
    #expect(abs(PrioritizedReplayBetaSchedule.beta(globalStep: 50, start: 0.4, annealSteps: 100) - 0.7) < 1e-5)
    #expect(abs(PrioritizedReplayBetaSchedule.beta(globalStep: 100, start: 0.4, annealSteps: 100) - 1.0) < 1e-5)
    #expect(abs(PrioritizedReplayBetaSchedule.beta(globalStep: 200, start: 0.4, annealSteps: 100) - 1.0) < 1e-5)
    #expect(abs(PrioritizedReplayBetaSchedule.beta(globalStep: -10, start: 0.4, annealSteps: 100) - 0.4) < 1e-5)
}

@Test func betaScheduleWithNonPositiveAnnealStepsReturnsOne() {
    #expect(PrioritizedReplayBetaSchedule.beta(globalStep: 10, start: 0.4, annealSteps: 0) == 1.0)
    #expect(PrioritizedReplayBetaSchedule.beta(globalStep: 10, start: 0.4, annealSteps: -5) == 1.0)
}
