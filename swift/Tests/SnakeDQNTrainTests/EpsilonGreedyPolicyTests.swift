import Testing
@testable import SnakeDQNTrain

@Test func epsilonGreedyPolicyUsesGreedyActionWhenEpsilonIsZero() {
    var policy = EpsilonGreedyPolicy(
        epsilonStart: 0,
        epsilonEnd: 0,
        epsilonDecaySteps: 1
    )

    let greedy: SnakeAction = .left
    let selected = policy.selectAction(globalStep: 10_000, greedyAction: greedy)

    #expect(selected == greedy)
}

@Test func epsilonGreedyPolicyDecaysLinearlyWithinBounds() {
    let policy = EpsilonGreedyPolicy(
        epsilonStart: 1.0,
        epsilonEnd: 0.1,
        epsilonDecaySteps: 100
    )

    #expect(abs(policy.epsilon(at: 0) - 1.0) < 0.0001)
    #expect(abs(policy.epsilon(at: 50) - 0.55) < 0.0001)
    #expect(abs(policy.epsilon(at: 100) - 0.1) < 0.0001)
    #expect(abs(policy.epsilon(at: 200) - 0.1) < 0.0001)
}
