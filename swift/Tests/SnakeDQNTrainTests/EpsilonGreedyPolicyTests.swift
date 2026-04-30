import Testing
@testable import SnakeDQNTrain

@Test func epsilonGreedyPolicyUsesGreedyActionWhenEpsilonIsZero() {
    let policy = EpsilonGreedyPolicy(
        epsilonStart: 0,
        epsilonEnd: 0,
        epsilonDecaySteps: 1
    )

    let greedy: SnakeAction = .left
    let selected = policy.selectAction(globalStep: 10_000, greedyAction: greedy)

    #expect(selected == greedy)
}
