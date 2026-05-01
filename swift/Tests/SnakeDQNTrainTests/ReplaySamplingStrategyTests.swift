import Testing
@testable import SnakeDQNTrain

@Test func replaySamplingStrategyHasTwoExplicitModes() {
    let modes: [ReplaySamplingStrategy] = [.withReplacement, .withoutReplacement]
    #expect(modes.count == 2)
}
