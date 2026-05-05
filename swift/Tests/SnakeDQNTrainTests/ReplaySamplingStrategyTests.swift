import Testing
@testable import SnakeDQNTrain

@Test func replaySamplingStrategyHasThreeExplicitModes() {
    let modes: [ReplaySamplingStrategy] = [.withReplacement, .withoutReplacement, .prioritized]
    #expect(modes.count == 3)
}

@Test func replaySamplingStrategyParsesEnvValues() {
    #expect(ReplaySamplingStrategy(envValue: "with_replacement") == .withReplacement)
    #expect(ReplaySamplingStrategy(envValue: "without_replacement") == .withoutReplacement)
    #expect(ReplaySamplingStrategy(envValue: "prioritized") == .prioritized)
    #expect(ReplaySamplingStrategy(envValue: "per") == .prioritized)
    #expect(ReplaySamplingStrategy(envValue: "unknown") == nil)
}
