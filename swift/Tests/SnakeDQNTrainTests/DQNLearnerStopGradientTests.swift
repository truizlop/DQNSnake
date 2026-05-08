import Foundation
import Testing

@Test func learnerSourceUsesStopGradientForNextStateTargets() throws {
    let learnerPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources/SnakeDQNTrain/DQNLearner.swift")
    let source = try String(contentsOf: learnerPath)
    #expect(source.contains("stopGradient(model(nextStates))"))
    #expect(source.contains("stopGradient(targetQNetwork(nextStates))"))
    #expect(source.contains("stopGradient(rewards + gamma * notDoneMask * maxNextQ)"))
}
