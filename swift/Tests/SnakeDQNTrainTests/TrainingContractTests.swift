import Foundation
import Testing

@Test func evaluatorPathUsesGreedyPolicyClosureContract() throws {
    let trainerPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Sources/SnakeDQNTrain/DQNTrainer.swift")
    let source = try String(contentsOf: trainerPath)

    #expect(source.contains("selectAction: { state in"))
    #expect(source.contains("learner.greedyAction(for: state)"))
}
