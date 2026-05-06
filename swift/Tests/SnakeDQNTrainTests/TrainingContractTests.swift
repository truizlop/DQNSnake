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

@Test func ablationHarnessDefinesAllThreePlannedRuns() throws {
    let scriptPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("scripts/run_ablation_20260506.sh")
    let content = try String(contentsOf: scriptPath)
    #expect(content.contains("ablation_a_eps003_train2"))
    #expect(content.contains("ablation_b_eps001_train1"))
    #expect(content.contains("ablation_c_alpha07"))
}
