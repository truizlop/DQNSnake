import Foundation
import Testing
@testable import SnakeDQNTrain

private struct MockCheckpointSaver: DQNModelCheckpointSaving {
    func saveOnlineModel(to url: URL, metadata: [String: String]) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(metadata.description.utf8).write(to: url)
    }
}

private final class CapturingCheckpointSaver: DQNModelCheckpointSaving {
    var lastMetadata: [String: String] = [:]
    func saveOnlineModel(to url: URL, metadata: [String: String]) throws {
        lastMetadata = metadata
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(metadata.description.utf8).write(to: url)
    }
}

private func temporaryDirectoryURL() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
}

@Test func checkpointManagerPersistsProvidedMetadataForBestCheckpoint() throws {
    let directory = temporaryDirectoryURL()
    defer { try? FileManager.default.removeItem(at: directory) }

    let saver = CapturingCheckpointSaver()
    var manager = DQNCheckpointManager(
        checkpointDirectory: directory.path,
        checkpointEverySteps: 0,
        saveBestCheckpoint: true
    )

    let metadata = [
        "global_step": "12345",
        "replay_sampling_strategy": "prioritized",
        "dqn_algorithm": "double",
    ]
    _ = try manager.maybeSaveBestCheckpoint(
        saver: saver,
        metricValue: 1.2,
        metricName: "eval_avg_score",
        metadata: metadata
    )

    #expect(saver.lastMetadata["global_step"] == "12345")
    #expect(saver.lastMetadata["replay_sampling_strategy"] == "prioritized")
    #expect(saver.lastMetadata["dqn_algorithm"] == "double")
    #expect(saver.lastMetadata["kind"] == "best")
    #expect(saver.lastMetadata["best_metric_name"] == "eval_avg_score")
    #expect(saver.lastMetadata["best_metric_value"] == "1.2")
}

@Test func checkpointManagerSavesPeriodicStepCheckpoints() throws {
    let directory = temporaryDirectoryURL()
    defer { try? FileManager.default.removeItem(at: directory) }

    let saver = MockCheckpointSaver()
    var manager = DQNCheckpointManager(
        checkpointDirectory: directory.path,
        checkpointEverySteps: 2,
        saveBestCheckpoint: false
    )

    _ = try manager.maybeSaveStepCheckpoint(saver: saver, globalStep: 1)
    let shouldNotExist = directory.appendingPathComponent("model_step_1.safetensors").path
    #expect(!FileManager.default.fileExists(atPath: shouldNotExist))

    _ = try manager.maybeSaveStepCheckpoint(saver: saver, globalStep: 2)
    let shouldExist = directory.appendingPathComponent("model_step_2.safetensors").path
    #expect(FileManager.default.fileExists(atPath: shouldExist))
}

@Test func checkpointManagerSavesBestOnlyWhenScoreImproves() throws {
    let directory = temporaryDirectoryURL()
    defer { try? FileManager.default.removeItem(at: directory) }

    let saver = MockCheckpointSaver()
    var manager = DQNCheckpointManager(
        checkpointDirectory: directory.path,
        checkpointEverySteps: 0,
        saveBestCheckpoint: true
    )

    let bestPath = directory.appendingPathComponent("model_best.safetensors").path
    _ = try manager.maybeSaveBestCheckpoint(
        saver: saver,
        metricValue: 1,
        metricName: "eval_avg_score",
        metadata: ["global_step": "10", "episode": "1"]
    )
    #expect(FileManager.default.fileExists(atPath: bestPath))

    let firstTimestamp = try FileManager.default.attributesOfItem(atPath: bestPath)[.modificationDate] as? Date

    // Lower score should not update best checkpoint.
    _ = try manager.maybeSaveBestCheckpoint(
        saver: saver,
        metricValue: 0,
        metricName: "eval_avg_score",
        metadata: ["global_step": "20", "episode": "2"]
    )
    let secondTimestamp = try FileManager.default.attributesOfItem(atPath: bestPath)[.modificationDate] as? Date
    #expect(firstTimestamp == secondTimestamp)
}
