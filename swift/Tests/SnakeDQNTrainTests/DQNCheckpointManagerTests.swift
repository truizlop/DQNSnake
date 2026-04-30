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

private func temporaryDirectoryURL() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
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

    try manager.maybeSaveStepCheckpoint(saver: saver, globalStep: 1)
    let shouldNotExist = directory.appendingPathComponent("model_step_1.safetensors").path
    #expect(!FileManager.default.fileExists(atPath: shouldNotExist))

    try manager.maybeSaveStepCheckpoint(saver: saver, globalStep: 2)
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
    try manager.maybeSaveBestCheckpoint(
        saver: saver,
        episodeResult: DQNEpisodeResult(steps: 10, totalReward: 5, finalScore: 1),
        episode: 1,
        globalStep: 10
    )
    #expect(FileManager.default.fileExists(atPath: bestPath))

    let firstTimestamp = try FileManager.default.attributesOfItem(atPath: bestPath)[.modificationDate] as? Date

    // Lower score should not update best checkpoint.
    try manager.maybeSaveBestCheckpoint(
        saver: saver,
        episodeResult: DQNEpisodeResult(steps: 12, totalReward: 3, finalScore: 0),
        episode: 2,
        globalStep: 20
    )
    let secondTimestamp = try FileManager.default.attributesOfItem(atPath: bestPath)[.modificationDate] as? Date
    #expect(firstTimestamp == secondTimestamp)
}
