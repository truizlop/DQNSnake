import Foundation

protocol DQNModelCheckpointSaving {
    func saveOnlineModel(to url: URL, metadata: [String: String]) throws
}

struct DQNCheckpointManager {
    private let directoryURL: URL
    private let checkpointEverySteps: Int
    private let saveBestCheckpoint: Bool
    private var bestScore: Float?

    init(
        checkpointDirectory: String,
        checkpointEverySteps: Int,
        saveBestCheckpoint: Bool
    ) {
        self.directoryURL = URL(fileURLWithPath: checkpointDirectory, isDirectory: true)
        self.checkpointEverySteps = checkpointEverySteps
        self.saveBestCheckpoint = saveBestCheckpoint
        self.bestScore = nil
    }

    mutating func maybeSaveStepCheckpoint(
        saver: some DQNModelCheckpointSaving,
        globalStep: Int
    ) throws -> Bool {
        guard checkpointEverySteps > 0, globalStep > 0, globalStep % checkpointEverySteps == 0 else {
            return false
        }
        try createDirectoryIfNeeded()
        let url = directoryURL.appendingPathComponent("model_step_\(globalStep).safetensors")
        try saver.saveOnlineModel(
            to: url,
            metadata: ["global_step": "\(globalStep)", "kind": "periodic"]
        )
        return true
    }

    mutating func maybeSaveBestCheckpoint(
        saver: some DQNModelCheckpointSaving,
        metricValue: Float,
        metricName: String,
        metadata: [String: String]
    ) throws -> Bool {
        guard
            saveBestCheckpoint,
            bestScore == nil || metricValue > bestScore!
        else {
            return false
        }
        bestScore = metricValue
        try createDirectoryIfNeeded()
        let url = directoryURL.appendingPathComponent("model_best.safetensors")
        var mergedMetadata = metadata
        mergedMetadata["kind"] = "best"
        mergedMetadata["best_metric_name"] = metricName
        mergedMetadata["best_metric_value"] = "\(metricValue)"
        try saver.saveOnlineModel(
            to: url,
            metadata: mergedMetadata
        )
        return true
    }

    private func createDirectoryIfNeeded() throws {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }
}
