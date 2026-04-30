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
    ) throws {
        guard checkpointEverySteps > 0, globalStep > 0, globalStep % checkpointEverySteps == 0 else {
            return
        }
        try createDirectoryIfNeeded()
        let url = directoryURL.appendingPathComponent("model_step_\(globalStep).safetensors")
        try saver.saveOnlineModel(
            to: url,
            metadata: ["global_step": "\(globalStep)", "kind": "periodic"]
        )
    }

    mutating func maybeSaveBestCheckpoint(
        saver: some DQNModelCheckpointSaving,
        episodeResult: DQNEpisodeResult,
        episode: Int,
        globalStep: Int
    ) throws {
        guard
            saveBestCheckpoint,
            bestScore == nil || episodeResult.finalScore > bestScore!
        else {
            return
        }
        bestScore = episodeResult.finalScore
        try createDirectoryIfNeeded()
        let url = directoryURL.appendingPathComponent("model_best.safetensors")
        try saver.saveOnlineModel(
            to: url,
            metadata: [
                "global_step": "\(globalStep)",
                "episode": "\(episode)",
                "kind": "best",
                "best_score": "\(episodeResult.finalScore)",
                "total_reward": "\(episodeResult.totalReward)",
            ]
        )
    }

    private func createDirectoryIfNeeded() throws {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }
}
