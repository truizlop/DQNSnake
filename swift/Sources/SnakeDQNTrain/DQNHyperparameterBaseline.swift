import Foundation

enum DQNHyperparameterBaseline {
    // Baseline tuned for stable first training runs on Snake.
    static let snakeV1 = DQNTrainingConfig(
        totalEnvironmentSteps: 200_000,
        maxStepsPerEpisode: 2_000,
        replayBufferCapacity: 100_000,
        replaySamplingStrategy: .withReplacement,
        checkpointDirectory: "checkpoints",
        checkpointEverySteps: 10_000,
        saveBestCheckpoint: true,
        resumeCheckpointPath: nil,
        evalEveryEpisodes: 25,
        evalEpisodes: 5,
        enableTensorBoard: true,
        launchTensorBoard: true,
        tensorBoardLogDir: "runs/snake_dqn",
        tensorBoardPort: 6006,
        enableStructuredLogs: true,
        structuredLogPath: "runs/snake_dqn/observability.jsonl",
        seed: nil,
        warmupSteps: 5_000,
        trainEvery: 4,
        targetSyncEvery: 10_000,
        batchSize: 32,
        gamma: 0.99,
        learningRate: 2.5e-4,
        maxConsecutiveSkippedUpdates: 500,
        maxLossForUpdate: 1_000_000,
        maxAbsQValue: 1_000_000,
        maxGradientL2Norm: 1_000_000,
        epsilonStart: 1.0,
        epsilonEnd: 0.1,
        epsilonDecaySteps: 100_000
    )
}
