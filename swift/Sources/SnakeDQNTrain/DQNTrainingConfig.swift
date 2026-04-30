import Foundation

struct DQNTrainingConfig {
    // Overall control
    var totalEnvironmentSteps: Int = 200_000
    var maxStepsPerEpisode: Int = 10_000
    var replayBufferCapacity: Int = 100_000
    var replaySamplingStrategy: ReplaySamplingStrategy = .withReplacement
    var checkpointDirectory: String = "checkpoints"
    var checkpointEverySteps: Int = 10_000
    var saveBestCheckpoint: Bool = true
    var resumeCheckpointPath: String? = nil
    var evalEveryEpisodes: Int = 0
    var evalEpisodes: Int = 5
    var enableTensorBoard: Bool = true
    var launchTensorBoard: Bool = true
    var tensorBoardLogDir: String = "runs/snake_dqn"
    var tensorBoardPort: Int = 6006
    var enableStructuredLogs: Bool = true
    var structuredLogPath: String = "runs/snake_dqn/observability.jsonl"

    // DQN scheduling knobs
    var warmupSteps: Int = 5_000
    var trainEvery: Int = 4
    var targetSyncEvery: Int = 10_000
    var batchSize: Int = 32
    var gamma: Float = 0.99
    var learningRate: Float = 2.5e-4

    // Epsilon-greedy schedule
    var epsilonStart: Float = 1.0
    var epsilonEnd: Float = 0.1
    var epsilonDecaySteps: Int = 1_000_000
}
