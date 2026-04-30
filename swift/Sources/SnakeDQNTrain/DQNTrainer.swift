import Foundation
import MLX
import SnakeEnv

struct DQNTrainer {
    let env: SnakeEnv
    var config: DQNTrainingConfig
    var replayBuffer: ReplayBuffer
    let learner: DQNLearner
    let explorationPolicy: EpsilonGreedyPolicy
    let episodeRunner: DQNEpisodeRunner
    var checkpointManager: DQNCheckpointManager

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.replayBuffer = ReplayBuffer(
            capacity: config.replayBufferCapacity,
            samplingStrategy: config.replaySamplingStrategy
        )
        self.learner = DQNLearner(gamma: config.gamma, learningRate: config.learningRate)
        self.explorationPolicy = EpsilonGreedyPolicy(
            epsilonStart: config.epsilonStart,
            epsilonEnd: config.epsilonEnd,
            epsilonDecaySteps: config.epsilonDecaySteps
        )
        self.episodeRunner = DQNEpisodeRunner(
            env: env,
            maxStepsPerEpisode: config.maxStepsPerEpisode
        )
        self.checkpointManager = DQNCheckpointManager(
            checkpointDirectory: config.checkpointDirectory,
            checkpointEverySteps: config.checkpointEverySteps,
            saveBestCheckpoint: config.saveBestCheckpoint
        )
    }

    mutating func run() async throws {
        var globalStep = try maybeResumeFromCheckpoint()
        var episode = 0

        while globalStep < config.totalEnvironmentSteps {
            episode += 1
            let episodeResult = try await episodeRunner.runEpisode(
                globalStep: &globalStep,
                totalEnvironmentSteps: config.totalEnvironmentSteps,
                selectAction: { state, step in
                    self.selectAction(state: state, globalStep: step)
                },
                onTransition: { transition, step in
                    self.replayBuffer.append(transition)
                    try self.applyTrainingSchedule(globalStep: step)
                }
            )

            try checkpointManager.maybeSaveBestCheckpoint(
                learner: learner,
                episodeResult: episodeResult,
                episode: episode,
                globalStep: globalStep
            )

            print(
                "episode=\(episode) steps=\(episodeResult.steps) reward=\(episodeResult.totalReward) score=\(episodeResult.finalScore) globalStep=\(globalStep)"
            )
        }
    }

    private func shouldOptimize(globalStep: Int) -> Bool {
        globalStep >= config.warmupSteps
            && globalStep % config.trainEvery == 0
            && replayBuffer.canSample(batchSize: config.batchSize)
    }

    private func shouldSyncTarget(globalStep: Int) -> Bool {
        config.targetSyncEvery > 0
            && globalStep > 0
            && globalStep % config.targetSyncEvery == 0
    }

    private mutating func applyTrainingSchedule(globalStep: Int) throws {
        if shouldOptimize(globalStep: globalStep) {
            optimizeFromReplay()
        }
        if shouldSyncTarget(globalStep: globalStep) {
            learner.syncTargetFromOnline()
        }
        try checkpointManager.maybeSaveStepCheckpoint(learner: learner, globalStep: globalStep)
    }

    private func selectAction(state: MLXArray, globalStep: Int) -> SnakeAction {
        explorationPolicy.selectAction(
            globalStep: globalStep,
            greedyAction: learner.greedyAction(for: state)
        )
    }

    private mutating func optimizeFromReplay() {
        let transitions = replayBuffer.sample(batchSize: config.batchSize)
        let batch = DQNBatch(transitions: transitions)
        _ = learner.trainStep(batch: batch)
    }

    private func maybeResumeFromCheckpoint() throws -> Int {
        guard let resumePath = config.resumeCheckpointPath, !resumePath.isEmpty else {
            return 0
        }

        let standardized = (resumePath as NSString).standardizingPath
        let url = URL(fileURLWithPath: standardized)
        let metadata = try learner.loadOnlineModel(from: url)
        let resumedStep = Int(metadata["global_step"] ?? "") ?? 0
        print("resumed from checkpoint=\(standardized) globalStep=\(resumedStep)")
        return resumedStep
    }
}
