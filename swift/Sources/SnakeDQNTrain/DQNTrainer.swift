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
    }

    mutating func run() async throws {
        var globalStep = 0
        var episode = 0

        while globalStep < config.totalEnvironmentSteps {
            episode += 1
            let stepsInEpisode = try await episodeRunner.runEpisode(
                globalStep: &globalStep,
                totalEnvironmentSteps: config.totalEnvironmentSteps,
                selectAction: { state, step in
                    self.selectAction(state: state, globalStep: step)
                },
                onTransition: { transition, step in
                    self.replayBuffer.append(transition)
                    self.applyTrainingSchedule(globalStep: step)
                }
            )

            print("episode=\(episode) steps=\(stepsInEpisode) globalStep=\(globalStep)")
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

    private mutating func applyTrainingSchedule(globalStep: Int) {
        if shouldOptimize(globalStep: globalStep) {
            optimizeFromReplay()
        }
        if shouldSyncTarget(globalStep: globalStep) {
            learner.syncTargetFromOnline()
        }
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
}
