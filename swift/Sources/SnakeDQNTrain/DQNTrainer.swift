import Foundation
import MLX
import SnakeEnv

struct DQNTrainer {
    let env: SnakeEnv
    var config: DQNTrainingConfig
    var replayBuffer: ReplayBuffer
    let learner: DQNLearner
    let explorationPolicy: EpsilonGreedyPolicy

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.replayBuffer = ReplayBuffer(capacity: config.replayBufferCapacity)
        self.learner = DQNLearner(gamma: config.gamma, learningRate: config.learningRate)
        self.explorationPolicy = EpsilonGreedyPolicy(
            epsilonStart: config.epsilonStart,
            epsilonEnd: config.epsilonEnd,
            epsilonDecaySteps: config.epsilonDecaySteps
        )
    }

    mutating func run() async throws {
        var globalStep = 0
        var episode = 0

        while globalStep < config.totalEnvironmentSteps {
            episode += 1
            var stepsInEpisode = 0

            let initial = try await env.reset()
            var frameStack = FrameStack(capacity: 4)
            frameStack.reset(frame: initial.data)
            var state = stateTensor(from: frameStack.stacked(), width: initial.width, height: initial.height)

            while globalStep < config.totalEnvironmentSteps && stepsInEpisode < config.maxStepsPerEpisode {
                let action = selectAction(state: state, globalStep: globalStep)
                let stepResult = try await env.step(action: action.rawValue)

                frameStack.append(stepResult.observation)
                let nextState = stateTensor(
                    from: frameStack.stacked(),
                    width: initial.width,
                    height: initial.height
                )

                let transition = DQNTransition(
                    state: state,
                    action: action,
                    reward: stepResult.reward,
                    nextState: nextState,
                    done: stepResult.done
                )

                replayBuffer.append(transition)

                maybeOptimizeFromReplay(globalStep: globalStep)

                maybeSyncTargetNetwork(globalStep: globalStep)

                state = nextState
                globalStep += 1
                stepsInEpisode += 1

                if stepResult.done {
                    break
                }
            }

            print("episode=\(episode) steps=\(stepsInEpisode) globalStep=\(globalStep)")
        }
    }

    private func stateTensor(from stackedFrames: [UInt8], width: Int, height: Int) -> MLXArray {
        // Snake observations are binary (0/1). Convert to float and shape as NCHW:
        // [batch=1, channels=4, height, width].
        let values = stackedFrames.map { Float($0) }
        let array = MLXArray(values)
        return array.reshaped(1, 4, height, width)
    }

    private func selectAction(state: MLXArray, globalStep: Int) -> SnakeAction {
        explorationPolicy.selectAction(
            globalStep: globalStep,
            greedyAction: learner.greedyAction(for: state)
        )
    }

    private func maybeOptimizeFromReplay(globalStep: Int) {
        guard
            globalStep >= config.warmupSteps,
            globalStep % config.trainEvery == 0,
            replayBuffer.canSample(batchSize: config.batchSize)
        else {
            return
        }
        let transitions = replayBuffer.sample(batchSize: config.batchSize)
        let batch = DQNBatch(transitions: transitions)
        _ = learner.trainStep(batch: batch)
    }

    private func maybeSyncTargetNetwork(globalStep: Int) {
        guard
            config.targetSyncEvery > 0,
            globalStep > 0, globalStep % config.targetSyncEvery == 0
        else {
            return
        }
        learner.syncTargetFromOnline()
    }
}
