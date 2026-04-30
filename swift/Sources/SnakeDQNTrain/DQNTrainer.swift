import Foundation
import MLX
import MLXNN
import SnakeEnv

struct DQNTrainer {
    let env: SnakeEnv
    let onlineQNetwork: DQNModel
    let targetQNetwork: DQNModel
    var config: DQNTrainingConfig
    var replayBuffer: ReplayBuffer
    let learner: DQNLearner

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.onlineQNetwork = DQNModel()
        self.targetQNetwork = DQNModel()
        self.replayBuffer = ReplayBuffer(capacity: config.replayBufferCapacity)
        self.learner = DQNLearner(
            onlineQNetwork: self.onlineQNetwork,
            targetQNetwork: self.targetQNetwork,
            gamma: config.gamma,
            learningRate: config.learningRate
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
                let action = selectActionPlaceholder(state: state, globalStep: globalStep)
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

                maybeSyncTargetNetworkPlaceholder(globalStep: globalStep)

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

    private func selectActionPlaceholder(state: MLXArray, globalStep: Int) -> SnakeAction {
        let epsilon = epsilonValue(globalStep: globalStep)
        if Float.random(in: 0..<1) < epsilon {
            return SnakeAction.allCases.randomElement()!
        }

        let qValues = onlineQNetwork(state)
        let greedyActionIndex = qValues.argMax().item(Int.self)
        return SnakeAction(rawValue: greedyActionIndex) ?? .up
    }

    private func epsilonValue(globalStep: Int) -> Float {
        guard config.epsilonDecaySteps > 0 else {
            return config.epsilonEnd
        }

        let clampedStep = min(max(globalStep, 0), config.epsilonDecaySteps)
        let progress = Float(clampedStep) / Float(config.epsilonDecaySteps)
        let epsilon = config.epsilonStart + (config.epsilonEnd - config.epsilonStart) * progress
        return min(max(epsilon, min(config.epsilonStart, config.epsilonEnd)), max(config.epsilonStart, config.epsilonEnd))
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

    private func maybeSyncTargetNetworkPlaceholder(globalStep: Int) {
        guard
            config.targetSyncEvery > 0,
            globalStep > 0, globalStep % config.targetSyncEvery == 0
        else {
            return
        }
        targetQNetwork.update(parameters: onlineQNetwork.parameters())
    }
}
