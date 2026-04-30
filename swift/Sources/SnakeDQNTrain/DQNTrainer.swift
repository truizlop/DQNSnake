import Foundation
import MLX
import SnakeEnv

struct DQNTrainingConfig {
    // Overall control
    var totalEnvironmentSteps: Int = 200_000
    var maxStepsPerEpisode: Int = 10_000

    // DQN scheduling knobs (placeholders for your implementation)
    var warmupSteps: Int = 5_000
    var trainEvery: Int = 4
    var targetSyncEvery: Int = 10_000
    var batchSize: Int = 32
    var gamma: Float = 0.99

    // Epsilon-greedy schedule placeholders
    var epsilonStart: Float = 1.0
    var epsilonEnd: Float = 0.1
    var epsilonDecaySteps: Int = 1_000_000
}

struct DQNTransition {
    let state: MLXArray
    let action: SnakeAction
    let reward: Float
    let nextState: MLXArray
    let done: Bool
}

struct DQNTrainer {
    let env: SnakeEnv
    let onlineQNetwork: DQNModel
    let targetQNetwork: DQNModel
    var config: DQNTrainingConfig

    init(env: SnakeEnv, config: DQNTrainingConfig = DQNTrainingConfig()) {
        self.env = env
        self.config = config
        self.onlineQNetwork = DQNModel()
        self.targetQNetwork = DQNModel()
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

                // TODO: Store into replay buffer.
                onTransitionPlaceholder(transition)

                // TODO: Sample replay + optimize online Q-network here.
                maybeTrainPlaceholder(globalStep: globalStep)

                // TODO: Periodically copy online params -> target network here.
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
        _ = state
        _ = globalStep
        // TODO: Implement epsilon-greedy action selection from onlineQNetwork(state).
        return SnakeAction.allCases.randomElement()!
    }

    private func onTransitionPlaceholder(_ transition: DQNTransition) {
        _ = transition
        // TODO: replayBuffer.append(transition)
    }

    private func maybeTrainPlaceholder(globalStep: Int) {
        _ = globalStep
        // TODO: if replayBuffer has enough samples, sample minibatch and run one gradient step.
    }

    private func maybeSyncTargetNetworkPlaceholder(globalStep: Int) {
        _ = globalStep
        // TODO: if globalStep % targetSyncEvery == 0, copy online weights to target network.
    }
}
