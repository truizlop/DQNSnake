import Foundation
import MLX
import SnakeEnv

struct DQNEpisodeRunner {
    let env: SnakeEnv
    let maxStepsPerEpisode: Int

    func runEpisode(
        globalStep: inout Int,
        totalEnvironmentSteps: Int,
        selectAction: (MLXArray, Int) -> SnakeAction,
        onTransition: (DQNTransition, Int) -> Void
    ) async throws -> Int {
        var stepsInEpisode = 0

        let initial = try await env.reset()
        var frameStack = FrameStack(capacity: 4)
        frameStack.reset(frame: initial.data)
        var state = stateTensor(from: frameStack.stacked(), width: initial.width, height: initial.height)

        while globalStep < totalEnvironmentSteps && stepsInEpisode < maxStepsPerEpisode {
            let action = selectAction(state, globalStep)
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
            onTransition(transition, globalStep)

            state = nextState
            globalStep += 1
            stepsInEpisode += 1

            if stepResult.done {
                break
            }
        }

        return stepsInEpisode
    }

    private func stateTensor(from stackedFrames: [UInt8], width: Int, height: Int) -> MLXArray {
        let values = stackedFrames.map { Float($0) }
        let array = MLXArray(values)
        return array.reshaped(1, 4, height, width)
    }
}
