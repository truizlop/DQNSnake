import Foundation
import MLX
import SnakeEnv

struct DQNEvaluator {
    let env: SnakeEnv
    let maxStepsPerEpisode: Int
    private let appleRewardThreshold: Float = 0.5

    func evaluate(
        episodes: Int,
        selectAction: (MLXArray) -> SnakeAction
    ) async throws -> DQNEvaluationResult {
        precondition(episodes > 0, "Evaluation requires at least one episode.")

        var totalReward: Float = 0
        var totalScore: Float = 0
        var totalApples = 0

        for _ in 0..<episodes {
            let episodeResult = try await runEvaluationEpisode(selectAction: selectAction)
            totalReward += episodeResult.totalReward
            totalScore += episodeResult.finalScore
            totalApples += episodeResult.applesEaten
        }

        return DQNEvaluationResult(
            episodes: episodes,
            averageReward: totalReward / Float(episodes),
            averageScore: totalScore / Float(episodes),
            averageApples: Float(totalApples) / Float(episodes)
        )
    }

    private func runEvaluationEpisode(
        selectAction: (MLXArray) -> SnakeAction
    ) async throws -> DQNEpisodeResult {
        var stepsInEpisode = 0
        var totalReward: Float = 0
        var finalScore: Float = 0
        var applesEaten = 0

        let initial = try await env.reset()
        var frameStack = FrameStack(capacity: 4)
        frameStack.reset(frame: initial.data)
        var state = stateTensor(from: frameStack.stacked(), width: initial.width, height: initial.height)

        while stepsInEpisode < maxStepsPerEpisode {
            let action = selectAction(state)
            let stepResult = try await env.step(action: action.rawValue)

            frameStack.append(stepResult.observation)
            let nextState = stateTensor(
                from: frameStack.stacked(),
                width: initial.width,
                height: initial.height
            )

            totalReward += stepResult.reward
            finalScore = stepResult.score
            if stepResult.reward > appleRewardThreshold {
                applesEaten += 1
            }
            state = nextState
            stepsInEpisode += 1

            if stepResult.done {
                break
            }
        }

        return DQNEpisodeResult(
            steps: stepsInEpisode,
            totalReward: totalReward,
            finalScore: finalScore,
            applesEaten: applesEaten,
            actionCounts: [:]
        )
    }

    private func stateTensor(from stackedFrames: [UInt8], width: Int, height: Int) -> MLXArray {
        let values = stackedFrames.map { Float($0) }
        let array = MLXArray(values)
        return array.reshaped(1, height, width, 4)
    }
}
