import Foundation
import MLX
import SnakeEnv

struct DQNPlayer {
    let env: SnakeEnv
    let maxStepsPerEpisode: Int
    let render: Bool
    let gifDirectory: String?
    let gifScale: Int
    let gifFrameDurationMs: Int
    let activationExporter: DQNActivationExporter?
    let activationDashboard: DQNActivationDashboardPublisher?
    let stepMode: Bool
    let stepIntervalSeconds: Double
    private let appleRewardThreshold: Float = 0.5

    func play(
        episodes: Int,
        fixedSeeds: [Int]?,
        selectAction: (MLXArray) -> SnakeAction,
        inspectAction: ((MLXArray) -> DQNActionInspection)? = nil
    ) async throws -> [DQNEpisodeResult] {
        precondition(episodes > 0, "Play mode requires at least one episode.")

        var results: [DQNEpisodeResult] = []
        var shouldPause = stepMode
        for index in 0..<episodes {
            let seed = fixedSeeds.map { seeds in seeds[index % seeds.count] }
            let result = try await playEpisode(
                index: index + 1,
                seed: seed,
                shouldPause: &shouldPause,
                selectAction: selectAction,
                inspectAction: inspectAction
            )
            results.append(result)
            try await maybeSaveGIF(episode: index + 1, result: result)
        }
        return results
    }

    private func playEpisode(
        index: Int,
        seed: Int?,
        shouldPause: inout Bool,
        selectAction: (MLXArray) -> SnakeAction,
        inspectAction: ((MLXArray) -> DQNActionInspection)?
    ) async throws -> DQNEpisodeResult {
        var stepsInEpisode = 0
        var totalReward: Float = 0
        var finalScore: Float = 0
        var applesEaten = 0
        var actionCounts: [SnakeAction: Int] = [:]
        var stopEpisode = false

        let initial = try await env.reset(seed: seed)
        if render {
            try await env.render()
        }
        var frameStack = FrameStack(capacity: 4)
        frameStack.reset(frame: initial.data)
        var state = stateTensor(from: frameStack.stacked(), width: initial.width, height: initial.height)

        while stepsInEpisode < maxStepsPerEpisode {
            let nextStep = stepsInEpisode + 1
            let inspection = inspectAction?(state)
            let action = inspection?.action ?? selectAction(state)
            if let inspection {
                try activationExporter?.export(
                    inspection: inspection,
                    episode: index,
                    step: nextStep
                )
                printActivationSummary(
                    inspection: inspection,
                    episode: index,
                    step: nextStep
                )
                activationDashboard?.publish(
                    inspection: inspection,
                    episode: index,
                    step: nextStep
                )
                if shouldPause {
                    switch waitForNextStep() {
                    case .next:
                        break
                    case .runContinuously:
                        shouldPause = false
                    case .stopEpisode:
                        stopEpisode = true
                    }
                }
                if stopEpisode {
                    break
                }
            }

            actionCounts[action, default: 0] += 1
            let stepResult = try await env.step(action: action.rawValue)
            if render {
                try await env.render()
            }

            frameStack.append(stepResult.observation)
            state = stateTensor(from: frameStack.stacked(), width: initial.width, height: initial.height)
            totalReward += stepResult.reward
            finalScore = stepResult.score
            if stepResult.reward > appleRewardThreshold {
                applesEaten += 1
            }
            stepsInEpisode += 1

            if stepResult.done {
                break
            }
        }

        let result = DQNEpisodeResult(
            steps: stepsInEpisode,
            totalReward: totalReward,
            finalScore: finalScore,
            applesEaten: applesEaten,
            actionCounts: actionCounts
        )
        print(
            "play episode=\(index) steps=\(result.steps) reward=\(result.totalReward) score=\(result.finalScore) apples=\(result.applesEaten)"
        )
        return result
    }

    private func printActivationSummary(
        inspection: DQNActionInspection,
        episode: Int,
        step: Int
    ) {
        let qValues = SnakeAction.allCases.map { action in
            "\(action)=\(String(format: "%.4f", inspection.qValues[action.rawValue]))"
        }.joined(separator: " ")
        print("activation episode=\(episode) step=\(step) action=\(inspection.action) \(qValues)")
    }

    private func waitForNextStep() -> DQNPlayStepCommand {
        if stepIntervalSeconds > 0 {
            print("step mode: auto-advancing in \(String(format: "%.2f", stepIntervalSeconds))s. Set SNAKE_PLAY_STEP_INTERVAL_SECONDS=0 for manual stepping.")
            Thread.sleep(forTimeInterval: stepIntervalSeconds)
            return .next
        }

        print("step mode: press Enter for next step, 'r' then Enter to run continuously, or 'q' then Enter to stop this episode.")
        let command = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        if command == "q" {
            return .stopEpisode
        }
        if command == "r" {
            return .runContinuously
        }
        return .next
    }

    private func maybeSaveGIF(episode: Int, result: DQNEpisodeResult) async throws {
        guard let gifDirectory, !gifDirectory.isEmpty else {
            return
        }

        let scoreTag = String(format: "%.3f", result.finalScore)
            .replacingOccurrences(of: "-", with: "m")
            .replacingOccurrences(of: ".", with: "p")
        let directory = URL(fileURLWithPath: gifDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory
            .appendingPathComponent("play_ep_\(episode)_score_\(scoreTag).gif")
            .path

        let frameCount = try await env.saveLastEpisodeGIF(
            path: path,
            scale: gifScale,
            frameDurationMs: gifFrameDurationMs
        )
        print("play gif=\(path) frames=\(frameCount)")
    }

    private func stateTensor(from stackedFrames: [UInt8], width: Int, height: Int) -> MLXArray {
        StateTensorLayout.stateTensor(from: stackedFrames, width: width, height: height)
    }
}
