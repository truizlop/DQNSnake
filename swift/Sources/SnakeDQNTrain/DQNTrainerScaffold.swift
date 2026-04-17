import Foundation
import MLX
import SnakeEnv

struct DQNTrainerScaffold {
    let env: SnakeEnv
    let enableMLX: Bool

    func smokeRollout(steps: Int) async throws {
        let initial = try await env.reset()
        var stack = FrameStack(capacity: 4)
        stack.reset(frame: initial.data)

        if enableMLX {
            let state0 = Stream.withNewDefaultStream(device: .cpu) {
                toStateTensor(stacked: stack.stacked(), width: initial.width, height: initial.height)
            }
            print("initial state shape: \(state0.shape)")
        } else {
            print("MLX tensor ops disabled (`SNAKE_ENABLE_MLX=0`). Running environment integration only.")
        }

        for i in 0..<steps {
            // Placeholder policy: random actions. Replace with epsilon-greedy Q-policy.
            let action = Int.random(in: 0...3)
            let result = try await env.step(action: action)

            stack.append(result.observation)
            if enableMLX {
                let state = Stream.withNewDefaultStream(device: .cpu) {
                    toStateTensor(stacked: stack.stacked(), width: initial.width, height: initial.height)
                }

                // This is where replay buffer insertion should happen next:
                // (state, action, reward, nextState, done)
                _ = state
            }

            print("step=\(i) action=\(action) reward=\(result.reward) score=\(result.score) done=\(result.done)")

            if result.done {
                let resetFrame = try await env.reset()
                stack.reset(frame: resetFrame.data)
            }
        }
    }

    private func toStateTensor(stacked: [UInt8], width: Int, height: Int) -> MLXArray {
        // Contract for DQN input: [batch=1, channels=4, height, width], Float32 in [0,1]
        let floatValues = stacked.map { Float($0) / 255.0 }
        let array = MLXArray(floatValues)
        return array.reshaped(1, 4, height, width)
    }
}
