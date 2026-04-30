import Foundation
import MLX

struct DQNTransition {
    let state: MLXArray
    let action: SnakeAction
    let reward: Float
    let nextState: MLXArray
    let done: Bool
}
