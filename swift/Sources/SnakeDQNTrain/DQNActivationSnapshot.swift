import Foundation
import MLX

struct DQNActivationSnapshot {
    let input: MLXArray
    let conv1: MLXArray
    let conv2: MLXArray
    let dense: MLXArray
    let qValues: MLXArray
}
