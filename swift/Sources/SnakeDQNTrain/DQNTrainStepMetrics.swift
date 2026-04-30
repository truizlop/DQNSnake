import Foundation

struct DQNTrainStepMetrics {
    let loss: Float
    let meanAbsQ: Float
    let maxAbsQ: Float
    let gradientL2Norm: Float
    let skippedUpdate: Bool
}
