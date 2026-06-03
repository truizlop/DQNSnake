import Foundation

struct DQNActionInspection {
    let action: SnakeAction
    let qValues: [Float]
    let activations: DQNActivationSnapshot
}
