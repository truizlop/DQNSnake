import Foundation
import MLX

struct DQNBatch {
    let states: MLXArray
    let nextStates: MLXArray
    let actions: MLXArray
    let rewards: MLXArray
    let notDoneMask: MLXArray

    init(transitions: [DQNTransition]) {
        precondition(!transitions.isEmpty, "Cannot build DQNBatch from empty transitions.")
        states = concatenated(transitions.map(\.state), axis: 0)
        nextStates = concatenated(transitions.map(\.nextState), axis: 0)
        actions = MLXArray(transitions.map { Int32($0.action.rawValue) })
        rewards = MLXArray(transitions.map(\.reward))
        notDoneMask = MLXArray(transitions.map { $0.done ? Float(0) : Float(1) })
    }
}
