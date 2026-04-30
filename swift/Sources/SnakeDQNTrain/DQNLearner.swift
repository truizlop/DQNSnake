import Foundation
import MLX
import MLXNN
import MLXOptimizers

final class DQNLearner {
    private let onlineQNetwork: DQNModel
    private let targetQNetwork: DQNModel
    private let optimizer: Adam
    // Discount factor for future rewards in Bellman targets:
    // targetQ = reward + gamma * (1 - done) * max_a' Q_target(nextState, a').
    // gamma near 0 emphasizes immediate rewards; gamma near 1 values long-term return.
    private let gamma: Float
    private let lossAndGrad: (DQNModel, [MLXArray]) -> ([MLXArray], ModuleParameters)

    init(gamma: Float, learningRate: Float) {
        self.onlineQNetwork = DQNModel()
        self.targetQNetwork = DQNModel()
        self.gamma = gamma
        self.optimizer = Adam(learningRate: learningRate)
        self.lossAndGrad = valueAndGrad(model: onlineQNetwork) { [targetQNetwork] model, arrays in
            let states = arrays[0]
            let nextStates = arrays[1]
            let actions = arrays[2]
            let rewards = arrays[3]
            let notDoneMask = arrays[4]

            // DQN is trained against Q(s, a) for the action actually taken in each transition.
            // The network outputs one Q-value per action for each state: [B, actionCount].
            // We need to "gather" one value per row using the sampled action index for that row.
            //
            // MLX does not provide a direct batched gather API in this codepath, so we build an
            // explicit one-hot action mask and use it to select the chosen action value:
            // 1) actionRange: [1, actionCount] = [0, 1, ..., actionCount-1]
            // 2) compare against actions reshaped to [B, 1] to produce actionMask [B, actionCount]
            // 3) multiply qValues * actionMask to keep only the selected action per row
            // 4) sum across the action axis -> predictedQ [B], where each entry is Q(s_i, a_i)
            //
            // This yields the exact scalar target we regress in DQN for each transition, while
            // keeping the operation fully vectorized across the minibatch.
            let qValues = model(states) // [B, actionCount]
            let actionSpace = qValues.shape[1]
            let actionRange = MLXArray(0 ..< actionSpace).reshaped(1, actionSpace)
            let actionMask = (actionRange .== actions.reshaped(-1, 1)).asType(qValues.dtype)
            let predictedQ = (qValues * actionMask).sum(axis: 1)

            let nextQValues = targetQNetwork(nextStates)
            let maxNextQ = nextQValues.max(axis: 1)
            let targetQ = rewards + gamma * notDoneMask * maxNextQ

            return [mseLoss(predictions: predictedQ, targets: targetQ, reduction: .mean)]
        }
    }

    func greedyAction(for state: MLXArray) -> SnakeAction {
        let qValues = onlineQNetwork(state)
        let greedyActionIndex = qValues.argMax().item(Int.self)
        return SnakeAction(rawValue: greedyActionIndex) ?? .up
    }

    func syncTargetFromOnline() {
        targetQNetwork.update(parameters: onlineQNetwork.parameters())
    }

    @discardableResult
    func trainStep(batch: DQNBatch) -> Float {
        let (values, gradients) = lossAndGrad(
            onlineQNetwork,
            [batch.states, batch.nextStates, batch.actions, batch.rewards, batch.notDoneMask]
        )
        optimizer.update(model: onlineQNetwork, gradients: gradients)
        return values[0].item(Float.self)
    }
}
