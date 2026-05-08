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
    private let dqnAlgorithm: DQNAlgorithm
    private let gradientClipNorm: Float
    private let lossAndGrad: (DQNModel, [MLXArray]) -> ([MLXArray], ModuleParameters)

    init(gamma: Float, learningRate: Float, dqnAlgorithm: DQNAlgorithm, gradientClipNorm: Float) {
        self.onlineQNetwork = DQNModel()
        self.targetQNetwork = DQNModel()
        self.gamma = gamma
        self.dqnAlgorithm = dqnAlgorithm
        self.gradientClipNorm = gradientClipNorm
        self.optimizer = Adam(learningRate: learningRate)
        // Start with a consistent target network; otherwise early TD targets are random/noisy
        // until the first periodic sync.
        self.targetQNetwork.update(parameters: self.onlineQNetwork.parameters())
        self.lossAndGrad = valueAndGrad(model: onlineQNetwork) { [targetQNetwork] model, arrays in
            let states = arrays[0]
            let nextStates = arrays[1]
            let actions = arrays[2]
            let rewards = arrays[3]
            let notDoneMask = arrays[4]

            let qValues = model(states) // [B, actionCount]
            let predictedQ = Self.gatherActionValues(qValues: qValues, actions: actions)

            let maxNextQ = Self.bootstrapNextQ(
                onlineNextQValues: model(nextStates),
                targetNextQValues: targetQNetwork(nextStates),
                algorithm: dqnAlgorithm
            )
            let targetQ = rewards + gamma * notDoneMask * maxNextQ
            let tdError = predictedQ - targetQ
            let weightedSquaredError = arrays[5] * tdError.square()
            let loss = weightedSquaredError.mean()
            return [loss]
        }
    }

    func greedyAction(for state: MLXArray) -> SnakeAction {
        let qValues = onlineQNetwork(state)
        let greedyActionIndex = qValues.argMax().item(Int.self)
        guard let action = SnakeAction(rawValue: greedyActionIndex) else {
            preconditionFailure("Invalid greedy action index produced by model: \(greedyActionIndex)")
        }
        return action
    }

    func syncTargetFromOnline() {
        targetQNetwork.update(parameters: onlineQNetwork.parameters())
    }

    func saveOnlineModel(to url: URL, metadata: [String: String] = [:]) throws {
        let arrays = Dictionary(uniqueKeysWithValues: onlineQNetwork.parameters().flattened())
        try save(arrays: arrays, metadata: metadata, url: url)
    }

    @discardableResult
    func loadOnlineModel(from url: URL) throws -> [String: String] {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        let parameters = ModuleParameters.unflattened(arrays)
        onlineQNetwork.update(parameters: parameters)
        syncTargetFromOnline()
        return metadata
    }

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
    static func gatherActionValues(qValues: MLXArray, actions: MLXArray) -> MLXArray {
        let actionSpace = qValues.shape[1]
        let actionRange = MLXArray(0 ..< actionSpace).reshaped(1, actionSpace)
        let actionMask = (actionRange .== actions.reshaped(-1, 1)).asType(qValues.dtype)
        return (qValues * actionMask).sum(axis: 1)
    }

    static func bootstrapNextQ(
        onlineNextQValues: MLXArray,
        targetNextQValues: MLXArray,
        algorithm: DQNAlgorithm
    ) -> MLXArray {
        switch algorithm {
        case .single:
            return targetNextQValues.max(axis: 1)
        case .double:
            let nextGreedyActions = onlineNextQValues.argMax(axis: 1)
            return gatherActionValues(qValues: targetNextQValues, actions: nextGreedyActions)
        }
    }

    func trainStep(batch: DQNBatch) -> DQNTrainStepMetrics {
        let preUpdateQValues = onlineQNetwork(batch.states)
        let predictedQ = Self.gatherActionValues(qValues: preUpdateQValues, actions: batch.actions)
        let meanAbsQ = predictedQ.abs().mean().item(Float.self)
        let maxAbsQ = predictedQ.abs().max().item(Float.self)

        let (values, gradients) = lossAndGrad(
            onlineQNetwork,
            [batch.states, batch.nextStates, batch.actions, batch.rewards, batch.notDoneMask, batch.importanceWeights]
        )
        let loss = values[0].item(Float.self)
        let gradientsFinite = Self.allFinite(parameters: gradients)
        let lossFinite = loss.isFinite
        let gradientL2Norm = Self.l2Norm(parameters: gradients)

        let shouldSkipUpdate = !lossFinite || !gradientsFinite || !gradientL2Norm.isFinite
        let clippedGradients = Self.clipGradients(
            gradients,
            gradientL2Norm: gradientL2Norm,
            maxNorm: gradientClipNorm
        )
        if !shouldSkipUpdate {
            optimizer.update(model: onlineQNetwork, gradients: clippedGradients)
        }

        return DQNTrainStepMetrics(
            loss: loss,
            meanAbsQ: meanAbsQ,
            maxAbsQ: maxAbsQ,
            gradientL2Norm: gradientL2Norm,
            skippedUpdate: shouldSkipUpdate
        )
    }

    func tdErrors(batch: DQNBatch) -> [Float] {
        let qValues = onlineQNetwork(batch.states)
        let predictedQ = Self.gatherActionValues(qValues: qValues, actions: batch.actions)

        let nextValue = Self.bootstrapNextQ(
            onlineNextQValues: onlineQNetwork(batch.nextStates),
            targetNextQValues: targetQNetwork(batch.nextStates),
            algorithm: dqnAlgorithm
        )

        let targetQ = batch.rewards + gamma * batch.notDoneMask * nextValue
        let absTdError = (predictedQ - targetQ).abs().asType(.float32)
        return absTdError.asArray(Float.self)
    }

    private static func allFinite(parameters: ModuleParameters) -> Bool {
        for (_, value) in parameters.flattened() {
            if !isFinite(value).all().item(Bool.self) {
                return false
            }
        }
        return true
    }

    private static func l2Norm(parameters: ModuleParameters) -> Float {
        var squaredNorm: Float = 0
        for (_, value) in parameters.flattened() {
            squaredNorm += value.square().sum().item(Float.self)
        }
        return Foundation.sqrt(squaredNorm)
    }

    static func gradientClipScale(gradientL2Norm: Float, maxNorm: Float) -> Float {
        guard maxNorm > 0, gradientL2Norm.isFinite, gradientL2Norm > maxNorm else {
            return 1
        }
        return maxNorm / gradientL2Norm
    }

    static func clipGradients(
        _ gradients: ModuleParameters,
        gradientL2Norm: Float,
        maxNorm: Float
    ) -> ModuleParameters {
        let scale = gradientClipScale(gradientL2Norm: gradientL2Norm, maxNorm: maxNorm)
        guard scale < 1 else {
            return gradients
        }
        let scaled = Dictionary(uniqueKeysWithValues: gradients.flattened().map { key, value in
            (key, value * scale)
        })
        return ModuleParameters.unflattened(scaled)
    }
}

extension DQNLearner: DQNModelCheckpointSaving {}
