import MLX
import Testing
@testable import SnakeDQNTrain

@Test func gatherActionValuesMatchesManualSelection() {
    let qValues = MLXArray([
        Float(1), Float(2), Float(3), Float(4),
        Float(10), Float(20), Float(30), Float(40),
        Float(5), Float(6), Float(7), Float(8),
    ]).reshaped(3, 4)
    let actions = MLXArray([Int32(3), Int32(0), Int32(2)])
    let gathered = DQNLearner.gatherActionValues(qValues: qValues, actions: actions).asArray(Float.self)
    #expect(gathered == [4.0, 10.0, 7.0])
}

@Test func bootstrapNextQDiffersBetweenSingleAndDouble() {
    let online = MLXArray([
        Float(1), Float(9), Float(2), Float(3),
        Float(8), Float(1), Float(2), Float(3),
    ]).reshaped(2, 4)
    let target = MLXArray([
        Float(10), Float(1), Float(2), Float(3),
        Float(1), Float(20), Float(2), Float(3),
    ]).reshaped(2, 4)

    let single = DQNLearner.bootstrapNextQ(
        onlineNextQValues: online,
        targetNextQValues: target,
        algorithm: .single
    ).asArray(Float.self)
    let double = DQNLearner.bootstrapNextQ(
        onlineNextQValues: online,
        targetNextQValues: target,
        algorithm: .double
    ).asArray(Float.self)

    #expect(single == [10.0, 20.0])
    #expect(double == [1.0, 1.0])
}
