import MLX
import Testing
@testable import SnakeDQNTrain

@Test func modelForwardWithActivationsReturnsExpectedShapes() {
    let model = DQNModel()
    let input = MLXArray([Float](repeating: 0, count: 84 * 84 * 4)).reshaped(1, 84, 84, 4)
    let snapshot = model.forwardWithActivations(input)

    #expect(snapshot.input.shape == [1, 84, 84, 4])
    #expect(snapshot.conv1.shape == [1, 20, 20, 16])
    #expect(snapshot.conv2.shape == [1, 9, 9, 32])
    #expect(snapshot.dense.shape == [1, 256])
    #expect(snapshot.qValues.shape == [1, 4])
}
