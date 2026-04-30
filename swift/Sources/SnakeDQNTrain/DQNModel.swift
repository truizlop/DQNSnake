import Foundation
import MLX
import MLXNN

class DQNModel: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let dense: Linear
    let output: Linear

    override init() {
        // input should be 4 frames of 84x84
        conv1 = Conv2d(
            inputChannels: 4,
            outputChannels: 16,
            kernelSize: IntOrPair(8),
            stride: 4
        ) // 16 * 20 * 20
        conv2 = Conv2d(
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: IntOrPair(4),
            stride: 2
        ) // 32 * 9 * 9
        dense = Linear(
            inputDimensions: 2592, // 32 * 9 * 9 after flattening
            outputDimensions: 256
        )
        output = Linear(
            inputDimensions: 256,
            outputDimensions: 4
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = relu(conv1(input))
        x = relu(conv2(x))
        x = x.flattened(start: 1)
        x = relu(dense(x))
        return output(x)
    }
}
