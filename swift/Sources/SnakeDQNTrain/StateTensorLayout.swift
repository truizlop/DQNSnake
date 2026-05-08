import Foundation
import MLX

enum StateTensorLayout {
    static func interleaveStackedFrames(
        stackedFrames: [UInt8],
        width: Int,
        height: Int,
        frameCount: Int = 4
    ) -> [Float] {
        let pixelsPerFrame = width * height
        precondition(
            stackedFrames.count == pixelsPerFrame * frameCount,
            "Expected \(pixelsPerFrame * frameCount) stacked frame values, got \(stackedFrames.count)."
        )

        var interleaved: [Float] = []
        interleaved.reserveCapacity(stackedFrames.count)

        for pixelIndex in 0..<pixelsPerFrame {
            for frameIndex in 0..<frameCount {
                let sourceIndex = frameIndex * pixelsPerFrame + pixelIndex
                interleaved.append(Float(stackedFrames[sourceIndex]))
            }
        }
        return interleaved
    }

    static func stateTensor(from stackedFrames: [UInt8], width: Int, height: Int) -> MLXArray {
        let values = interleaveStackedFrames(stackedFrames: stackedFrames, width: width, height: height)
        return MLXArray(values).reshaped(1, height, width, 4)
    }
}
