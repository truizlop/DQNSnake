import Foundation

public struct StepResult: Sendable {
    public let observation: [UInt8]
    public let reward: Float
    public let done: Bool
    public let score: Float

    public init(observation: [UInt8], reward: Float, done: Bool, score: Float) {
        self.observation = observation
        self.reward = reward
        self.done = done
        self.score = score
    }
}

public struct Frame: Sendable {
    public let width: Int
    public let height: Int
    public let data: [UInt8]

    public init(width: Int, height: Int, data: [UInt8]) {
        self.width = width
        self.height = height
        self.data = data
    }
}
