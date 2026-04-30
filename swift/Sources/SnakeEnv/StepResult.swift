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
