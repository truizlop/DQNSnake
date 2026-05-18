import Foundation

struct UnavailableSnakeBridge: SnakeBridgeClient {
    func reset(seed _: Int?) throws -> [[UInt8]] {
        throw SnakeEnvError.bridgeUnavailable
    }

    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float) {
        throw SnakeEnvError.bridgeUnavailable
    }

    func score() -> Float {
        0
    }

    func isDone() -> Bool {
        false
    }

    func render() throws {
        throw SnakeEnvError.bridgeUnavailable
    }

    func saveLastEpisodeGIF(path _: String, scale _: Int, frameDurationMs _: Int) throws -> Int {
        throw SnakeEnvError.bridgeUnavailable
    }
}
