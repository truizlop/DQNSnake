import Testing
@testable import SnakeEnv

struct MockBridge: SnakeBridgeClient {
    let frame: [[UInt8]]

    func reset(seed _: Int?) throws -> [[UInt8]] {
        frame
    }

    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float) {
        (frame, 1, false, 1)
    }

    func score() -> Float {
        1
    }

    func isDone() -> Bool {
        false
    }

    func render() throws {}

    func saveLastEpisodeGIF(path _: String, scale _: Int, frameDurationMs _: Int) throws -> Int {
        0
    }
}

@Test func frameStackFlattensFourFrames() {
    var stack = FrameStack(capacity: 4)
    stack.reset(frame: [1, 2])
    #expect(stack.stacked() == [1, 2, 1, 2, 1, 2, 1, 2])

    stack.append([3, 4])
    #expect(stack.stacked() == [1, 2, 1, 2, 1, 2, 3, 4])
}

@Test func snakeEnvReturnsFlattenedObservation() async throws {
    let env = SnakeEnv(bridge: MockBridge(frame: [[0, 1], [2, 3]]))

    let resetFrame = try await env.reset()
    #expect(resetFrame.width == 2)
    #expect(resetFrame.height == 2)
    #expect(resetFrame.data == [0, 1, 2, 3])

    let step = try await env.step(action: 0)
    #expect(step.observation == [0, 1, 2, 3])
    #expect(step.reward == 1)
    #expect(step.done == false)
    #expect(step.score == 1)
}
