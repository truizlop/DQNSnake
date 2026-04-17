import Foundation

public enum SnakeEnvError: Error {
    case bridgeUnavailable
    case invalidFrameShape
    case invalidAction
    case pythonBridgeInitializationFailed(String)
    case pythonConversionFailed(String)
}

protocol SnakeBridgeClient: Sendable {
    func reset() throws -> [[UInt8]]
    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float)
    func score() -> Float
    func isDone() -> Bool
    func render() throws
}

struct UnavailableSnakeBridge: SnakeBridgeClient {
    func reset() throws -> [[UInt8]] {
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
}

public actor SnakeEnv {
    private let bridge: SnakeBridgeClient

    public init(usePythonBridge: Bool = true, pythonModulePath: String? = nil) {
        if usePythonBridge, let pythonBridge = PythonSnakeBridge.make(pythonModulePath: pythonModulePath) {
            self.bridge = pythonBridge
        } else {
            self.bridge = UnavailableSnakeBridge()
        }
    }

    init(bridge: any SnakeBridgeClient) {
        self.bridge = bridge
    }

    public func reset() async throws -> Frame {
        let grid = try bridge.reset()
        return try flatten(grid: grid)
    }

    public func step(action: Int) async throws -> StepResult {
        guard action >= 0 else {
            throw SnakeEnvError.invalidAction
        }

        let out = try bridge.step(action: action)
        let frame = try flatten(grid: out.observation)

        return StepResult(
            observation: frame.data,
            reward: out.reward,
            done: out.done,
            score: out.score
        )
    }

    public func score() async -> Float {
        bridge.score()
    }

    public func isDone() async -> Bool {
        bridge.isDone()
    }

    public func render() async throws {
        try bridge.render()
    }

    private func flatten(grid: [[UInt8]]) throws -> Frame {
        guard let firstRow = grid.first else {
            throw SnakeEnvError.invalidFrameShape
        }

        let height = grid.count
        let width = firstRow.count
        guard width > 0 else {
            throw SnakeEnvError.invalidFrameShape
        }

        for row in grid where row.count != width {
            throw SnakeEnvError.invalidFrameShape
        }

        return Frame(width: width, height: height, data: grid.flatMap { $0 })
    }
}
