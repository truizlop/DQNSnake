import Foundation

protocol SnakeBridgeClient: Sendable {
    func reset(seed: Int?) throws -> [[UInt8]]
    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float)
    func score() -> Float
    func isDone() -> Bool
    func render() throws
    func saveLastEpisodeGIF(path: String, scale: Int, frameDurationMs: Int) throws -> Int
}

public actor SnakeEnv {
    private let bridge: SnakeBridgeClient

    public init(usePythonBridge: Bool = true, pythonModulePath: String? = nil, seed: Int? = nil) {
        if usePythonBridge, let pythonBridge = PythonSnakeBridge.make(pythonModulePath: pythonModulePath, seed: seed)
        {
            self.bridge = pythonBridge
        } else {
            self.bridge = UnavailableSnakeBridge()
        }
    }

    init(bridge: any SnakeBridgeClient) {
        self.bridge = bridge
    }

    public func reset(seed: Int? = nil) async throws -> Frame {
        let grid = try bridge.reset(seed: seed)
        return try flatten(grid: grid)
    }

    public func step(action: Int) async throws -> StepResult {
        guard (0...3).contains(action) else {
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

    public func saveLastEpisodeGIF(path: String, scale: Int = 8, frameDurationMs: Int = 80) async throws -> Int {
        try bridge.saveLastEpisodeGIF(path: path, scale: scale, frameDurationMs: frameDurationMs)
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
