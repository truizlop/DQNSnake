import Foundation

public struct FrameStack: Sendable {
    private var frames: [[UInt8]] = []
    private let capacity: Int

    public init(capacity: Int = 4) {
        self.capacity = max(1, capacity)
    }

    public mutating func reset(frame: [UInt8]) {
        frames = Array(repeating: frame, count: capacity)
    }

    public mutating func append(_ frame: [UInt8]) {
        frames.append(frame)
        if frames.count > capacity {
            frames.removeFirst(frames.count - capacity)
        }
    }

    public func stacked() -> [UInt8] {
        frames.flatMap { $0 }
    }

    public var count: Int {
        frames.count
    }
}
