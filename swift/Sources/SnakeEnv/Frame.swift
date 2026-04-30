import Foundation

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
