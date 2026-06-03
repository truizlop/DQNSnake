import Foundation
import MLX
import Testing
@testable import SnakeDQNTrain

@Test func activationImageWriterCreatesPGMForTiledChannels() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("DQNSnakeActivationImageWriterTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer {
        try? FileManager.default.removeItem(at: directory)
    }

    let url = directory.appendingPathComponent("channels.pgm")
    let array = MLXArray([
        Float(0), Float(1),
        Float(2), Float(3),
        Float(4), Float(5),
        Float(6), Float(7),
    ]).reshaped(1, 2, 2, 2)

    try DQNActivationImageWriter.writeTiledChannels(array, to: url)
    let data = try Data(contentsOf: url)
    let header = String(decoding: data.prefix(11), as: UTF8.self)
    #expect(header == "P5\n6 2\n255\n")
}

@Test func activationImageWriterCreatesPGMForVectorGrid() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("DQNSnakeActivationVectorWriterTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer {
        try? FileManager.default.removeItem(at: directory)
    }

    let url = directory.appendingPathComponent("vector.pgm")
    try DQNActivationImageWriter.writeVectorAsGrid([0, 1, 2, 3], columns: 2, to: url)
    let data = try Data(contentsOf: url)
    let header = String(decoding: data.prefix(11), as: UTF8.self)
    #expect(header == "P5\n2 2\n255\n")
}
