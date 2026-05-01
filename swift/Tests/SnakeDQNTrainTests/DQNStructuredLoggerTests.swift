import Foundation
import Testing
@testable import SnakeDQNTrain

private func tempLogFilePath() -> String {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathComponent("obs.jsonl")
        .path
}

@Test func structuredLoggerWritesJsonLineWithRequiredFields() throws {
    let path = tempLogFilePath()
    let logger = DQNStructuredLogger(outputPath: path)
    try logger.start()
    logger.log(event: "unit_test", step: 7, fields: ["k": "v"])
    logger.stop()

    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let text = String(decoding: data, as: UTF8.self)
    let line = text.split(separator: "\n").last
    #expect(line != nil)

    let parsed = try JSONSerialization.jsonObject(with: Data((line ?? "").utf8)) as? [String: String]
    #expect(parsed?["event"] == "unit_test")
    #expect(parsed?["step"] == "7")
    #expect(parsed?["k"] == "v")
    #expect(parsed?["ts"]?.isEmpty == false)
}
