import Foundation

final class DQNStructuredLogger {
    private let outputURL: URL
    private let encoder: JSONEncoder
    private var handle: FileHandle?

    init(outputPath: String) {
        self.outputURL = URL(fileURLWithPath: (outputPath as NSString).standardizingPath)
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.sortedKeys]
    }

    func start() throws {
        let directory = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: outputURL.path) {
            FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        }
        handle = try FileHandle(forWritingTo: outputURL)
        try handle?.seekToEnd()
    }

    func log(event: String, step: Int, fields: [String: String]) {
        guard let handle else {
            return
        }
        var payload = fields
        payload["event"] = event
        payload["step"] = "\(step)"
        payload["ts"] = Self.iso8601DateString()
        do {
            let data = try encoder.encode(payload)
            var line = data
            line.append(0x0A)
            try handle.write(contentsOf: line)
        } catch {
            fputs("Structured log write error: \(error)\n", stderr)
        }
    }

    func stop() {
        do {
            try handle?.close()
        } catch {
            // no-op
        }
        handle = nil
    }

    private static func iso8601DateString() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }
}
