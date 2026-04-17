import Foundation

final class PythonSnakeBridge: SnakeBridgeClient, @unchecked Sendable {
    private let process: Process
    private let stdinHandle: FileHandle
    private let stdoutHandle: FileHandle
    private let lock = NSLock()

    private init(process: Process, stdinHandle: FileHandle, stdoutHandle: FileHandle) {
        self.process = process
        self.stdinHandle = stdinHandle
        self.stdoutHandle = stdoutHandle
    }

    deinit {
        _ = try? sendRaw(["cmd": "quit"])
        if process.isRunning {
            process.terminate()
        }
    }

    static func make(pythonModulePath: String?) -> PythonSnakeBridge? {
        let executable = pythonExecutable()
        let modulePath = resolvePythonModulePath(explicitPath: pythonModulePath)
        let serverScript = (modulePath as NSString).appendingPathComponent("bridge_server.py")
        guard FileManager.default.fileExists(atPath: serverScript) else {
            return nil
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = [serverScript]

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONPATH"] = modulePath
        process.environment = environment

        let inputPipe = Pipe()
        let outputPipe = Pipe()
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.standardError

        do {
            try process.run()
        } catch {
            return nil
        }

        let bridge = PythonSnakeBridge(
            process: process,
            stdinHandle: inputPipe.fileHandleForWriting,
            stdoutHandle: outputPipe.fileHandleForReading
        )

        do {
            let response = try bridge.sendRaw(["cmd": "create_env"])
            guard response["ok"] as? Bool == true else {
                return nil
            }
            return bridge
        } catch {
            return nil
        }
    }

    func reset() throws -> [[UInt8]] {
        let response = try send(["cmd": "reset"])
        guard let frame = response["frame"] as? [[NSNumber]] else {
            throw SnakeEnvError.pythonConversionFailed("Missing frame in reset response")
        }
        return frame.map { row in row.map { UInt8(clamping: $0.intValue) } }
    }

    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float) {
        let response = try send(["cmd": "step", "action": action])

        guard let observation = response["observation"] as? [[NSNumber]] else {
            throw SnakeEnvError.pythonConversionFailed("Missing observation in step response")
        }
        guard let reward = response["reward"] as? NSNumber else {
            throw SnakeEnvError.pythonConversionFailed("Missing reward in step response")
        }
        guard let done = response["done"] as? Bool else {
            throw SnakeEnvError.pythonConversionFailed("Missing done in step response")
        }
        guard let score = response["score"] as? NSNumber else {
            throw SnakeEnvError.pythonConversionFailed("Missing score in step response")
        }

        return (
            observation.map { row in row.map { UInt8(clamping: $0.intValue) } },
            reward.floatValue,
            done,
            score.floatValue
        )
    }

    func score() -> Float {
        do {
            let response = try send(["cmd": "score"])
            return (response["score"] as? NSNumber)?.floatValue ?? 0
        } catch {
            return 0
        }
    }

    func isDone() -> Bool {
        do {
            let response = try send(["cmd": "is_done"])
            return (response["done"] as? Bool) ?? false
        } catch {
            return false
        }
    }

    func render() throws {
        _ = try send(["cmd": "render"])
    }

    private func send(_ payload: [String: Any]) throws -> [String: Any] {
        let response = try sendRaw(payload)
        if response["ok"] as? Bool == false {
            let error = (response["error"] as? String) ?? "Python bridge error"
            throw SnakeEnvError.pythonBridgeInitializationFailed(error)
        }
        return response
    }

    private func sendRaw(_ payload: [String: Any]) throws -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let json = try JSONSerialization.data(withJSONObject: payload)
        guard var line = String(data: json, encoding: .utf8) else {
            throw SnakeEnvError.pythonConversionFailed("Unable to encode request JSON")
        }
        line += "\n"

        guard let request = line.data(using: .utf8) else {
            throw SnakeEnvError.pythonConversionFailed("Unable to build request bytes")
        }

        try stdinHandle.write(contentsOf: request)
        let responseLine = try readLine()

        guard let responseData = responseLine.data(using: .utf8) else {
            throw SnakeEnvError.pythonConversionFailed("Invalid response encoding")
        }
        let object = try JSONSerialization.jsonObject(with: responseData)
        guard let dict = object as? [String: Any] else {
            throw SnakeEnvError.pythonConversionFailed("Response is not a JSON object")
        }
        return dict
    }

    private func readLine() throws -> String {
        var buffer = Data()
        while true {
            guard let chunk = try stdoutHandle.read(upToCount: 1), !chunk.isEmpty else {
                throw SnakeEnvError.bridgeUnavailable
            }
            if chunk[0] == 0x0A {
                break
            }
            buffer.append(chunk)
        }
        guard let line = String(data: buffer, encoding: .utf8) else {
            throw SnakeEnvError.pythonConversionFailed("Unable to decode response line")
        }
        return line
    }

    private static func resolvePythonModulePath(explicitPath: String?) -> String {
        if let explicitPath, !explicitPath.isEmpty {
            return (explicitPath as NSString).standardizingPath
        }
        if let envPath = ProcessInfo.processInfo.environment["SNAKE_PYTHON_DIR"], !envPath.isEmpty {
            return (envPath as NSString).standardizingPath
        }

        let cwd = FileManager.default.currentDirectoryPath
        let direct = (cwd as NSString).appendingPathComponent("python")
        if FileManager.default.fileExists(atPath: direct) {
            return direct
        }

        return (cwd as NSString).appendingPathComponent("../python")
    }

    private static func pythonExecutable() -> String {
        if let configured = ProcessInfo.processInfo.environment["SNAKE_PYTHON_EXE"], !configured.isEmpty {
            return configured
        }
        if FileManager.default.fileExists(atPath: "/opt/anaconda3/bin/python3") {
            return "/opt/anaconda3/bin/python3"
        }
        return "/usr/bin/python3"
    }
}
