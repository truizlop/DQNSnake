import Foundation

final class PythonSnakeBridge: SnakeBridgeClient, @unchecked Sendable {
    private struct BaseResponse: Decodable {
        let ok: Bool
        let error: String?
    }

    private struct EmptyResponse: Decodable {}

    private struct CreateEnvRequest: Encodable {
        let cmd: String = "create_env"
        let envName: String?
        let seed: Int?

        enum CodingKeys: String, CodingKey {
            case cmd
            case envName = "env_name"
            case seed
        }
    }

    private struct ResetRequest: Encodable {
        let cmd: String = "reset"
        let seed: Int?
    }

    private struct ResetResponse: Decodable {
        let frame: [[UInt8]]
    }

    private struct StepRequest: Encodable {
        let cmd: String = "step"
        let action: Int
    }

    private struct StepResponse: Decodable {
        let observation: [[UInt8]]
        let reward: Float
        let done: Bool
        let score: Float
    }

    private struct ScoreRequest: Encodable {
        let cmd: String = "score"
    }

    private struct ScoreResponse: Decodable {
        let score: Float
    }

    private struct IsDoneRequest: Encodable {
        let cmd: String = "is_done"
    }

    private struct IsDoneResponse: Decodable {
        let done: Bool
    }

    private struct RenderRequest: Encodable {
        let cmd: String = "render"
    }

    private struct SaveLastEpisodeGIFRequest: Encodable {
        let cmd: String = "save_last_episode_gif"
        let path: String
        let scale: Int
        let frameDurationMs: Int

        enum CodingKeys: String, CodingKey {
            case cmd
            case path
            case scale
            case frameDurationMs = "frame_duration_ms"
        }
    }

    private struct SaveLastEpisodeGIFResponse: Decodable {
        let frameCount: Int

        enum CodingKeys: String, CodingKey {
            case frameCount = "frame_count"
        }
    }

    private struct QuitRequest: Encodable {
        let cmd: String = "quit"
    }

    private let process: Process
    private let stdinHandle: FileHandle
    private let stdoutHandle: FileHandle
    private let lock = NSLock()
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private init(process: Process, stdinHandle: FileHandle, stdoutHandle: FileHandle) {
        self.process = process
        self.stdinHandle = stdinHandle
        self.stdoutHandle = stdoutHandle
    }

    deinit {
        _ = try? sendRaw(QuitRequest(), as: EmptyResponse.self)
        if process.isRunning {
            process.terminate()
        }
    }

    static func make(pythonModulePath: String?, seed: Int?) -> PythonSnakeBridge? {
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
            _ = try bridge.send(CreateEnvRequest(envName: nil, seed: seed), as: EmptyResponse.self)
            return bridge
        } catch {
            if process.isRunning {
                process.terminate()
            }
            return nil
        }
    }

    func reset(seed: Int?) throws -> [[UInt8]] {
        let response = try send(ResetRequest(seed: seed), as: ResetResponse.self)
        return response.frame
    }

    func step(action: Int) throws -> (observation: [[UInt8]], reward: Float, done: Bool, score: Float) {
        let response = try send(StepRequest(action: action), as: StepResponse.self)

        return (
            response.observation,
            response.reward,
            response.done,
            response.score
        )
    }

    func score() -> Float {
        do {
            let response = try send(ScoreRequest(), as: ScoreResponse.self)
            return response.score
        } catch {
            return 0
        }
    }

    func isDone() -> Bool {
        do {
            let response = try send(IsDoneRequest(), as: IsDoneResponse.self)
            return response.done
        } catch {
            return false
        }
    }

    func render() throws {
        _ = try send(RenderRequest(), as: EmptyResponse.self)
    }

    func saveLastEpisodeGIF(path: String, scale: Int, frameDurationMs: Int) throws -> Int {
        let response = try send(
            SaveLastEpisodeGIFRequest(path: path, scale: scale, frameDurationMs: frameDurationMs),
            as: SaveLastEpisodeGIFResponse.self
        )
        return response.frameCount
    }

    private func send<Request: Encodable, Response: Decodable>(_ payload: Request, as: Response.Type) throws
        -> Response
    {
        let (base, body) = try sendRaw(payload, as: Response.self)
        if !base.ok {
            let error = base.error ?? "Python bridge error"
            throw SnakeEnvError.pythonBridgeInitializationFailed(error)
        }
        return body
    }

    private func sendRaw<Request: Encodable, Response: Decodable>(_ payload: Request, as: Response.Type) throws
        -> (BaseResponse, Response)
    {
        try autoreleasepool {
            lock.lock()
            defer { lock.unlock() }

            let requestData = try encoder.encode(payload)
            var lineData = requestData
            lineData.append(0x0A)

            try stdinHandle.write(contentsOf: lineData)
            let responseData = try readLineData()

            let base = try decodeResponse(BaseResponse.self, from: responseData)
            let body = try decodeResponse(Response.self, from: responseData)
            return (base, body)
        }
    }

    private func decodeResponse<Response: Decodable>(_ type: Response.Type, from data: Data) throws -> Response {
        do {
            return try decoder.decode(type, from: data)
        } catch {
            let raw = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw SnakeEnvError.pythonConversionFailed(
                "Unable to decode \(String(describing: type)) from response: \(raw)"
            )
        }
    }

    private func readLineData() throws -> Data {
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
        return buffer
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
