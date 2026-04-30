import Foundation

final class TensorBoardMetricsPublisher {
    private let logDir: String
    private let launchTensorBoard: Bool
    private let tensorBoardPort: Int
    private let pythonExecutable: String

    private var writerProcess: Process?
    private var writerStdin: FileHandle?
    private var tensorBoardProcess: Process?

    init(
        logDir: String,
        launchTensorBoard: Bool,
        tensorBoardPort: Int,
        pythonExecutable: String
    ) {
        self.logDir = logDir
        self.launchTensorBoard = launchTensorBoard
        self.tensorBoardPort = tensorBoardPort
        self.pythonExecutable = pythonExecutable
    }

    func start() throws {
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: logDir, isDirectory: true),
            withIntermediateDirectories: true
        )

        let modulePath = resolvePythonModulePath()
        let streamScript = (modulePath as NSString).appendingPathComponent("tensorboard_stream.py")
        guard FileManager.default.fileExists(atPath: streamScript) else {
            throw NSError(
                domain: "TensorBoardMetricsPublisher",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tensorboard_stream.py not found at \(streamScript)"]
            )
        }

        let writer = Process()
        writer.executableURL = URL(fileURLWithPath: pythonExecutable)
        writer.arguments = [streamScript, "--logdir", logDir]
        writer.standardError = FileHandle.standardError
        let stdinPipe = Pipe()
        writer.standardInput = stdinPipe
        try writer.run()
        self.writerProcess = writer
        self.writerStdin = stdinPipe.fileHandleForWriting
        // If the Python writer exits immediately (e.g. missing tensorboard package),
        // fail fast here instead of crashing later on pipe writes.
        usleep(200_000)
        if !writer.isRunning {
            throw NSError(
                domain: "TensorBoardMetricsPublisher",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "tensorboard stream writer exited early. Ensure `tensorboard` is installed for \(pythonExecutable)."
                ]
            )
        }

        if launchTensorBoard {
            let tensorBoard = Process()
            tensorBoard.executableURL = URL(fileURLWithPath: pythonExecutable)
            tensorBoard.arguments = [
                "-m", "tensorboard.main", "--logdir", logDir, "--port", String(tensorBoardPort),
                "--reload_interval", "1",
            ]
            tensorBoard.standardError = FileHandle.standardError
            tensorBoard.standardOutput = FileHandle.standardOutput
            try tensorBoard.run()
            self.tensorBoardProcess = tensorBoard
            print("TensorBoard running at http://localhost:\(tensorBoardPort)")
        }
    }

    func publish(step: Int, scalars: [String: Float]) {
        guard let writerStdin else {
            return
        }
        let payload: [String: Any] = ["step": step, "scalars": scalars]
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            var line = data
            line.append(0x0A)
            try writerStdin.write(contentsOf: line)
        } catch {
            fputs("TensorBoard publish error: \(error)\n", stderr)
        }
    }

    func stop() {
        do {
            try writerStdin?.close()
        } catch {
            // no-op
        }
        writerStdin = nil

        if let writerProcess, writerProcess.isRunning {
            writerProcess.terminate()
        }
        if let tensorBoardProcess, tensorBoardProcess.isRunning {
            tensorBoardProcess.terminate()
        }
        writerProcess = nil
        tensorBoardProcess = nil
    }

    private func resolvePythonModulePath() -> String {
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
}
