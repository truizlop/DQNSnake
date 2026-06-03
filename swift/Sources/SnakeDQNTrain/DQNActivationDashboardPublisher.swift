import Foundation

final class DQNActivationDashboardPublisher {
    private let pythonExecutable: String
    private let pythonModulePath: String
    private var process: Process?
    private var stdin: FileHandle?

    init(pythonExecutable: String, pythonModulePath: String) {
        self.pythonExecutable = pythonExecutable
        self.pythonModulePath = pythonModulePath
    }

    func start() throws {
        let scriptPath = (pythonModulePath as NSString).appendingPathComponent("activation_viewer.py")
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            throw NSError(
                domain: "DQNActivationDashboardPublisher",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "activation_viewer.py not found at \(scriptPath)"]
            )
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = [scriptPath]
        process.standardError = FileHandle.standardError
        let inputPipe = Pipe()
        process.standardInput = inputPipe
        try process.run()

        self.process = process
        self.stdin = inputPipe.fileHandleForWriting
    }

    func publish(inspection: DQNActionInspection, episode: Int, step: Int) {
        guard let stdin else {
            return
        }

        do {
            let payload: [String: Any] = [
                "episode": episode,
                "step": step,
                "action": "\(inspection.action)",
                "q_values": inspection.qValues,
                "selected_action_index": inspection.action.rawValue,
                "images": [
                    "input": imagePayload(
                        DQNActivationImageWriter.tiledChannelsImage(
                            inspection.activations.input,
                            transposeSpatial: true,
                            tileLabels: ["t-3", "t-2", "t-1", "t"]
                        )
                    ),
                    "conv1": imagePayload(
                        DQNActivationImageWriter.tiledChannelsImage(
                            inspection.activations.conv1,
                            transposeSpatial: true,
                            tileLabels: (0..<16).map { "c\($0)" }
                        )
                    ),
                    "conv2": imagePayload(
                        DQNActivationImageWriter.tiledChannelsImage(
                            inspection.activations.conv2,
                            transposeSpatial: true,
                            tileLabels: (0..<32).map { "c\($0)" }
                        )
                    ),
                    "dense": imagePayload(
                        DQNActivationImageWriter.vectorGridImage(
                            inspection.activations.dense.asType(.float32).asArray(Float.self),
                            columns: 16
                        )
                    ),
                    "qvalues": imagePayload(
                        DQNActivationImageWriter.vectorGridImage(inspection.qValues, columns: 4)
                    ),
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: payload)
            var line = data
            line.append(0x0A)
            try stdin.write(contentsOf: line)
        } catch {
            fputs("Activation dashboard publish error: \(error)\n", stderr)
        }
    }

    func stop() {
        do {
            try stdin?.close()
        } catch {
            // no-op
        }
        stdin = nil

        if let process, process.isRunning {
            process.terminate()
        }
        process = nil
    }

    private func imagePayload(_ image: DQNGrayscaleImage) -> [String: Any] {
        var payload: [String: Any] = [
            "width": image.width,
            "height": image.height,
            "pixels": image.pixels.map(Int.init),
            "tile_labels": image.tileLabels,
        ]
        if let tileWidth = image.tileWidth {
            payload["tile_width"] = tileWidth
        }
        if let tileHeight = image.tileHeight {
            payload["tile_height"] = tileHeight
        }
        if let tileColumns = image.tileColumns {
            payload["tile_columns"] = tileColumns
        }
        if let tilePadding = image.tilePadding {
            payload["tile_padding"] = tilePadding
        }
        return payload
    }
}
