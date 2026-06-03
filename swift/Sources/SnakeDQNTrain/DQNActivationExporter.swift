import Foundation

struct DQNActivationExporter {
    let directory: URL
    let exportEverySteps: Int

    func export(
        inspection: DQNActionInspection,
        episode: Int,
        step: Int
    ) throws {
        guard exportEverySteps > 0, (step - 1) % exportEverySteps == 0 else {
            return
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let prefix = "ep_\(padded(episode, width: 3))_step_\(padded(step, width: 6))"
        try DQNActivationImageWriter.writeTiledChannels(
            inspection.activations.input,
            to: directory.appendingPathComponent("\(prefix)_input.pgm")
        )
        try DQNActivationImageWriter.writeTiledChannels(
            inspection.activations.conv1,
            to: directory.appendingPathComponent("\(prefix)_conv1.pgm")
        )
        try DQNActivationImageWriter.writeTiledChannels(
            inspection.activations.conv2,
            to: directory.appendingPathComponent("\(prefix)_conv2.pgm")
        )
        try DQNActivationImageWriter.writeVectorAsGrid(
            inspection.activations.dense.asType(.float32).asArray(Float.self),
            columns: 16,
            to: directory.appendingPathComponent("\(prefix)_dense.pgm")
        )
        try DQNActivationImageWriter.writeVectorAsGrid(
            inspection.qValues,
            columns: 4,
            to: directory.appendingPathComponent("\(prefix)_qvalues.pgm")
        )
        try qValueText(inspection: inspection)
            .write(
                to: directory.appendingPathComponent("\(prefix)_qvalues.tsv"),
                atomically: true,
                encoding: .utf8
            )
    }

    private func qValueText(inspection: DQNActionInspection) -> String {
        let rows = SnakeAction.allCases.map { action in
            let selected = action == inspection.action ? "1" : "0"
            let qValue = inspection.qValues[action.rawValue]
            return "\(action)\t\(qValue)\t\(selected)"
        }
        return "action\tq_value\tselected\n" + rows.joined(separator: "\n") + "\n"
    }

    private func padded(_ value: Int, width: Int) -> String {
        String(format: "%0\(width)d", value)
    }
}
