import Foundation
import MLX

enum DQNActivationImageWriter {
    static func tiledChannelsImage(
        _ array: MLXArray,
        tilePadding: Int = 2,
        transposeSpatial: Bool = false,
        tileLabels: [String] = []
    ) -> DQNGrayscaleImage {
        let shape = array.shape
        precondition(shape.count == 4, "Expected NHWC tensor, got shape \(shape).")
        let sourceHeight = shape[1]
        let sourceWidth = shape[2]
        let tileHeight = transposeSpatial ? sourceWidth : sourceHeight
        let tileWidth = transposeSpatial ? sourceHeight : sourceWidth
        let channels = shape[3]
        let values = array.asType(.float32).asArray(Float.self)

        let columns = Int(ceil(sqrt(Double(channels))))
        let rows = Int(ceil(Double(channels) / Double(columns)))
        let outputWidth = columns * tileWidth + max(0, columns - 1) * tilePadding
        let outputHeight = rows * tileHeight + max(0, rows - 1) * tilePadding
        var pixels = [UInt8](repeating: 0, count: outputWidth * outputHeight)

        for channel in 0..<channels {
            let normalized = normalizedChannel(
                values: values,
                height: sourceHeight,
                width: sourceWidth,
                channels: channels,
                channel: channel,
                transposeSpatial: transposeSpatial
            )
            let tileX = channel % columns
            let tileY = channel / columns
            let originX = tileX * (tileWidth + tilePadding)
            let originY = tileY * (tileHeight + tilePadding)

            for y in 0..<tileHeight {
                for x in 0..<tileWidth {
                    let sourceIndex = y * tileWidth + x
                    let destinationIndex = (originY + y) * outputWidth + (originX + x)
                    pixels[destinationIndex] = normalized[sourceIndex]
                }
            }
        }

        return DQNGrayscaleImage(
            width: outputWidth,
            height: outputHeight,
            pixels: pixels,
            tileWidth: tileWidth,
            tileHeight: tileHeight,
            tileColumns: columns,
            tilePadding: tilePadding,
            tileLabels: tileLabels
        )
    }

    static func vectorGridImage(_ values: [Float], columns: Int) -> DQNGrayscaleImage {
        precondition(columns > 0, "Vector grid requires at least one column.")
        let rows = Int(ceil(Double(values.count) / Double(columns)))
        var padded = values
        padded.append(contentsOf: repeatElement(0, count: rows * columns - values.count))
        return DQNGrayscaleImage(width: columns, height: rows, pixels: normalize(values: padded))
    }

    static func writeTiledChannels(
        _ array: MLXArray,
        to url: URL,
        tilePadding: Int = 2
    ) throws {
        let image = tiledChannelsImage(array, tilePadding: tilePadding)
        try writePGM(pixels: image.pixels, width: image.width, height: image.height, to: url)
    }

    static func writeVectorAsGrid(_ values: [Float], columns: Int, to url: URL) throws {
        let image = vectorGridImage(values, columns: columns)
        try writePGM(pixels: image.pixels, width: image.width, height: image.height, to: url)
    }

    static func writePGM(pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        precondition(pixels.count == width * height, "PGM pixel count does not match dimensions.")
        var data = Data("P5\n\(width) \(height)\n255\n".utf8)
        data.append(contentsOf: pixels)
        try data.write(to: url, options: .atomic)
    }

    private static func normalizedChannel(
        values: [Float],
        height: Int,
        width: Int,
        channels: Int,
        channel: Int,
        transposeSpatial: Bool
    ) -> [UInt8] {
        var channelValues: [Float] = []
        channelValues.reserveCapacity(height * width)

        if transposeSpatial {
            for y in 0..<width {
                for x in 0..<height {
                    let index = ((x * width) + y) * channels + channel
                    channelValues.append(values[index])
                }
            }
        } else {
            for y in 0..<height {
                for x in 0..<width {
                    let index = ((y * width) + x) * channels + channel
                    channelValues.append(values[index])
                }
            }
        }

        return normalize(values: channelValues)
    }

    private static func normalize(values: [Float]) -> [UInt8] {
        guard let minValue = values.min(), let maxValue = values.max() else {
            return []
        }
        let range = maxValue - minValue
        guard range.isFinite, range > 0 else {
            return [UInt8](repeating: 0, count: values.count)
        }
        return values.map { value in
            let normalized = max(0, min(1, (value - minValue) / range))
            return UInt8((normalized * 255).rounded())
        }
    }
}
