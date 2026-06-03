import Foundation

struct DQNGrayscaleImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]
    let tileWidth: Int?
    let tileHeight: Int?
    let tileColumns: Int?
    let tilePadding: Int?
    let tileLabels: [String]

    init(
        width: Int,
        height: Int,
        pixels: [UInt8],
        tileWidth: Int? = nil,
        tileHeight: Int? = nil,
        tileColumns: Int? = nil,
        tilePadding: Int? = nil,
        tileLabels: [String] = []
    ) {
        self.width = width
        self.height = height
        self.pixels = pixels
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tileColumns = tileColumns
        self.tilePadding = tilePadding
        self.tileLabels = tileLabels
    }
}
