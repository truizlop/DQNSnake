import Testing
@testable import SnakeDQNTrain

@Test func stateTensorLayoutInterleavesTemporalChannelsPerPixel() {
    let width = 2
    let height = 2
    // frame-major stacked order: [f0 pixels..., f1 pixels..., f2..., f3...]
    // each frame has 4 pixels.
    let stacked: [UInt8] = [
        10, 11, 12, 13, // frame 0
        20, 21, 22, 23, // frame 1
        30, 31, 32, 33, // frame 2
        40, 41, 42, 43, // frame 3
    ]
    let interleaved = StateTensorLayout.interleaveStackedFrames(
        stackedFrames: stacked,
        width: width,
        height: height
    )

    // Pixel 0 channels should be [10,20,30,40], then pixel 1 [11,21,31,41], etc.
    #expect(interleaved == [
        10, 20, 30, 40,
        11, 21, 31, 41,
        12, 22, 32, 42,
        13, 23, 33, 43,
    ])
}
