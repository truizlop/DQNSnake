import Foundation

struct DQNStabilityTracker {
    private(set) var consecutiveSkippedUpdates: Int = 0

    mutating func record(trainStepMetrics: DQNTrainStepMetrics) {
        if trainStepMetrics.skippedUpdate {
            consecutiveSkippedUpdates += 1
        } else {
            consecutiveSkippedUpdates = 0
        }
    }
}
