import Foundation
import MLX
import Testing
@testable import SnakeDQNTrain

@Test func prioritizedReplaySamplesHighPriorityIndicesMoreFrequently() {
    var buffer = ReplayBuffer(
        capacity: 4,
        samplingStrategy: .prioritized,
        prioritizedAlpha: 0.6,
        prioritizedEpsilon: 1e-3,
        seed: 123
    )
    for i in 0..<4 {
        buffer.append(makeTransition(id: i))
    }

    buffer.updatePriorities(indices: [0, 1, 2, 3], tdErrors: [100, 1, 1, 1])

    var counts = [0, 0, 0, 0]
    for _ in 0..<4000 {
        let sample = buffer.sample(batchSize: 1, importanceSamplingBeta: 1)
        counts[sample.indices[0]] += 1
    }

    let dominant = Float(counts[0]) / 4000
    #expect(dominant > 0.55)
    #expect(counts[0] > counts[1])
    #expect(counts[0] > counts[2])
    #expect(counts[0] > counts[3])
}

@Test func prioritizedReplayRespondsToUpdatedPriorities() {
    var buffer = ReplayBuffer(
        capacity: 4,
        samplingStrategy: .prioritized,
        prioritizedAlpha: 0.6,
        prioritizedEpsilon: 1e-3,
        seed: 42
    )
    for i in 0..<4 {
        buffer.append(makeTransition(id: i))
    }

    var baselineCounts = [0, 0, 0, 0]
    for _ in 0..<2000 {
        let sample = buffer.sample(batchSize: 1, importanceSamplingBeta: 1)
        baselineCounts[sample.indices[0]] += 1
    }
    let baselineRate = Float(baselineCounts[3]) / 2000

    buffer.updatePriorities(indices: [0, 1, 2, 3], tdErrors: [1, 1, 1, 50])

    var boostedCounts = [0, 0, 0, 0]
    for _ in 0..<2000 {
        let sample = buffer.sample(batchSize: 1, importanceSamplingBeta: 1)
        boostedCounts[sample.indices[0]] += 1
    }
    let boostedRate = Float(boostedCounts[3]) / 2000

    #expect(boostedRate > baselineRate + 0.20)
}

@Test func prioritizedReplayImportanceWeightsAreNormalizedAndInverseToPriority() {
    let alpha: Float = 0.6
    let epsilon: Float = 1e-3
    let tdErrors: [Float] = [10, 1, 0.1, 0.01]
    var buffer = ReplayBuffer(
        capacity: tdErrors.count,
        samplingStrategy: .prioritized,
        prioritizedAlpha: alpha,
        prioritizedEpsilon: epsilon,
        seed: 7
    )
    for i in 0..<tdErrors.count {
        buffer.append(makeTransition(id: i))
    }
    buffer.updatePriorities(indices: [0, 1, 2, 3], tdErrors: tdErrors)

    let sample = buffer.sample(batchSize: 4, importanceSamplingBeta: 1)
    let priorities = tdErrors.map { Foundation.pow(Swift.abs($0) + epsilon, alpha) }

    #expect(sample.importanceWeights.count == sample.indices.count)
    #expect(sample.importanceWeights.max() ?? 0 <= 1.00001)
    #expect(sample.importanceWeights.max() ?? 0 > 0.95)
    #expect(sample.importanceWeights.allSatisfy { $0.isFinite && $0 > 0 })

    for i in 0..<sample.indices.count {
        for j in 0..<sample.indices.count where i != j {
            let pi = priorities[sample.indices[i]]
            let pj = priorities[sample.indices[j]]
            let wi = sample.importanceWeights[i]
            let wj = sample.importanceWeights[j]
            if pi > pj {
                #expect(wi <= wj + 1e-5)
            }
        }
    }
}

@Test func prioritizedReplayNewlyOverwrittenTransitionGetsMaxPriority() {
    var buffer = ReplayBuffer(
        capacity: 3,
        samplingStrategy: .prioritized,
        prioritizedAlpha: 0.6,
        prioritizedEpsilon: 1e-3,
        seed: 999
    )
    buffer.append(makeTransition(id: 0))
    buffer.append(makeTransition(id: 1))
    buffer.append(makeTransition(id: 2))

    buffer.updatePriorities(indices: [0, 1, 2], tdErrors: [0.01, 0.01, 100])
    buffer.append(makeTransition(id: 99)) // overwrites index 0, should inherit max priority.

    var replacementHits = 0
    for _ in 0..<1500 {
        let sample = buffer.sample(batchSize: 1, importanceSamplingBeta: 1)
        if sample.indices[0] == 0 {
            replacementHits += 1
        }
    }
    #expect(Float(replacementHits) / 1500 > 0.45)
}

@Test func prioritizedReplayMaxPriorityTracksCurrentBufferNotHistoricalPeak() {
    var buffer = ReplayBuffer(
        capacity: 4,
        samplingStrategy: .prioritized,
        prioritizedAlpha: 0.6,
        prioritizedEpsilon: 1e-3,
        seed: 2026
    )
    for i in 0..<4 {
        buffer.append(makeTransition(id: i))
    }

    // Historical spike.
    buffer.updatePriorities(indices: [0], tdErrors: [100])
    // Later all priorities become small; maxPriority should drop accordingly.
    buffer.updatePriorities(indices: [0, 1, 2, 3], tdErrors: [0.1, 0.1, 0.1, 0.1])
    // New transition should not inherit stale huge priority.
    buffer.append(makeTransition(id: 99))

    var counts = [0, 0, 0, 0]
    for _ in 0..<2000 {
        let sample = buffer.sample(batchSize: 1, importanceSamplingBeta: 1)
        counts[sample.indices[0]] += 1
    }

    // In near-uniform priorities, no single index should dominate.
    let maxRate = Float(counts.max() ?? 0) / 2000
    #expect(maxRate < 0.45)
}

@Test func prioritizedReplayWithEqualPrioritiesProducesFiniteUnitWeights() {
    var buffer = ReplayBuffer(
        capacity: 4,
        samplingStrategy: .prioritized,
        prioritizedAlpha: 0.6,
        prioritizedEpsilon: 1e-3,
        seed: 12
    )
    for i in 0..<4 {
        buffer.append(makeTransition(id: i))
    }
    buffer.updatePriorities(indices: [0, 1, 2, 3], tdErrors: [1, 1, 1, 1])

    let sample = buffer.sample(batchSize: 4, importanceSamplingBeta: 1)
    #expect(sample.importanceWeights.allSatisfy { $0.isFinite })
    #expect(sample.importanceWeights.allSatisfy { abs($0 - 1) < 1e-5 })
}

private func makeTransition(id: Int) -> DQNTransition {
    DQNTransition(
        state: MLXArray([Float(id)]).reshaped(1, 1),
        action: .left,
        reward: Float(id),
        nextState: MLXArray([Float(id + 1)]).reshaped(1, 1),
        done: false
    )
}
