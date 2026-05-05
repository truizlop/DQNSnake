import Foundation

struct ReplayBuffer {
    private let capacity: Int
    private let samplingStrategy: ReplaySamplingStrategy
    private let prioritizedAlpha: Float
    private let prioritizedEpsilon: Float
    private var storage: [DQNTransition]
    private var priorities: [Float]
    private var writeIndex: Int = 0
    private(set) var count: Int = 0
    private var rng: SeededRandomNumberGenerator?
    private var maxPriority: Float = 1

    init(
        capacity: Int,
        samplingStrategy: ReplaySamplingStrategy = .withReplacement,
        prioritizedAlpha: Float = 0.6,
        prioritizedEpsilon: Float = 1e-3,
        seed: Int? = nil
    ) {
        precondition(capacity > 0, "ReplayBuffer capacity must be > 0.")
        precondition(prioritizedAlpha >= 0, "prioritizedAlpha must be >= 0.")
        precondition(prioritizedEpsilon > 0, "prioritizedEpsilon must be > 0.")
        self.capacity = capacity
        self.samplingStrategy = samplingStrategy
        self.prioritizedAlpha = prioritizedAlpha
        self.prioritizedEpsilon = prioritizedEpsilon
        self.storage = []
        self.storage.reserveCapacity(capacity)
        self.priorities = []
        self.priorities.reserveCapacity(capacity)
        if let seed {
            self.rng = SeededRandomNumberGenerator(seed: UInt64(bitPattern: Int64(seed)))
        } else {
            self.rng = nil
        }
    }

    mutating func append(_ transition: DQNTransition) {
        if count < capacity {
            storage.append(transition)
            priorities.append(maxPriority)
            count += 1
        } else {
            storage[writeIndex] = transition
            priorities[writeIndex] = maxPriority
        }
        writeIndex = (writeIndex + 1) % capacity
    }

    func canSample(batchSize: Int) -> Bool {
        count >= batchSize
    }

    mutating func sample(batchSize: Int, importanceSamplingBeta: Float = 1) -> ReplaySample {
        precondition(batchSize > 0, "batchSize must be > 0.")
        precondition(canSample(batchSize: batchSize), "Not enough transitions to sample.")
        precondition(importanceSamplingBeta >= 0 && importanceSamplingBeta <= 1, "importanceSamplingBeta must be in [0, 1].")
        switch samplingStrategy {
        case .withReplacement:
            return sampleWithReplacement(batchSize: batchSize)
        case .withoutReplacement:
            return sampleWithoutReplacement(batchSize: batchSize)
        case .prioritized:
            return samplePrioritized(batchSize: batchSize, beta: importanceSamplingBeta)
        }
    }

    mutating func updatePriorities(indices: [Int], tdErrors: [Float]) {
        precondition(indices.count == tdErrors.count, "indices and tdErrors must have equal length.")
        guard samplingStrategy == .prioritized else {
            return
        }
        for (index, error) in zip(indices, tdErrors) {
            precondition(index >= 0 && index < count, "Priority index out of bounds.")
            let priority = pow(abs(error) + prioritizedEpsilon, prioritizedAlpha)
            priorities[index] = priority
            if priority > maxPriority {
                maxPriority = priority
            }
        }
    }

    private mutating func sampleWithReplacement(batchSize: Int) -> ReplaySample {
        var indices: [Int] = []
        indices.reserveCapacity(batchSize)
        if var rng {
            for _ in 0..<batchSize {
                indices.append(Int.random(in: 0..<count, using: &rng))
            }
            self.rng = rng
        } else {
            for _ in 0..<batchSize {
                indices.append(Int.random(in: 0..<count))
            }
        }
        return ReplaySample(
            transitions: indices.map { storage[$0] },
            indices: indices,
            importanceWeights: Array(repeating: 1, count: batchSize)
        )
    }

    private mutating func sampleWithoutReplacement(batchSize: Int) -> ReplaySample {
        var indices = Array(0..<count)
        if var rng {
            indices.shuffle(using: &rng)
            self.rng = rng
        } else {
            indices.shuffle()
        }
        let selectedIndices = Array(indices.prefix(batchSize))
        return ReplaySample(
            transitions: selectedIndices.map { storage[$0] },
            indices: selectedIndices,
            importanceWeights: Array(repeating: 1, count: batchSize)
        )
    }

    private mutating func samplePrioritized(batchSize: Int, beta: Float) -> ReplaySample {
        let activePriorities = Array(priorities.prefix(count))
        let prioritySum = activePriorities.reduce(Float(0), +)
        if prioritySum <= 0 {
            return sampleWithReplacement(batchSize: batchSize)
        }

        var cumulative: [Float] = []
        cumulative.reserveCapacity(count)
        var running: Float = 0
        for p in activePriorities {
            running += p
            cumulative.append(running)
        }

        var sampledIndices: [Int] = []
        sampledIndices.reserveCapacity(batchSize)
        if var rng {
            for _ in 0..<batchSize {
                let draw = Float.random(in: 0..<running, using: &rng)
                sampledIndices.append(Self.binarySearchCumulative(cumulative, value: draw))
            }
            self.rng = rng
        } else {
            for _ in 0..<batchSize {
                let draw = Float.random(in: 0..<running)
                sampledIndices.append(Self.binarySearchCumulative(cumulative, value: draw))
            }
        }

        let probabilities = sampledIndices.map { activePriorities[$0] / prioritySum }
        let n = Float(count)
        let unnormalizedWeights = probabilities.map { pow(n * $0, -beta) }
        let maxWeight = unnormalizedWeights.max() ?? 1
        let normalizedWeights = unnormalizedWeights.map { $0 / maxWeight }

        return ReplaySample(
            transitions: sampledIndices.map { storage[$0] },
            indices: sampledIndices,
            importanceWeights: normalizedWeights
        )
    }

    private static func binarySearchCumulative(_ cumulative: [Float], value: Float) -> Int {
        var low = 0
        var high = cumulative.count - 1
        while low < high {
            let mid = (low + high) / 2
            if value < cumulative[mid] {
                high = mid
            } else {
                low = mid + 1
            }
        }
        return low
    }
}
