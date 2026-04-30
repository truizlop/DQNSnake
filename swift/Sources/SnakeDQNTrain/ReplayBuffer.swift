import Foundation

struct ReplayBuffer {
    private let capacity: Int
    private let samplingStrategy: ReplaySamplingStrategy
    private var storage: [DQNTransition]
    private var writeIndex: Int = 0
    private(set) var count: Int = 0
    private var rng: SeededRandomNumberGenerator?

    init(capacity: Int, samplingStrategy: ReplaySamplingStrategy = .withReplacement, seed: Int? = nil) {
        precondition(capacity > 0, "ReplayBuffer capacity must be > 0.")
        self.capacity = capacity
        self.samplingStrategy = samplingStrategy
        self.storage = []
        self.storage.reserveCapacity(capacity)
        if let seed {
            self.rng = SeededRandomNumberGenerator(seed: UInt64(bitPattern: Int64(seed)))
        } else {
            self.rng = nil
        }
    }

    mutating func append(_ transition: DQNTransition) {
        if count < capacity {
            storage.append(transition)
            count += 1
        } else {
            storage[writeIndex] = transition
        }
        writeIndex = (writeIndex + 1) % capacity
    }

    func canSample(batchSize: Int) -> Bool {
        count >= batchSize
    }

    mutating func sample(batchSize: Int) -> [DQNTransition] {
        precondition(batchSize > 0, "batchSize must be > 0.")
        precondition(canSample(batchSize: batchSize), "Not enough transitions to sample.")
        switch samplingStrategy {
        case .withReplacement:
            return sampleWithReplacement(batchSize: batchSize)
        case .withoutReplacement:
            return sampleWithoutReplacement(batchSize: batchSize)
        }
    }

    private mutating func sampleWithReplacement(batchSize: Int) -> [DQNTransition] {
        if var rng {
            let batch = (0..<batchSize).map { _ in
                storage[Int.random(in: 0..<count, using: &rng)]
            }
            self.rng = rng
            return batch
        }
        return (0..<batchSize).map { _ in
            storage[Int.random(in: 0..<count)]
        }
    }

    private mutating func sampleWithoutReplacement(batchSize: Int) -> [DQNTransition] {
        var indices = Array(0..<count)
        if var rng {
            indices.shuffle(using: &rng)
            self.rng = rng
        } else {
            indices.shuffle()
        }
        return indices.prefix(batchSize).map { storage[$0] }
    }
}
