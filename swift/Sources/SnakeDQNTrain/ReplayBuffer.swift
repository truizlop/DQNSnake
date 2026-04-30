import Foundation

struct ReplayBuffer {
    private let capacity: Int
    private let samplingStrategy: ReplaySamplingStrategy
    private var storage: [DQNTransition]
    private var writeIndex: Int = 0
    private(set) var count: Int = 0

    init(capacity: Int, samplingStrategy: ReplaySamplingStrategy = .withReplacement) {
        precondition(capacity > 0, "ReplayBuffer capacity must be > 0.")
        self.capacity = capacity
        self.samplingStrategy = samplingStrategy
        self.storage = []
        self.storage.reserveCapacity(capacity)
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

    func sample(batchSize: Int) -> [DQNTransition] {
        precondition(batchSize > 0, "batchSize must be > 0.")
        precondition(canSample(batchSize: batchSize), "Not enough transitions to sample.")
        switch samplingStrategy {
        case .withReplacement:
            return (0..<batchSize).map { _ in
                storage[Int.random(in: 0..<count)]
            }
        case .withoutReplacement:
            return Array(storage[0..<count].shuffled().prefix(batchSize))
        }
    }
}
