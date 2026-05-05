import Foundation

// Controls how replay minibatches are sampled:
// - withReplacement: each draw is independent from the full buffer, so duplicates
//   can appear in the same minibatch.
// - withoutReplacement: sampled transitions are unique within a minibatch.
//
// This is an algorithmic choice (not just an implementation detail):
// with-replacement sampling can increase gradient noise due to repeated items,
// while without-replacement sampling tends to provide more diverse batches.
// Keeping it explicit in config improves reproducibility and experimentation.
enum ReplaySamplingStrategy: Equatable, CustomStringConvertible {
    case withReplacement
    case withoutReplacement
    case prioritized

    init?(envValue: String) {
        switch envValue.lowercased() {
        case "with_replacement", "withreplacement", "uniform_with_replacement":
            self = .withReplacement
        case "without_replacement", "withoutreplacement", "uniform_without_replacement":
            self = .withoutReplacement
        case "prioritized", "per":
            self = .prioritized
        default:
            return nil
        }
    }

    var description: String {
        switch self {
        case .withReplacement:
            return "with_replacement"
        case .withoutReplacement:
            return "without_replacement"
        case .prioritized:
            return "prioritized"
        }
    }
}
