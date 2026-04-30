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
enum ReplaySamplingStrategy {
    case withReplacement
    case withoutReplacement
}
