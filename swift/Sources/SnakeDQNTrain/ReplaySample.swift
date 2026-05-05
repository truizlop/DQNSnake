import Foundation

struct ReplaySample {
    let transitions: [DQNTransition]
    let indices: [Int]
    let importanceWeights: [Float]
}
