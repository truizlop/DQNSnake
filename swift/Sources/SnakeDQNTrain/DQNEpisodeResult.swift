import Foundation

struct DQNEpisodeResult {
    let steps: Int
    let totalReward: Float
    let finalScore: Float
    let actionCounts: [SnakeAction: Int]
}
