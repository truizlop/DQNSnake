import Foundation

struct DQNEpisodeResult {
    let steps: Int
    let totalReward: Float
    let finalScore: Float
    let applesEaten: Int
    let actionCounts: [SnakeAction: Int]
}
