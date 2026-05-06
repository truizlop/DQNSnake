import Foundation

enum PrioritizedReplayBetaSchedule {
    static func beta(
        globalStep: Int,
        start: Float,
        annealSteps: Int
    ) -> Float {
        guard annealSteps > 0 else {
            return 1
        }
        let clamped = min(max(globalStep, 0), annealSteps)
        let progress = Float(clamped) / Float(annealSteps)
        let beta = start + (1 - start) * progress
        return min(max(beta, 0), 1)
    }
}
