import Foundation

enum DQNLearningRateSchedule {
    static func learningRate(
        at globalStep: Int,
        initial: Float,
        final: Float?,
        decayStartStep: Int?,
        decayEndStep: Int?
    ) -> Float {
        guard
            let final,
            let decayStartStep,
            let decayEndStep,
            decayEndStep > decayStartStep
        else {
            return initial
        }

        if globalStep <= decayStartStep {
            return initial
        }
        if globalStep >= decayEndStep {
            return final
        }

        let progress = Float(globalStep - decayStartStep) / Float(decayEndStep - decayStartStep)
        return initial + (final - initial) * progress
    }
}
