import Foundation

struct EpsilonGreedyPolicy {
    let epsilonStart: Float
    let epsilonEnd: Float
    let epsilonDecaySteps: Int

    init(epsilonStart: Float, epsilonEnd: Float, epsilonDecaySteps: Int) {
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = epsilonDecaySteps
    }

    func selectAction(globalStep: Int, greedyAction: SnakeAction) -> SnakeAction {
        if Float.random(in: 0..<1) < epsilonValue(globalStep: globalStep) {
            return SnakeAction.allCases.randomElement()!
        }
        return greedyAction
    }

    func epsilon(at globalStep: Int) -> Float {
        epsilonValue(globalStep: globalStep)
    }

    private func epsilonValue(globalStep: Int) -> Float {
        guard epsilonDecaySteps > 0 else {
            return epsilonEnd
        }

        let clampedStep = min(max(globalStep, 0), epsilonDecaySteps)
        let progress = Float(clampedStep) / Float(epsilonDecaySteps)
        let epsilon = epsilonStart + (epsilonEnd - epsilonStart) * progress
        return min(max(epsilon, min(epsilonStart, epsilonEnd)), max(epsilonStart, epsilonEnd))
    }
}
