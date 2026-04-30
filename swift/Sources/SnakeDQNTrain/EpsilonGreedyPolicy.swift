import Foundation

struct EpsilonGreedyPolicy {
    let epsilonStart: Float
    let epsilonEnd: Float
    let epsilonDecaySteps: Int
    private var rng: SeededRandomNumberGenerator?

    init(epsilonStart: Float, epsilonEnd: Float, epsilonDecaySteps: Int, seed: Int? = nil) {
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = epsilonDecaySteps
        if let seed {
            self.rng = SeededRandomNumberGenerator(seed: UInt64(bitPattern: Int64(seed)))
        } else {
            self.rng = nil
        }
    }

    mutating func selectAction(globalStep: Int, greedyAction: SnakeAction) -> SnakeAction {
        if randomUnit() < epsilonValue(globalStep: globalStep) {
            return randomAction()
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

    private mutating func randomUnit() -> Float {
        if var rng {
            let value = Float.random(in: 0..<1, using: &rng)
            self.rng = rng
            return value
        }
        return Float.random(in: 0..<1)
    }

    private mutating func randomAction() -> SnakeAction {
        if var rng {
            let index = Int.random(in: 0..<SnakeAction.allCases.count, using: &rng)
            self.rng = rng
            return SnakeAction.allCases[index]
        }
        return SnakeAction.allCases.randomElement()!
    }
}
