import Foundation
import Testing
@testable import SnakeDQNTrain

@Test func gradientClipScaleIsOneWhenNoClippingNeeded() {
    #expect(DQNLearner.gradientClipScale(gradientL2Norm: 5, maxNorm: 10) == 1)
    #expect(DQNLearner.gradientClipScale(gradientL2Norm: 10, maxNorm: 10) == 1)
    #expect(DQNLearner.gradientClipScale(gradientL2Norm: .infinity, maxNorm: 10) == 1)
    #expect(DQNLearner.gradientClipScale(gradientL2Norm: 5, maxNorm: 0) == 1)
}

@Test func gradientClipScaleReducesLargeGradientNorm() {
    let scale = DQNLearner.gradientClipScale(gradientL2Norm: 100, maxNorm: 10)
    #expect(abs(scale - 0.1) < 1e-6)
}
