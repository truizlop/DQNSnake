import Foundation

enum DQNTrainingGuardError: Error, LocalizedError {
    case consecutiveSkippedUpdatesExceeded(limit: Int, observed: Int, step: Int)
    case lossExceeded(limit: Float, observed: Float, step: Int)
    case qValueExceeded(limit: Float, observed: Float, step: Int)
    case gradientNormExceeded(limit: Float, observed: Float, step: Int)

    var errorDescription: String? {
        switch self {
        case let .consecutiveSkippedUpdatesExceeded(limit, observed, step):
            return
                "stability guard triggered at step \(step): consecutive skipped updates \(observed) exceeded limit \(limit)"
        case let .lossExceeded(limit, observed, step):
            return "stability guard triggered at step \(step): loss \(observed) exceeded limit \(limit)"
        case let .qValueExceeded(limit, observed, step):
            return "stability guard triggered at step \(step): q-value magnitude \(observed) exceeded limit \(limit)"
        case let .gradientNormExceeded(limit, observed, step):
            return "stability guard triggered at step \(step): gradient L2 \(observed) exceeded limit \(limit)"
        }
    }
}
