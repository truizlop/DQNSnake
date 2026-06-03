import Foundation

enum DQNEvaluationOnlyError: LocalizedError {
    case missingCheckpoint

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "SNAKE_EVAL_ONLY requires SNAKE_RESUME_CHECKPOINT."
        }
    }
}
