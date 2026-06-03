import Foundation

enum DQNPlayOnlyError: LocalizedError {
    case missingCheckpoint

    var errorDescription: String? {
        switch self {
        case .missingCheckpoint:
            return "SNAKE_PLAY_ONLY requires SNAKE_RESUME_CHECKPOINT."
        }
    }
}
