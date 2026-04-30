import Foundation

public enum SnakeEnvError: Error {
    case bridgeUnavailable
    case invalidFrameShape
    case invalidAction
    case pythonBridgeInitializationFailed(String)
    case pythonConversionFailed(String)
}
