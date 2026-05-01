import Foundation
import Testing
@testable import SnakeEnv

private enum BridgeAction: Int {
    case left = 0
    case up = 1
    case right = 2
    case down = 3
}

private func resolvedPythonModulePath() -> String {
    let cwd = FileManager.default.currentDirectoryPath
    let direct = (cwd as NSString).appendingPathComponent("python")
    if FileManager.default.fileExists(atPath: direct) {
        return direct
    }
    let parent = (cwd as NSString).appendingPathComponent("../python")
    return parent
}

private func pythonBridgeAvailable() -> Bool {
    PythonSnakeBridge.make(pythonModulePath: resolvedPythonModulePath(), seed: 123) != nil
}

@Test(.enabled(if: pythonBridgeAvailable()))
func snakeEnvPythonBridgeReturnsChangingFramesAcrossSteps() async throws {
    let env = SnakeEnv(usePythonBridge: true, pythonModulePath: resolvedPythonModulePath(), seed: 123)
    let resetFrame = try await env.reset()

    let step1 = try await env.step(action: BridgeAction.left.rawValue)
    let step2 = try await env.step(action: BridgeAction.up.rawValue)
    let step3 = try await env.step(action: BridgeAction.right.rawValue)

    // At least one non-terminal step should produce a different observation than reset.
    let anyChanged =
        step1.observation != resetFrame.data || step2.observation != resetFrame.data
        || step3.observation != resetFrame.data
    #expect(anyChanged)

    var stack = FrameStack(capacity: 4)
    stack.reset(frame: resetFrame.data)
    stack.append(step1.observation)
    stack.append(step2.observation)
    stack.append(step3.observation)

    let stacked = stack.stacked()
    #expect(stacked.count == resetFrame.data.count * 4)

    // Final channel should match the most recent frame exactly.
    let frameSize = resetFrame.data.count
    let lastFrame = Array(stacked[(stacked.count - frameSize)..<stacked.count])
    #expect(lastFrame == step3.observation)
}

@Test(.enabled(if: pythonBridgeAvailable()))
func snakeEnvPythonBridgeAppliesActionDeterministicallyWithSeed() async throws {
    let pythonPath = resolvedPythonModulePath()
    let envA = SnakeEnv(usePythonBridge: true, pythonModulePath: pythonPath, seed: 999)
    let envB = SnakeEnv(usePythonBridge: true, pythonModulePath: pythonPath, seed: 999)

    let frameA = try await envA.reset()
    let frameB = try await envB.reset()
    #expect(frameA.data == frameB.data)

    let actions: [BridgeAction] = [.left, .up, .right, .down, .left]
    for action in actions {
        let outA = try await envA.step(action: action.rawValue)
        let outB = try await envB.step(action: action.rawValue)
        #expect(outA.observation == outB.observation)
        #expect(outA.reward == outB.reward)
        #expect(outA.done == outB.done)
        #expect(outA.score == outB.score)
    }
}
