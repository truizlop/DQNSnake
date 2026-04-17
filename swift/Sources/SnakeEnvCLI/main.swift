import Foundation
import SnakeEnv

@main
struct SnakeEnvCLI {
    static func main() async {
        let steps = Int(ProcessInfo.processInfo.environment["SNAKE_STEPS"] ?? "30") ?? 30
        let pythonDir = ProcessInfo.processInfo.environment["SNAKE_PYTHON_DIR"]
        let visual = (ProcessInfo.processInfo.environment["SNAKE_VISUAL"] ?? "0") == "1"
        let fps = Double(ProcessInfo.processInfo.environment["SNAKE_FPS"] ?? "12") ?? 12
        let frameDelay = fps > 0 ? (1.0 / fps) : 0

        let env = SnakeEnv(usePythonBridge: true, pythonModulePath: pythonDir)

        do {
            let frame = try await env.reset()
            print("reset: \(frame.width)x\(frame.height), pixels=\(frame.data.count)")
            if visual {
                try await env.render()
            }

            for i in 0..<steps {
                let action = Int.random(in: 0...3)
                let out = try await env.step(action: action)
                if visual {
                    try await env.render()
                }
                print("step=\(i) action=\(action) reward=\(out.reward) score=\(out.score) done=\(out.done)")
                if out.done {
                    _ = try await env.reset()
                    if visual {
                        try await env.render()
                    }
                    print("reset after terminal state")
                }
                if frameDelay > 0 {
                    let nanoseconds = UInt64(frameDelay * 1_000_000_000)
                    try await Task.sleep(nanoseconds: nanoseconds)
                }
            }

            print("final score: \(await env.score())")
        } catch {
            fputs("SnakeEnvCLI error: \(error)\n", stderr)
            exit(1)
        }
    }
}
