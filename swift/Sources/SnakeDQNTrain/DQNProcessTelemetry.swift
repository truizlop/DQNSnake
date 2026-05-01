import Darwin
import Foundation

struct DQNProcessTelemetry {
    let rssBytes: UInt64
    let virtualBytes: UInt64
    let userCPUSeconds: Double
    let systemCPUSeconds: Double

    static func capture() -> DQNProcessTelemetry? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else {
            return nil
        }

        var times = task_thread_times_info()
        var timeCount = mach_msg_type_number_t(MemoryLayout<task_thread_times_info>.size) / 4
        let kerrTime = withUnsafeMutablePointer(to: &times) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(timeCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &timeCount)
            }
        }
        guard kerrTime == KERN_SUCCESS else {
            return nil
        }

        let user = Double(times.user_time.seconds) + Double(times.user_time.microseconds) / 1_000_000
        let system = Double(times.system_time.seconds) + Double(times.system_time.microseconds) / 1_000_000

        return DQNProcessTelemetry(
            rssBytes: UInt64(info.resident_size),
            virtualBytes: UInt64(info.virtual_size),
            userCPUSeconds: user,
            systemCPUSeconds: system
        )
    }
}
