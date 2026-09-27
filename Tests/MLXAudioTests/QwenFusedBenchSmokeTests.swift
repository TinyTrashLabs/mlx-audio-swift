import XCTest
import MLX
@testable import MLXAudioTTS

/// The on-device bench must at least run on the Mac GPU (shapes, kernels).
final class QwenFusedBenchSmokeTests: XCTestCase {
    func testBenchRuns() {
        MLX.Device.setDefault(device: Device(.gpu))
        let r = Qwen3TTSFusedBench.run(ops: 4) { print("[kbench-mac]", $0) }
        XCTAssertGreaterThan(r.count, 10)
    }
}
