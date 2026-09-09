import MLX
import MLXFast
import XCTest

@testable import MLXAudioTTS

/// The iOS speed work (2026-09-09) swaps Qwen3-TTS's hand-rolled rotary
/// (slice / negate / concat / multiply / add, per q and per k, per layer)
/// for the single-kernel `MLXFast.RoPE`. These pin that the two agree for
/// both networks, including the talker's interleaved M-RoPE, which collapses
/// to plain rotate-half RoPE because generation feeds identical position
/// streams to all three axes.
final class QwenRopeParityTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Deterministic and free of the Metal library lookup under `swift test`.
        MLX.Device.setDefault(device: Device(.cpu))
        MLXRandom.seed(7)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// (B, H, L, D) tensor, the layout both attentions rotate in.
    private func heads(_ l: Int, d: Int = 64) -> MLXArray {
        MLXRandom.normal([1, 4, l, d])
    }

    func testCodePredictorRotaryMatchesTheFusedKernel() {
        let theta: Float = 1_000_000
        let d = 64
        for (offset, l) in [(0, 1), (0, 5), (3, 1), (9, 8)] {
            let q = heads(l, d: d)
            let rotary = Qwen3TTSRotaryEmbedding(dim: d, base: theta)
            let pos = MLXArray(Int32(offset) ..< Int32(offset + l)).reshaped(1, l)
            let (cosV, sinV) = rotary(q, positionIds: pos)
            let cosE = expandedDimensions(cosV, axis: 1)
            let sinE = expandedDimensions(sinV, axis: 1)
            let chain = q * cosE + cpRotateHalf(q) * sinE

            let fused = MLXFast.RoPE(q, dimensions: d, traditional: false, base: theta, scale: 1, offset: offset)

            XCTAssertLessThan(maxAbsDiff(chain, fused), 1e-4, "offset \(offset) len \(l)")
        }
    }

    func testTalkerMRopeWithEqualStreamsMatchesTheFusedKernel() {
        let theta: Float = 1_000_000
        let d = 64
        for (offset, l) in [(0, 1), (0, 6), (41, 1), (200, 4)] {
            let q = heads(l, d: d)
            let rotary = TalkerRotaryEmbedding(dim: d, base: theta, mropeSection: [24, 20, 20])
            // What the talker builds when no explicit positions are given:
            // the same 1-D positions on all three M-RoPE axes.
            let pos = MLXArray(Int32(offset) ..< Int32(offset + l)).reshaped(1, l)
            let posIds = stacked([pos, pos, pos], axis: 0)
            let (cosV, sinV) = rotary(q, positionIds: posIds)
            let (chainQ, _) = applyRotaryPosEmbForTest(q, q, cos: cosV, sin: sinV)

            let fused = MLXFast.RoPE(q, dimensions: d, traditional: false, base: theta, scale: 1, offset: offset)

            XCTAssertLessThan(maxAbsDiff(chainQ, fused), 1e-4, "offset \(offset) len \(l)")
        }
    }
}
