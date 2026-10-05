import MLX
import XCTest

@testable import MLXAudioTTS

/// `Qwen3TTSModel.fastSample0` replaces the first-codebook sampler's
/// per-frame host work (a ~1,023-id suppression array and the repetition
/// history re-uploaded and scattered every frame) with per-generation GPU
/// masks. These pin that it is bit-identical to the upstream `sampleToken`:
/// same processed logits, and the same token sequence for the same seed
/// across a multi-frame loop (so it also consumes the random stream the
/// same way). No weights needed.
final class QwenSample0ParityTests: XCTestCase {
    // The talker's codec vocab and EOS (Qwen3-TTS 12Hz configs).
    private let vocab = 3072
    private let eos = 2150

    private var suppress: [Int] { (vocab - 1024 ..< vocab).filter { $0 != eos } }

    private func randomLogits(_ dtype: DType, scale: Float = 4) -> MLXArray {
        let l = (MLXRandom.normal([1, 1, vocab]) * scale).asType(dtype)
        eval(l)
        return l
    }

    private func randomHistory(_ n: Int) -> [Int] {
        // Duplicates on purpose; drawn from the unsuppressed range like real codes.
        (0 ..< n).map { _ in Int.random(in: 0 ..< (vocab - 1024)) } + (n > 0 ? [5, 5, 17] : [])
    }

    /// The upstream pre-sampling stage, copied verbatim from `sampleToken`
    /// as of 233c0d7 (bias, scatter suppression, gather/scatter penalty).
    private func referenceLogits(
        _ logits: MLXArray, history: [Int], repetitionPenalty: Float, eosLogitBias: Float
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1)
        if eosLogitBias != 0 {
            let eosIdx = MLXArray([Int32(eos)]).reshaped(1, 1)
            let biased = takeAlong(logitsSlice, eosIdx, axis: -1) + MLXArray(eosLogitBias).asType(logitsSlice.dtype)
            logitsSlice = putAlong(logitsSlice, eosIdx, values: biased, axis: -1)
        }
        let suppressArr = MLXArray(suppress.map { Int32($0) }).reshaped(1, -1)
        let negInf = MLXArray.full([1, suppress.count], values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
        logitsSlice = putAlong(logitsSlice, suppressArr, values: negInf, axis: -1)
        if !history.isEmpty, repetitionPenalty != 1.0 {
            let unique = Array(Set(history)).filter { $0 < logitsSlice.dim(-1) }
            let tokenIds = MLXArray(unique.map { Int32($0) }).reshaped(1, -1)
            let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
            let penalized = which(selected .< 0, selected * repetitionPenalty, selected / repetitionPenalty)
            logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
        }
        return logitsSlice
    }

    private func bits(_ a: MLXArray) -> [UInt32] {
        a.asType(.float32).asArray(Float.self).map { $0.bitPattern }
    }

    private func state(history: [Int]) -> Qwen3TTSModel.Sample0State {
        let s = Qwen3TTSModel.Sample0State(vocabSize: vocab, suppressTokens: suppress, eosTokenId: eos)
        for t in history { s.note(t) }
        return s
    }

    func testProcessedLogitsAreBitIdentical() {
        MLXRandom.seed(11)
        for dtype in [DType.float32, .bfloat16, .float16] {
            for historyLen in [0, 1, 40, 300] {
                for penalty: Float in [1.0, 1.05, 1.3] {
                    for bias: Float in [0, 2.5, -40] {
                        let logits = randomLogits(dtype)
                        let history = randomHistory(historyLen)
                        let ref = referenceLogits(logits, history: history, repetitionPenalty: penalty, eosLogitBias: bias)
                        let fast = Qwen3TTSModel.sample0Logits(
                            logits, state: state(history: history), repetitionPenalty: penalty, eosLogitBias: bias)
                        XCTAssertEqual(fast.dtype, ref.dtype)
                        XCTAssertEqual(fast.shape, ref.shape)
                        XCTAssertEqual(bits(fast), bits(ref),
                                       "dtype \(dtype) history \(historyLen) penalty \(penalty) bias \(bias)")
                    }
                }
            }
        }
    }

    func testSampledTokenMatchesUpstreamAcrossSettings() {
        struct Case { let temp: Float; let topK: Int; let topP: Float; let minP: Float }
        let cases = [
            Case(temp: 0, topK: 0, topP: 1, minP: 0),
            Case(temp: 0.9, topK: 0, topP: 1, minP: 0),       // phone default
            Case(temp: 1.5, topK: 0, topP: 1, minP: 0),
            Case(temp: 0.9, topK: 50, topP: 1, minP: 0),
            Case(temp: 0.9, topK: 0, topP: 0.9, minP: 0),
            Case(temp: 0.7, topK: 40, topP: 0.9, minP: 0.05),
        ]
        MLXRandom.seed(3)
        for dtype in [DType.float32, .bfloat16] {
            for c in cases {
                for trial in 0 ..< 12 {
                    let logits = randomLogits(dtype, scale: trial % 2 == 0 ? 1 : 4)
                    let history = randomHistory(trial * 13)
                    let bias: Float = trial % 3 == 0 ? 1.5 : 0
                    let seed = UInt64(1000 + trial)

                    MLXRandom.seed(seed)
                    let old = Qwen3TTSModel.sampleToken(
                        logits, temperature: c.temp, topP: c.topP, topK: c.topK, repetitionPenalty: 1.05,
                        generatedTokens: history, suppressTokens: suppress, eosTokenId: eos, minP: c.minP,
                        eosLogitBias: bias)
                    let oldId = old.asType(.int32).item(Int32.self)

                    MLXRandom.seed(seed)
                    let new = Qwen3TTSModel.sampleFirstCode(
                        logits, state: state(history: history), temperature: c.temp, topP: c.topP,
                        topK: c.topK, repetitionPenalty: 1.05, minP: c.minP, eosLogitBias: bias)
                    let newId = new.asType(.int32).item(Int32.self)

                    XCTAssertEqual(new.shape, old.shape)
                    XCTAssertEqual(newId, oldId, "dtype \(dtype) temp \(c.temp) topK \(c.topK) topP \(c.topP) minP \(c.minP) trial \(trial)")
                }
            }
        }
    }

    /// A generation-shaped loop: each path keeps its own history from its own
    /// draws (old: Swift array; new: presence mask), one seed for the run.
    func testMultiFrameSequenceIsIdentical() {
        MLXRandom.seed(5)
        let frames = (0 ..< 60).map { _ in randomLogits(.bfloat16, scale: 2) }
        for (temp, topK) in [(Float(0.9), 0), (Float(0.9), 50), (Float(0), 0)] {
            MLXRandom.seed(42)
            var oldHistory = [Int]()
            var oldSeq = [Int32]()
            for l in frames {
                let t = Qwen3TTSModel.sampleToken(
                    l, temperature: temp, topP: 1, topK: topK, repetitionPenalty: 1.05,
                    generatedTokens: oldHistory, suppressTokens: suppress, eosTokenId: eos, minP: 0)
                let id = t.asType(.int32).item(Int32.self)
                oldSeq.append(id)
                oldHistory.append(Int(id))
            }

            MLXRandom.seed(42)
            let st = Qwen3TTSModel.Sample0State(vocabSize: vocab, suppressTokens: suppress, eosTokenId: eos)
            var newSeq = [Int32]()
            for l in frames {
                let t = Qwen3TTSModel.sampleFirstCode(
                    l, state: st, temperature: temp, topP: 1, topK: topK, repetitionPenalty: 1.05,
                    minP: 0, eosLogitBias: 0)
                let id = t.asType(.int32).item(Int32.self)
                newSeq.append(id)
                st.note(Int(id))
            }
            XCTAssertEqual(newSeq, oldSeq, "temp \(temp) topK \(topK)")
            XCTAssertFalse(oldSeq.contains { Int($0) >= vocab - 1024 && Int($0) != eos }, "suppressed id drawn")
        }
    }

    /// Mac micro-benchmark (informational): one sample0 per "frame" with the
    /// phone's settings and a 300-token history, forced eval each frame like
    /// `profileLoop` does.
    func testSample0MicroBenchmark() {
        MLXRandom.seed(9)
        let frames = (0 ..< 50).map { _ in randomLogits(.bfloat16, scale: 2) }
        let history = randomHistory(300)
        func run(_ body: (MLXArray) -> MLXArray) -> Double {
            for l in frames.prefix(5) { eval(body(l)) }
            let start = Date()
            for l in frames { eval(body(l)) }
            return Date().timeIntervalSince(start) / Double(frames.count) * 1000
        }
        let oldMs = run { l in
            Qwen3TTSModel.sampleToken(
                l, temperature: 0.9, topP: 1, topK: 0, repetitionPenalty: 1.05,
                generatedTokens: history, suppressTokens: suppress, eosTokenId: eos, minP: 0)
        }
        let st = state(history: history)
        let newMs = run { l in
            Qwen3TTSModel.sampleFirstCode(
                l, state: st, temperature: 0.9, topP: 1, topK: 0, repetitionPenalty: 1.05, minP: 0, eosLogitBias: 0)
        }
        print(String(format: "SAMPLE0_BENCH old=%.3f ms new=%.3f ms per frame", oldMs, newMs))
    }
}
