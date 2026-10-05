import Foundation
import Metal
import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN
import ObjectiveC
import XCTest

@testable import MLXAudioTTS

/// Counts compute dispatches (kernel launches) by swizzling the concrete
/// MTLComputeCommandEncoder class MLX encodes through. Test-only.
enum MetalDispatchCounter {
    nonisolated(unsafe) static var count = 0
    nonisolated(unsafe) private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
        var classes: [AnyClass] = []
        for dispatch in [MTLDispatchType.serial, .concurrent] {
            guard let buffer = queue.makeCommandBuffer(),
                  let encoder = buffer.makeComputeCommandEncoder(dispatchType: dispatch) else { continue }
            if let cls = object_getClass(encoder), !classes.contains(where: { $0 == cls }) { classes.append(cls) }
            encoder.endEncoding()
        }
        typealias Dispatch = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
        for cls in classes {
            for name in ["dispatchThreadgroups:threadsPerThreadgroup:", "dispatchThreads:threadsPerThreadgroup:"] {
                let sel = NSSelectorFromString(name)
                guard let method = class_getInstanceMethod(cls, sel) else { continue }
                let original = unsafeBitCast(method_getImplementation(method), to: Dispatch.self)
                let block: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = { obj, a, b in
                    count += 1
                    original(obj, sel, a, b)
                }
                method_setImplementation(method, imp_implementationWithBlock(block))
            }
        }
    }
}

/// The fused code-predictor frame (`Qwen3TTSFusedCodePredictor`,
/// 2026-10-05) against the per-layer fused path it replaces. Greedy
/// sub-codes ride on bf16 near-ties, so the exact mode is held to bit
/// identity: each kernel against the MLX ops it reproduces, whole frames on
/// a random 4-bit predictor and, with a checkpoint (`QWEN_PROFILE_WEIGHTS`,
/// `QWEN_PARITY_VOICE_DIRS`), every frame of real renders and the codes of
/// whole renders; plus launches and time per frame.
final class QwenFusedCodePredictorTests: XCTestCase {
    override func setUp() {
        super.setUp()
        MLX.Device.setDefault(device: Device(.gpu))
        MLXRandom.seed(5)
        Qwen3TTSModel.fastRope = true
        Qwen3TTSFusedStep.mode = .hybrid
    }

    override func tearDown() {
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.fusedCodePredictor = true
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.pipelineFrame = false
        Qwen3TTSModel.subCodeParityProbe = nil
        Qwen3TTSFusedCodePredictor.exact = true
        super.tearDown()
    }

    private func randomPredictor(dtype: DType = .bfloat16) throws -> Qwen3TTSCodePredictor {
        let config = try JSONDecoder().decode(Qwen3TTSTalkerCodePredictorConfig.self, from: "{}".data(using: .utf8)!)
        let cp = Qwen3TTSCodePredictor(config: config, talkerHiddenSize: config.hiddenSize)
        let params = cp.parameters().flattened().map { key, value -> (String, MLXArray) in
            if key.hasSuffix("norm.weight") { return (key, (1 + 0.2 * MLXRandom.normal(value.shape)).asType(dtype)) }
            if key.contains("codec_embedding") { return (key, MLXRandom.normal(value.shape).asType(dtype)) }
            return (key, value.asType(dtype))
        }
        cp.update(parameters: ModuleParameters.unflattened(params))
        quantize(model: cp, groupSize: 64, bits: 4, mode: .affine,
                 filter: { path, module in !path.contains("codec_embedding") && module is Linear })
        eval(cp.parameters())
        return cp
    }

    private func bits(_ a: MLXArray) -> [UInt32] {
        a.dtype == .float32 ? a.view(dtype: .uint32).asArray(UInt32.self)
            : a.asType(.bfloat16).view(dtype: .uint16).asArray(UInt16.self).map(UInt32.init)
    }

    // MARK: Kernels, bit for bit

    func testAddRmsMatchesMLXAddThenRmsNorm() throws {
        let k = 1024
        for rows in Array(repeating: 1, count: 50) + [2] {
            let a = (3 * MLXRandom.normal([1, rows, k])).asType(.bfloat16)
            let b = MLXRandom.normal([1, rows, k]).asType(.bfloat16)
            let w = (1 + 0.2 * MLXRandom.normal([k])).asType(.bfloat16)
            let params = MLXArray([Float(1e-6), 1_000_000, 0.088])
            let out = Qwen3TTSFusedCodePredictor.kernel(.addRms, dtype: .bfloat16, k: k, rows: rows)(
                [a, b, w, params], grid: (256, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[1, rows, k], [1, rows, k]], outputDTypes: [.bfloat16, .bfloat16])
            let sum = a + b
            XCTAssertEqual(bits(out[0]), bits(sum), "sum, rows \(rows)")
            XCTAssertEqual(bits(out[1]), bits(MLXFast.rmsNorm(sum, weight: w, eps: 1e-6)), "rmsnorm, rows \(rows)")
        }
    }

    func testArgmaxEmbedMatchesArgMaxGatherAddAndRmsNorm() throws {
        let (vocab, k) = (2048, 1024)
        let table = MLXRandom.normal([vocab, k]).asType(.bfloat16)
        let acc = MLXRandom.normal([1, 1, k]).asType(.bfloat16)
        let w = (1 + 0.2 * MLXRandom.normal([k])).asType(.bfloat16)
        let params = MLXArray([Float(1e-6), 1_000_000, 0.088])
        for trial in 0 ..< 8 {
            let logitsType: DType = trial % 2 == 0 ? .bfloat16 : .float32
            var logits = (4 * MLXRandom.normal([1, 1, vocab])).asType(logitsType)
            if trial >= 4 {
                // Ties at the max: MLX's argmax takes the lowest index.
                let top = logits.max().item(Float.self) + 1
                let idx = MLXArray([Int32(1900 - trial * 300), Int32(77 + trial), Int32(1500)])
                logits = putAlong(logits.reshaped(1, vocab), idx.reshaped(1, -1),
                                  values: MLXArray.full([1, 3], values: MLXArray(top), dtype: logitsType), axis: -1)
                    .reshaped(1, 1, vocab)
            }
            let out = Qwen3TTSFusedCodePredictor.kernel(.argmaxEmbedNorm, dtype: .bfloat16, k: k, vocab: vocab, cacheType: logitsType)(
                [logits, table, acc, w, params], grid: (256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[1, 1], [1, 1, k], [1, 1, k], [1, 1, k]], outputDTypes: [.uint32, .bfloat16, .bfloat16, .bfloat16])
            let token = argMax(logits[0..., (-1)..., 0...].squeezed(axis: 1), axis: -1, keepDims: true)
            let emb = table[token]
            XCTAssertEqual(out[0].item(UInt32.self), token.item(UInt32.self), "token, trial \(trial)")
            XCTAssertEqual(bits(out[1]), bits(emb.reshaped(1, 1, k)))
            XCTAssertEqual(bits(out[2]), bits((acc + emb).reshaped(1, 1, k)))
            XCTAssertEqual(bits(out[3]), bits(MLXFast.rmsNorm(emb, weight: w, eps: 1e-6).reshaped(1, 1, k)))
        }
    }

    /// The exact glue against qkv_post + KVCacheSimple (a float32 cache, as
    /// the float32 step 0 leaves it), steps 1…14 after a two-row prefill.
    func testExactGlueMatchesQkvPostAndCacheUpdate() throws {
        let cp = try randomPredictor()
        let layer = try XCTUnwrap(cp.model.layers[0].fusedLayer())
        let (h, d, cap) = (16, 8, 16)
        let width = (h + 2 * d) * 128
        let params = MLXArray([layer.eps, layer.ropeTheta, layer.scale])
        let cache = KVCacheSimple()
        _ = cache.update(keys: MLXRandom.normal([1, d, 2, 128]), values: MLXRandom.normal([1, d, 2, 128]))
        var kin = cache.innerState()[0], vin = cache.innerState()[1]
        for pos in 2 ..< 16 {
            let qkv = (2 * MLXRandom.normal([1, 1, width])).asType(.bfloat16)
            let split = Qwen3TTSFusedStep.kernel(.qkvPost, dtype: .bfloat16, paramType: .bfloat16, k: 1024, nt: 128, nq: h, nkv: d)(
                [qkv, layer.qNorm, layer.kNorm, MLXArray([layer.eps, layer.ropeTheta]), MLXArray([Int32(pos)])],
                grid: ((h + 2 * d) * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[1, h, 1, 128], [1, d, 1, 128], [1, d, 1, 128]], outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
            let (keys, values) = cache.update(keys: split[1], values: split[2])
            let glue = Qwen3TTSFusedCodePredictor.kernel(.exactGlue, dtype: .bfloat16, k: 1024, nq: h, nkv: d, cap: cap,
                                                         kinCap: kin.dim(2), cacheType: .float32)(
                [qkv, layer.qNorm, layer.kNorm, params, MLXArray([Int32(pos)]), kin, vin],
                grid: ((h + 2 * d) * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[1, h, 1, 128], [1, d, cap, 128], [1, d, cap, 128]], outputDTypes: [.float32, .float32, .float32])
            XCTAssertEqual(bits(glue[0]), bits(split[0].asType(.float32)), "q at \(pos)")
            XCTAssertEqual(bits(glue[1][0..., 0..., ..<(pos + 1), 0...]), bits(keys), "keys at \(pos)")
            XCTAssertEqual(bits(glue[2][0..., 0..., ..<(pos + 1), 0...]), bits(values), "values at \(pos)")
            let ref = MLXFast.scaledDotProductAttention(queries: split[0], keys: keys, values: values, scale: layer.scale, mask: nil)
            let mine = MLXFast.scaledDotProductAttention(
                queries: glue[0], keys: glue[1][0..., 0..., ..<(pos + 1), 0...], values: glue[2][0..., 0..., ..<(pos + 1), 0...],
                scale: layer.scale, mask: nil)
            XCTAssertEqual(bits(mine), bits(ref), "attention at \(pos)")
            (kin, vin) = (glue[1], glue[2])
        }
    }

    /// The fast mode's one-kernel attention against qkv_post + cache + MLX
    /// SDPA: same keys/values, attention within bf16 rounding (it cannot be
    /// bit-exact: MLX's own kernels are built fast-math, a runtime kernel is
    /// not).
    func testFastAttentionAgreesWithSDPA() throws {
        let cp = try randomPredictor()
        let layer = try XCTUnwrap(cp.model.layers[0].fusedLayer())
        let (h, d, cap) = (16, 8, 16)
        let width = (h + 2 * d) * 128
        let params = MLXArray([layer.eps, layer.ropeTheta, layer.scale])
        let cache = KVCacheSimple()
        var kin = MLXArray.zeros([1, d, cap, 128], dtype: .bfloat16)
        var vin = kin
        var pos = 0
        var mismatched = 0, total = 0
        for rows in [2] + Array(repeating: 1, count: 14) {
            let qkv = (2 * MLXRandom.normal([1, rows, width])).asType(.bfloat16)
            var qs: [MLXArray] = [], ks: [MLXArray] = [], vs: [MLXArray] = []
            for r in 0 ..< rows {
                let split = Qwen3TTSFusedStep.kernel(.qkvPost, dtype: .bfloat16, paramType: .bfloat16, k: 1024, nt: 128, nq: h, nkv: d)(
                    [qkv[0..., r ..< r + 1, 0...], layer.qNorm, layer.kNorm, MLXArray([layer.eps, layer.ropeTheta]),
                     MLXArray([Int32(pos + r)])],
                    grid: ((h + 2 * d) * 128, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[1, h, 1, 128], [1, d, 1, 128], [1, d, 1, 128]], outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
                qs.append(split[0]); ks.append(split[1]); vs.append(split[2])
            }
            let (keys, values) = cache.update(keys: concatenated(ks, axis: 2), values: concatenated(vs, axis: 2))
            let mask: MLXArray? = rows > 1 ? MultiHeadAttention.createAdditiveCausalMask(rows).asType(.bfloat16) : nil
            let ref = MLXFast.scaledDotProductAttention(queries: concatenated(qs, axis: 2), keys: keys, values: values,
                                                        scale: layer.scale, mask: mask)
                .transposed(0, 2, 1, 3).reshaped(1, rows, h * 128)
            let out = Qwen3TTSFusedCodePredictor.kernel(.attn, dtype: .bfloat16, k: 1024, nq: h, nkv: d, rows: rows, cap: cap)(
                [qkv, layer.qNorm, layer.kNorm, params, MLXArray([Int32(pos)]), kin, vin],
                grid: (h * 1024, rows, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, rows, h * 128], kin.shape, vin.shape], outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
            pos += rows
            XCTAssertEqual(bits(out[1][0..., 0..., ..<pos, 0...]), bits(keys), "keys at \(pos)")
            XCTAssertEqual(bits(out[2][0..., 0..., ..<pos, 0...]), bits(values), "values at \(pos)")
            let (x, y) = (bits(out[0]), bits(ref))
            mismatched += zip(x, y).filter { $0 != $1 }.count
            total += x.count
            XCTAssertLessThan(abs(out[0].asType(.float32) - ref.asType(.float32)).max().item(Float.self), 0.02)
            (kin, vin) = (out[1], out[2])
        }
        print("[fused-cp] fast attention vs MLX SDPA: \(mismatched) of \(total) bf16 outputs differ (by an ulp)")
    }

    // MARK: Whole frames on a random predictor

    /// The per-layer fused path as the loop ran it before (KV cache, module
    /// path for step 0's two rows, argmax), and its codec-embedding sum.
    private func referenceFrame(_ cp: Qwen3TTSCodePredictor, hidden: MLXArray, code0Embed: MLXArray) -> ([MLXArray], MLXArray) {
        let cache = cp.makeCache()
        var codes: [MLXArray] = []
        var sum = code0Embed
        for step in 0 ..< cp.numCodeGroups - 1 {
            let input = step == 0 ? concatenated([hidden, code0Embed], axis: 1) : cp.codecEmbedding[step - 1](codes.last!)
            let (logits, _, _) = cp(input, cache: cache, generationStep: step)
            let code = argMax(logits[0..., (-1)..., 0...].squeezed(axis: 1), axis: -1, keepDims: true)
            codes.append(code)
            sum = sum + cp.codecEmbedding[step](code)
        }
        return (codes, sum)
    }

    /// (frames identical, sub-codes identical, sub-codes total)
    private func frameAgreement(exact: Bool, hiddenType: DType, frames: Int) throws -> (Int, Int, Int) {
        MLXRandom.seed(5)
        Qwen3TTSFusedCodePredictor.exact = exact
        let cp = try randomPredictor()
        Qwen3TTSModel.fusedLayers = true
        let fused = try XCTUnwrap(cp.fusedFrame())
        var stepsSame = 0, stepsTotal = 0, framesSame = 0
        for _ in 0 ..< frames {
            let hidden = MLXRandom.normal([1, 1, 1024]).asType(hiddenType)
            let code0 = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
            let (refCodes, refSum) = referenceFrame(cp, hidden: hidden, code0Embed: code0)
            let frame = fused.frame(codeHidden: hidden, code0Embed: code0, stepZeroCache: cp.makeCache(),
                                    greedy: true, pipeline: false) { _ in fatalError() }
            let a = concatenated(refCodes, axis: 1).asArray(UInt32.self)
            let b = concatenated(frame.codes, axis: 1).asArray(UInt32.self)
            stepsTotal += a.count
            stepsSame += zip(a, b).filter { $0 == $1 }.count
            if a == b {
                framesSame += 1
                XCTAssertEqual(bits(frame.codecEmbedSum), bits(refSum), "codec embedding sum")
            }
        }
        return (framesSame, stepsSame, stepsTotal)
    }

    func testExactFrameMatchesPerLayerFusedPath() throws {
        // The talker hands the code predictor a float32 hidden state; bf16 covered too.
        for hiddenType in [DType.float32, .bfloat16] {
            let (frames, same, total) = try frameAgreement(exact: true, hiddenType: hiddenType, frames: 30)
            print("[fused-cp] exact, \(hiddenType) hidden: \(frames)/30 frames, \(same)/\(total) sub-codes identical")
            XCTAssertEqual(frames, 30, "\(hiddenType) hidden")
        }
    }

    func testFastFrameAgreementIsReported() throws {
        let (frames, same, total) = try frameAgreement(exact: false, hiddenType: .float32, frames: 30)
        print("[fused-cp] fast, float32 hidden: \(frames)/30 frames, \(same)/\(total) sub-codes identical")
        XCTAssertGreaterThan(same, 0)
    }

    /// The non-greedy path (sampler per sub-step) runs the same layers.
    func testFrameSamplesThroughTheGivenSampler() throws {
        let cp = try randomPredictor()
        Qwen3TTSModel.fusedLayers = true
        let fused = try XCTUnwrap(cp.fusedFrame())
        for exact in [true, false] {
            Qwen3TTSFusedCodePredictor.exact = exact
            let hidden = MLXRandom.normal([1, 1, 1024])
            let code0 = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
            var calls = 0
            let frame = fused.frame(codeHidden: hidden, code0Embed: code0, stepZeroCache: cp.makeCache(),
                                    greedy: false, pipeline: false) { logits in
                calls += 1
                XCTAssertEqual(logits.dim(-1), 2048)
                return argMax(logits[0..., (-1)..., 0...].squeezed(axis: 1), axis: -1, keepDims: true)
            }
            let greedy = fused.frame(codeHidden: hidden, code0Embed: code0, stepZeroCache: cp.makeCache(),
                                     greedy: true, pipeline: true) { _ in fatalError() }
            XCTAssertEqual(calls, 15)
            XCTAssertEqual(concatenated(frame.codes, axis: 1).asArray(UInt32.self), concatenated(greedy.codes, axis: 1).asArray(UInt32.self))
            XCTAssertEqual(bits(frame.codecEmbedSum), bits(greedy.codecEmbedSum))
        }
    }

    // MARK: Real checkpoint

    static let env = ProcessInfo.processInfo.environment
    static let weightsDir = env["QWEN_PROFILE_WEIGHTS"].map(URL.init(fileURLWithPath:))
    static let voiceDirs = (env["QWEN_PARITY_VOICE_DIRS"] ?? env["QWEN_PROFILE_VOICE_DIR"] ?? "")
        .split(separator: ":").map { URL(fileURLWithPath: String($0)) }
    static let parityText = env["QWEN_PARITY_TEXT"]
        ?? "The lighthouse keeper counted the ships every night, writing each name in a book that nobody read. "
        + "When the storm came in October, the lamp went dark for the first time in forty years, and he climbed "
        + "the stairs with a candle, one step at a time, until the beam swept the water again."
    static let seeds = (env["QWEN_PARITY_SEEDS"] ?? "1,2").split(separator: ",").compactMap { UInt64($0) }

    private func setUpRealRun() {
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        addTeardownBlock { Qwen3TTSModel.eosGreedyStop = false }
    }

    private func reference(_ voiceDir: URL, _ model: Qwen3TTSModel) throws -> (String, MLXArray) {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: voiceDir.appendingPathComponent("meta.json"))) as? [String: Any]
        let refText = try XCTUnwrap(meta?["refText"] as? String)
        let (_, refAudio) = try loadAudioArray(from: voiceDir.appendingPathComponent("ref.wav"), sampleRate: model.sampleRate)
        return (refText, refAudio)
    }

    /// Launches and wall time of one frame's code predictor — per-layer
    /// fused path, exact fused frame, fast fused frame — on real weights.
    func testLaunchesAndTimePerFrame() async throws {
        guard let weightsDir = Self.weightsDir else { throw XCTSkip("set QWEN_PROFILE_WEIGHTS") }
        MetalDispatchCounter.install()
        setUpRealRun()
        let model = try await Qwen3TTSModel.fromModelDirectory(weightsDir)
        let talker = model.talker
        let hidden = 2 * MLXRandom.normal([1, 1, 1024])                 // float32, as the talker's
        let token = MLXArray([Int32(1234)]).reshaped(1, 1)
        let codeCache = talker.codePredictor.makeCache()
        func frame() -> [MLXArray] {
            let r = model.predictSubCodes(hidden: hidden, nextToken: token, codeCache: codeCache,
                                          temperature: 0.9, topP: 1, topK: 50, minP: 0)
            var codecEmbed = r.codecEmbedSum ?? talker.getInputEmbeddings()(token)
            if r.codecEmbedSum == nil {
                for (i, code) in r.codes.enumerated() { codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code) }
            }
            return r.codes + [codecEmbed]
        }
        let arms: [(String, Bool, Bool)] = [("per-layer fused", false, true), ("fused frame, exact", true, true),
                                             ("fused frame, fast", true, false)]
        var codes: [String: [UInt32]] = [:]
        for _ in 0 ..< 2 {
            for (label, fusedCP, exact) in arms {
                Qwen3TTSModel.fusedCodePredictor = fusedCP
                Qwen3TTSFusedCodePredictor.exact = exact
                for _ in 0 ..< 3 { eval(frame()) }
                MetalDispatchCounter.count = 0
                let out = frame(); eval(out)
                let launches = MetalDispatchCounter.count
                let n = 60
                let t0 = Date()
                for _ in 0 ..< n { eval(frame()) }
                let ms = Date().timeIntervalSince(t0) * 1000 / Double(n)
                Qwen3TTSModel.pipelineFrame = true
                let t1 = Date()
                for _ in 0 ..< n { eval(frame()) }
                let pipelined = Date().timeIntervalSince(t1) * 1000 / Double(n)
                Qwen3TTSModel.pipelineFrame = false
                codes[label] = concatenated(Array(out.dropLast()), axis: 1).asArray(UInt32.self)
                print(String(format: "[fused-cp] %@: %d launches/frame, %.2f ms/frame (pipelined %.2f ms)",
                             label, launches, ms, pipelined))
            }
        }
        XCTAssertEqual(codes["fused frame, exact"], codes["per-layer fused"])
    }

    /// Every frame of real renders through both paths on identical inputs
    /// (`subCodeParityProbe`; no divergence cascade): sub-code agreement.
    func testSubCodesAgreePerFrameOnRealRenders() async throws {
        guard let weightsDir = Self.weightsDir, !Self.voiceDirs.isEmpty else {
            throw XCTSkip("set QWEN_PROFILE_WEIGHTS and QWEN_PARITY_VOICE_DIRS")
        }
        setUpRealRun()
        Qwen3TTSFusedCodePredictor.exact = Self.env["QWEN_PARITY_FAST"] != "1"
        let model = try await Qwen3TTSModel.fromModelDirectory(weightsDir)
        var frames = 0, framesSame = 0, codes = 0, codesSame = 0
        Qwen3TTSModel.subCodeParityProbe = { ref, fused in
            let a = concatenated(ref, axis: 1).asArray(UInt32.self)
            let b = concatenated(fused, axis: 1).asArray(UInt32.self)
            frames += 1; codes += a.count
            codesSame += zip(a, b).filter { $0 == $1 }.count
            if a == b { framesSame += 1 }
        }
        for voiceDir in Self.voiceDirs {
            let (refText, refAudio) = try reference(voiceDir, model)
            for seed in Self.seeds {
                MLXRandom.seed(seed)
                let stream = model.generateStream(
                    text: Self.parityText, voice: nil, refAudio: refAudio, refText: refText, language: nil,
                    generationParameters: model.defaultGenerationParameters, streamingInterval: 1.0)
                for try await _ in stream {}
            }
        }
        Qwen3TTSModel.subCodeParityProbe = nil
        print("[fused-cp] per-frame parity (\(Qwen3TTSFusedCodePredictor.exact ? "exact" : "fast")): "
              + "\(framesSame)/\(frames) frames, \(codesSame)/\(codes) sub-codes identical")
        if Qwen3TTSFusedCodePredictor.exact { XCTAssertEqual(framesSame, frames) }
    }

    /// Whole renders, same seed, fused code predictor off and on (exact):
    /// the codes must be identical frame for frame. Voices from
    /// `QWEN_PARITY_VOICE_DIRS` (colon-separated voice-store folders), seeds
    /// from `QWEN_PARITY_SEEDS`.
    func testRenderCodesMatchWithAndWithoutFusedFrame() async throws {
        guard let weightsDir = Self.weightsDir, !Self.voiceDirs.isEmpty else {
            throw XCTSkip("set QWEN_PROFILE_WEIGHTS and QWEN_PARITY_VOICE_DIRS")
        }
        setUpRealRun()
        Qwen3TTSModel.pipelineFrame = Self.env["QWEN_PARITY_PIPELINE"] != "0"
        let model = try await Qwen3TTSModel.fromModelDirectory(weightsDir)
        var totalFrames = 0
        var rates: [Bool: [Double]] = [:]
        for voiceDir in Self.voiceDirs {
            let (refText, refAudio) = try reference(voiceDir, model)
            for seed in Self.seeds {
                var takes: [[Int32]] = []
                for fusedCP in [false, true] {
                    Qwen3TTSModel.fusedCodePredictor = fusedCP
                    MLXRandom.seed(seed)
                    let started = Date()
                    let stream = model.generateStream(
                        text: Self.parityText, voice: nil, refAudio: refAudio, refText: refText, language: nil,
                        generationParameters: model.defaultGenerationParameters, streamingInterval: 1.0)
                    for try await _ in stream {}
                    rates[fusedCP, default: []].append(Double(Qwen3TTSModel.lastFrameCount) / Date().timeIntervalSince(started))
                    takes.append(try XCTUnwrap(Qwen3TTSModel.lastGeneratedCodes).asArray(Int32.self))
                }
                let (a, b) = (takes[0], takes[1])
                print("[fused-cp] \(voiceDir.lastPathComponent) seed \(seed): \(a.count / 16) vs \(b.count / 16) frames, "
                      + (a == b ? "codes IDENTICAL" : "codes DIFFER"))
                XCTAssertEqual(a, b, "\(voiceDir.lastPathComponent) seed \(seed)")
                totalFrames += a.count / 16
            }
        }
        func median(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
        print(String(format: "[fused-cp] render parity over %d frames; median render f/s per-layer %.1f, fused %.1f",
                     totalFrames, median(rates[false] ?? [0]), median(rates[true] ?? [0])))
    }
}
