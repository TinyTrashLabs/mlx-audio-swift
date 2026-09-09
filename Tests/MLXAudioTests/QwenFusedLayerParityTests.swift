import MLX
import MLXNN
import MLXLMCommon
import XCTest

@testable import MLXAudioTTS

/// The fused single-token decoder step (`Qwen3TTSFusedStep`, 2026-09-09)
/// replaces ~22 MLX launches per layer with four custom Metal kernels plus
/// the cache update and SDPA. These pin it against the module path for both
/// networks, at a nonzero cache offset, in every dtype the phone runs.
/// Custom kernels are Metal-only, so this suite needs the GPU (and under
/// `swift test`, the metallib symlinked into the xctest bundle).
final class QwenFusedLayerParityTests: XCTestCase {
    override func setUp() {
        super.setUp()
        MLX.Device.setDefault(device: Device(.gpu))
        MLXRandom.seed(11)
        Qwen3TTSModel.fastRope = true
        Qwen3TTSFusedStep.mode = .hybrid
    }

    override func tearDown() {
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.fastRope = false
        super.tearDown()
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Random norm weights (the modules default to ones, which would hide a
    /// missing multiply) and 4-bit affine quantized projections, as loaded
    /// from the mobile checkpoint.
    private func randomize(_ layer: Module, dtype: DType) {
        let params = layer.parameters().flattened().map { key, value -> (String, MLXArray) in
            if key.hasSuffix("norm.weight") {
                return (key, (1 + 0.2 * MLXRandom.normal(value.shape)).asType(dtype))
            }
            return (key, value.asType(dtype))
        }
        layer.update(parameters: ModuleParameters.unflattened(params))
        quantize(model: layer, groupSize: 64, bits: 4)
        // quantize() leaves scales/biases in the weight's dtype already.
        eval(layer.parameters())
    }

    private func talkerLayer(dtype: DType) throws -> TalkerDecoderLayer {
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: "{}".data(using: .utf8)!)
        let layer = TalkerDecoderLayer(config: config, layerIdx: 0)
        randomize(layer, dtype: dtype)
        return layer
    }

    private func codePredictorLayer(dtype: DType) throws -> CodePredictorDecoderLayer {
        let config = try JSONDecoder().decode(Qwen3TTSTalkerCodePredictorConfig.self, from: "{}".data(using: .utf8)!)
        let layer = CodePredictorDecoderLayer(config: config, layerIdx: 0)
        randomize(layer, dtype: dtype)
        return layer
    }

    /// Runs `prefix` tokens through the module path (both modes prefill that
    /// way), then one token through the path under test.
    private func step(_ run: (MLXArray, KVCacheSimple) -> MLXArray, prefix: MLXArray, token: MLXArray, fused: Bool) -> MLXArray {
        let cache = KVCacheSimple()
        Qwen3TTSModel.fusedLayers = false
        _ = run(prefix, cache)
        Qwen3TTSModel.fusedLayers = fused
        let out = run(token, cache)
        eval(out)
        return out
    }

    private func assertParity(dtype: DType, tolerance: Float, file: StaticString = #filePath, line: UInt = #line) throws {
        let hidden = 1024
        let prefix = MLXRandom.normal([1, 5, hidden]).asType(dtype)
        let token = MLXRandom.normal([1, 1, hidden]).asType(dtype)
        let unused = (token, token)

        let talker = try talkerLayer(dtype: dtype)
        // A layer the kernels cannot take falls back to the module path,
        // which would make the comparison below vacuous.
        XCTAssertNotNil(talker.fusedLayer(), "talker layer must be fusable", file: file, line: line)
        let tRef = step({ x, c in talker(x, positionEmbeddings: unused, mask: nil, cache: c) }, prefix: prefix, token: token, fused: false)
        let tFused = step({ x, c in talker(x, positionEmbeddings: unused, mask: nil, cache: c) }, prefix: prefix, token: token, fused: true)
        XCTAssertEqual(tFused.shape, [1, 1, hidden], file: file, line: line)
        let tDiff = maxAbsDiff(tRef, tFused)
        XCTAssertLessThan(tDiff, tolerance, "talker layer \(dtype): max abs diff \(tDiff)", file: file, line: line)

        let cp = try codePredictorLayer(dtype: dtype)
        XCTAssertNotNil(cp.fusedLayer(), "code predictor layer must be fusable", file: file, line: line)
        let cRef = step({ x, c in cp(x, positionEmbeddings: unused, mask: nil, cache: c) }, prefix: prefix, token: token, fused: false)
        let cFused = step({ x, c in cp(x, positionEmbeddings: unused, mask: nil, cache: c) }, prefix: prefix, token: token, fused: true)
        let cDiff = maxAbsDiff(cRef, cFused)
        XCTAssertLessThan(cDiff, tolerance, "code predictor layer \(dtype): max abs diff \(cDiff)", file: file, line: line)
        print("[fused-parity] \(dtype): talker \(tDiff), code predictor \(cDiff), output scale \(abs(tRef.asType(.float32)).max().item(Float.self))")
    }

    func testFusedStepRunsTheKernelsDirectly() throws {
        // Bypasses the flag: the kernels themselves against the module path.
        let talker = try talkerLayer(dtype: .float32)
        let fused = try XCTUnwrap(talker.fusedLayer())
        let x = MLXRandom.normal([1, 1, 1024])
        Qwen3TTSModel.fusedLayers = false
        let ref = talker(x, positionEmbeddings: (x, x), mask: nil, cache: KVCacheSimple())
        let out = Qwen3TTSFusedStep.run(x, layer: fused, cache: KVCacheSimple())
        eval(ref, out)
        let diff = maxAbsDiff(ref, out)
        XCTAssertGreaterThan(diff, 0, "float summation order differs between the paths; an exact match means the kernels did not run")
        XCTAssertLessThan(diff, 2e-3)
    }

    func testFusedStepMatchesModulePathInFloat32() throws {
        try assertParity(dtype: .float32, tolerance: 2e-3)
    }

    func testCustomMatvecModeMatchesModulePath() throws {
        Qwen3TTSFusedStep.mode = .customMatvec
        defer { Qwen3TTSFusedStep.mode = .hybrid }
        try assertParity(dtype: .float32, tolerance: 2e-3)
        try assertParity(dtype: .bfloat16, tolerance: 0.25)
    }

    func testFusedStepMatchesModulePathInBFloat16() throws {
        // The module path rounds to bf16 after every op; the fused path keeps
        // float32 inside a kernel, so agreement is at bf16 resolution.
        try assertParity(dtype: .bfloat16, tolerance: 0.25)
    }

    func testFusedStepMatchesModulePathInFloat16() throws {
        try assertParity(dtype: .float16, tolerance: 0.1)
    }

    func testMatvecAcceptsAWiderInputThanItsResidual() throws {
        // On the phone MLX's SDPA returns float32 for bf16 q/k/v, so the
        // o_proj matvec sees a float32 vector and a bf16 residual. The
        // kernel must build and agree with the module path either way.
        let talker = try talkerLayer(dtype: .bfloat16)
        let fused = try XCTUnwrap(talker.fusedLayer())
        let attended = MLXRandom.normal([1, 16, 1, 128])              // float32
        let residual = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
        let out = Qwen3TTSFusedStep.kernel(.matvec, dtype: .bfloat16, paramType: .bfloat16, k: 2048,
                                           nt: Qwen3TTSFusedStep.matvecThreads, xType: .float32)(
            [attended, fused.o.weight, fused.o.scales, fused.o.biases, residual],
            grid: (1024 / Qwen3TTSFusedStep.matvecRowsPerGroup * Qwen3TTSFusedStep.matvecThreads, 1, 1),
            threadGroup: (Qwen3TTSFusedStep.matvecThreads, 1, 1),
            outputShapes: [[1, 1, 1024]], outputDTypes: [.bfloat16])[0]
        let ref = residual + talker.selfAttn.oProj(attended.reshaped(1, 1, 2048).asType(.bfloat16))
        eval(out, ref)
        XCTAssertEqual(out.dtype, .bfloat16)
        XCTAssertLessThan(maxAbsDiff(ref, out), 0.25)
    }

    func testFusedStepIsSkippedForPrefill() throws {
        // A multi-token input never takes the fused path (it is a single-
        // token kernel); the flag must not change prefill results.
        let talker = try talkerLayer(dtype: .float32)
        let prefix = MLXRandom.normal([1, 5, 1024])
        let unused = (prefix, prefix)
        Qwen3TTSModel.fusedLayers = false
        let a = talker(prefix, positionEmbeddings: unused, mask: nil, cache: KVCacheSimple())
        Qwen3TTSModel.fusedLayers = true
        let b = talker(prefix, positionEmbeddings: unused, mask: nil, cache: KVCacheSimple())
        XCTAssertLessThan(maxAbsDiff(a, b), 1e-6)
    }
}
