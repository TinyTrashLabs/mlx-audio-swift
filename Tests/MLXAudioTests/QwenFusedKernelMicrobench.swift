import MLX
import MLXNN
import MLXLMCommon
import XCTest

@testable import MLXAudioTTS

/// Where does a fused layer step spend its time on this Mac's GPU? Prints
/// only. Every `eval` costs ~200 µs of sync here, so the chained rows (40
/// dependent ops per eval) are the ones that show per-op cost.
///
/// History: the first cut passed dtype/width as `template:` args, and MLX's
/// builder cost ~43 µs per call for that (a std::regex built per call);
/// the kernels are now specialised by source text and cached instead.
final class QwenFusedKernelMicrobench: XCTestCase {
    private func time(_ name: String, iterations: Int = 300, _ body: () -> MLXArray) {
        for _ in 0 ..< 10 { eval(body()) }
        let t0 = Date()
        for _ in 0 ..< iterations { eval(body()) }
        print(String(format: "[fused-micro] %@: %.1f µs", name, Date().timeIntervalSince(t0) / Double(iterations) * 1e6))
    }

    func testPrintKernelTimes() throws {
        MLX.Device.setDefault(device: Device(.gpu))
        Qwen3TTSModel.fastRope = true
        defer { Qwen3TTSModel.fusedLayers = false; Qwen3TTSModel.fastRope = false }
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: "{}".data(using: .utf8)!)
        let layer = TalkerDecoderLayer(config: config, layerIdx: 0)
        layer.update(parameters: ModuleParameters.unflattened(layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: layer, groupSize: 64, bits: 4)
        eval(layer.parameters())
        let fused = try XCTUnwrap(layer.fusedLayer())
        let x = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
        let mvT = Qwen3TTSFusedStep.matvecThreads

        let trivial = MLXFast.metalKernel(name: "qwen_trivial", inputNames: ["a"], outputNames: ["o"],
                                          source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];")
        time("trivial custom kernel, chained x40", iterations: 30) {
            var y = x
            for _ in 0 ..< 40 {
                y = trivial([y], grid: (1024, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[1, 1, 1024]], outputDTypes: [.bfloat16])[0]
            }
            return y
        }
        time("MLX rmsNorm, chained x40", iterations: 30) {
            var y = x
            for _ in 0 ..< 40 { y = layer.inputLayernorm(y) }
            return y
        }
        let plain = Linear(1024, 1024, bias: false)
        plain.update(parameters: ModuleParameters.unflattened([("weight", plain.weight.asType(.bfloat16))]))
        let square = QuantizedLinear(plain, groupSize: 64, bits: 4)
        eval(square.parameters())
        time("MLX quantizedMM 1024x1024, chained x40", iterations: 30) {
            var y = x
            for _ in 0 ..< 40 { y = square(y) }
            return y
        }
        let matvec = Qwen3TTSFusedStep.kernel(.matvec, dtype: .bfloat16, paramType: .bfloat16, k: 1024, nt: mvT)
        time("fused matvec 1024x1024 + residual, chained x40", iterations: 30) {
            var y = x
            for _ in 0 ..< 40 {
                y = matvec([y, square.weight, square.scales, square.biases!, x],
                           grid: (1024 / (mvT / 8) * mvT, 1, 1), threadGroup: (mvT, 1, 1),
                           outputShapes: [[1, 1, 1024]], outputDTypes: [.bfloat16])[0]
            }
            return y
        }
        let caches = (0 ..< 8).map { _ in KVCacheSimple() }
        time("8 fused layer steps (one eval)", iterations: 100) {
            var h = x
            for c in caches { h = Qwen3TTSFusedStep.run(h, layer: fused, cache: c) }
            return h
        }
        Qwen3TTSModel.fusedLayers = false
        let caches2 = (0 ..< 8).map { _ in KVCacheSimple() }
        time("8 module layer steps (one eval)", iterations: 100) {
            var h = x
            for c in caches2 { h = layer(h, positionEmbeddings: (h, h), mask: nil, cache: c) }
            return h
        }
    }
}
