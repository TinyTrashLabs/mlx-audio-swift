import MLX
import MLXNN
import MLXLMCommon
import XCTest

@testable import MLXAudioTTS

/// Not a correctness test: prints wall time per single-token layer step for
/// the module path and the fused kernels on this Mac's GPU. Run by name.
final class QwenFusedLayerBenchmark: XCTestCase {
    func testPrintStepTimes() throws {
        MLX.Device.setDefault(device: Device(.gpu))
        Qwen3TTSModel.fastRope = true
        defer { Qwen3TTSModel.fusedLayers = false; Qwen3TTSModel.fastRope = false }
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: "{}".data(using: .utf8)!)
        let layers = (0 ..< 8).map { TalkerDecoderLayer(config: config, layerIdx: $0) }
        for layer in layers {
            let params = layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
            layer.update(parameters: ModuleParameters.unflattened(params))
            quantize(model: layer, groupSize: 64, bits: 4)
        }
        eval(layers.map { $0.parameters() })
        let x = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
        for (name, fused) in [("module", false), ("fused", true), ("module", false), ("fused", true)] {
            Qwen3TTSModel.fusedLayers = fused
            let caches = layers.map { _ in KVCacheSimple() }
            // warm up (kernel compile, cache allocation)
            for _ in 0 ..< 20 {
                var h = x
                for (i, layer) in layers.enumerated() { h = layer(h, positionEmbeddings: (h, h), mask: nil, cache: caches[i]) }
                eval(h)
            }
            let steps = 200
            let t0 = Date()
            for _ in 0 ..< steps {
                var h = x
                for (i, layer) in layers.enumerated() { h = layer(h, positionEmbeddings: (h, h), mask: nil, cache: caches[i]) }
                eval(h)
            }
            let perLayer = Date().timeIntervalSince(t0) / Double(steps * layers.count) * 1e6
            print(String(format: "[fused-bench] %@: %.0f µs per layer step", name, perLayer))
        }
    }
}
