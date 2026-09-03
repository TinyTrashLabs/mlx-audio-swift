import Foundation
import MLX
import MLXNN

/// Swaps in quantized layers before weights are loaded.
///
/// A quantized checkpoint stores `scales`/`biases` alongside each quantized
/// `weight`, and those only fit a `QuantizedLinear`/`QuantizedEmbedding` — a
/// plain `Linear` rejects them outright. So the module tree has to be converted
/// first, and converted for EXACTLY the layers the converter actually
/// quantized: it skips anything narrower than the group size, so guessing from
/// the config would quantize layers whose weights are still bf16.
///
/// The checkpoint itself is the authority. A layer is quantized here if, and
/// only if, the file carries scales for it.
enum Dia2Quantization {
    static func apply(to model: Module, weights: [String: MLXArray], config: Dia2Config) {
        guard let q = config.quantization else { return }
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, _ in
            weights["\(path).scales"] != nil
        }
    }
}
