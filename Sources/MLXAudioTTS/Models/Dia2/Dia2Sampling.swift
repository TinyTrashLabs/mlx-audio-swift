import Foundation
import MLX

/// Codebook delay bookkeeping. Codebook 0 leads by 16 frames and the rest by
/// 18, so a frame's codebooks describe different instants; these two functions
/// convert between the model's staggered view and real time.
public enum Dia2Grid {
    public static func delay(_ aligned: MLXArray, delays: [Int], padID: Int) -> MLXArray {
        let channels = aligned.dim(0)
        let total = aligned.dim(1)
        let maxDelay = delays.max() ?? 0
        var out = MLXArray.full([channels, total + maxDelay], values: MLXArray(Int32(padID)),
                                type: Int32.self)
        for (index, delay) in delays.enumerated() where index < channels {
            out[index, delay ..< (delay + total)] = aligned[index].asType(.int32)
        }
        return out
    }

    public static func undelay(_ delayed: MLXArray, delays: [Int], padID: Int) -> MLXArray {
        let channels = delayed.dim(0)
        let total = delayed.dim(1)
        let maxDelay = delays.max() ?? 0
        let target = max(0, total - maxDelay)
        var out = MLXArray.full([channels, target], values: MLXArray(Int32(padID)), type: Int32.self)
        for (index, delay) in delays.enumerated() where index < channels {
            out[index] = delayed[index, delay ..< (delay + target)]
        }
        return out
    }

    /// PAD and BOS are control tokens; sampling either would emit a click.
    public static func maskAudioLogits(_ logits: MLXArray, padIdx: Int, bosIdx: Int) -> MLXArray {
        let vocab = logits.dim(-1)
        guard vocab > 0 else { return logits }
        var out = logits
        let negInf = MLXArray(-Float.greatestFiniteMagnitude)
        for index in [padIdx, bosIdx] where index >= 0 && index < vocab {
            out[.ellipsis, index] = negInf.asType(out.dtype)
        }
        return out
    }
}

/// Classifier-free guidance. Batch row 0 is the conditional branch, row 1 the
/// unconditional one; the result keeps only row 0.
public enum Dia2Guidance {
    public static func apply(_ logits: MLXArray, active: Bool, scale: Float, filterK: Int) -> MLXArray {
        guard active, logits.dim(0) > 1 else { return logits }
        let conditional = logits[0 ..< 1].asType(.float32)
        let unconditional = logits[1 ..< 2].asType(.float32)
        let guided = unconditional + (conditional - unconditional) * scale
        guard filterK > 0, guided.dim(-1) > 0 else { return guided.asType(logits.dtype) }
        // Keep the guided top-k, but score them with the CONDITIONAL logits —
        // guidance selects the candidates, it does not distort their ranking.
        let k = min(filterK, guided.dim(-1))
        let threshold = MLX.top(guided, k: k, axis: -1)[.ellipsis, (k - 1) ..< k]
        let keep = guided .>= threshold
        let negInf = MLXArray(-Float.infinity)
        return MLX.where(keep, conditional, negInf).asType(logits.dtype)
    }
}

public enum Dia2Sampler {
    /// Returns one token id. `temperature <= 0` is greedy.
    public static func sample(_ logits: MLXArray, temperature: Float, topK: Int,
                              key: MLXArray?) -> Int {
        let flat = logits.reshaped([-1, logits.dim(-1)]).asType(.float32)
        if temperature <= 0 {
            return argMax(flat, axis: -1).item(Int32.self).asInt
        }
        let scaled = flat / MLXArray(max(temperature, 1e-6))
        let vocab = scaled.dim(-1)
        if topK > 0 && topK < vocab {
            let top = MLX.top(scaled, k: topK, axis: -1)
            let threshold = top[.ellipsis, (topK - 1) ..< topK]
            let filtered = MLX.where(scaled .>= threshold, scaled, MLXArray(-Float.infinity))
            return MLXRandom.categorical(filtered, key: key).item(Int32.self).asInt
        }
        return MLXRandom.categorical(scaled, key: key).item(Int32.self).asInt
    }
}

private extension Int32 { var asInt: Int { Int(self) } }
