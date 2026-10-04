@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN
import Foundation

// Breeze's backbone is a stock Qwen3 decoder stack. Upstream reuses the public
// Qwen3Attention/Qwen3MLP/Qwen3TransformerBlock that its OmniVoice change
// (Blaizzy/mlx-audio-swift#209) carved out of the VyvoTTS backbone. This fork
// predates that refactor and keeps those classes private, so Breeze carries
// its own copy instead of rewriting a backbone other models depend on. Same
// math and the same weight keys as upstream's blocks.

final class BreezeQwen3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: BreezeBackboneConfig) {
        let dim = config.hiddenSize
        let headDim = config.headDim
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        scale = Foundation.pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        var ropeScale: Float = 1
        if let scaling = config.ropeScaling, scaling["type"] == .string("linear"),
           let factor = scaling["factor"]?.asFloat() {
            ropeScale = 1 / factor
        }
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta, scale: ropeScale)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, numHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, numKVHeads, -1)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class BreezeQwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class BreezeQwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: BreezeQwen3Attention
    let mlp: BreezeQwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: BreezeBackboneConfig) {
        _attention.wrappedValue = BreezeQwen3Attention(config)
        mlp = BreezeQwen3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}
