import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Dia2's rotary embedding. Timescales are geometric between min and max, and
/// the rotation splits the head dim in half (`[-x2, x1]`), matching the
/// reference's `torch.chunk(x, 2, dim=-1)` — NOT the interleaved variant.
/// Deliberately NOT a `Module`: the cos/sin tables are constants derived from
/// the config, not learned weights. As a Module they would be collected as
/// parameters and `update(parameters:verify: .all)` would demand they appear in
/// model.safetensors, which they never do.
final class Dia2RoPE {
    private let cosCache: MLXArray
    private let sinCache: MLXArray

    init(headDim: Int, minTimescale: Int, maxTimescale: Float, maxSeqLen: Int) {
        precondition(headDim % 2 == 0, "RoPE dimension must be even")
        let half = headDim / 2
        // Computed in Double on the CPU: these are the geometric timescales the
        // reference builds in float32, and the cache is materialised once, so
        // the extra precision costs nothing and avoids a powf rounding drift.
        let minTS = Double(minTimescale)
        let ratio = Double(maxTimescale) / minTS
        let fraction = (0 ..< half).map { 2.0 * Double($0) / Double(headDim) }
        let invFreq = fraction.map { 1.0 / (minTS * pow(ratio, $0)) }
        let t = MLXArray(converting: (0 ..< maxSeqLen).map { Double($0) }).reshaped([maxSeqLen, 1])
        let freqs = t * MLXArray(converting: invFreq).reshaped([1, half])
        let emb = concatenated([freqs, freqs], axis: -1)   // [maxSeqLen, headDim]
        cosCache = cos(emb)
        sinCache = sin(emb)
    }

    /// - Parameters:
    ///   - x: `[B, T, H, D]`
    ///   - positions: `[B, T]` of Int32
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let c = cosCache[positions].expandedDimensions(axis: 2).asType(x.dtype)  // [B,T,1,D]
        let s = sinCache[positions].expandedDimensions(axis: 2).asType(x.dtype)
        let d = x.dim(-1)
        let x1 = x[.ellipsis, 0 ..< (d / 2)]
        let x2 = x[.ellipsis, (d / 2) ..< d]
        let rotated = concatenated([-x2, x1], axis: -1)
        return (x * c) + (rotated * s)
    }
}

/// Two text streams share one embedding table but get separate projections.
/// The second stream carries the lookahead word and is gated off when it is pad.
final class MultiStreamEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "main_proj") var mainProj: Linear
    @ModuleInfo(key: "second_proj") var secondProj: Linear
    private let padID: Int

    init(vocabSize: Int, dim: Int, padID: Int, lowRankDim: Int?) {
        let base = lowRankDim ?? dim
        self.padID = padID
        _embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: base)
        _mainProj.wrappedValue = Linear(base, dim, bias: false)
        _secondProj.wrappedValue = Linear(base, dim, bias: false)
        super.init()
    }

    func mainOnly(_ main: MLXArray) -> MLXArray {
        mainProj(embedding(main).asType(.float32))
    }

    func callAsFunction(_ main: MLXArray, _ second: MLXArray) -> MLXArray {
        let outMain = mainProj(embedding(main).asType(.float32))
        let outSecond = secondProj(embedding(second).asType(.float32))
        let usable = (second .!= MLXArray(Int32(padID))).expandedDimensions(axis: -1)
        return outMain + MLX.where(usable, outSecond, MLXArray(Float(0)))
    }
}

/// Gated MLP: one fused `wi` of width `branches * hidden`, split into gate and
/// up, combined as `act0(gate) * act1(up)`.
final class Dia2Mlp: Module, UnaryLayer {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    private let hidden: Int
    private let activations: [String]

    init(dim: Int, hidden: Int, activations: [String]) {
        precondition(activations.count == 2, "Dia2Mlp expects two activations")
        self.hidden = hidden
        self.activations = activations
        _wi.wrappedValue = Linear(dim, activations.count * hidden, bias: false)
        _wo.wrappedValue = Linear(hidden, dim, bias: false)
        super.init()
    }

    private func apply(_ name: String, _ x: MLXArray) -> MLXArray {
        switch name.lowercased() {
        case "silu", "swish", "swiglu": return silu(x)
        case "gelu", "geglu": return gelu(x)
        case "relu": return relu(x)
        case "linear": return x
        default: fatalError("Unsupported Dia2 activation \(name)")
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let proj = wi(x.asType(.float32))
        let shaped = proj.reshaped(Array(x.shape.dropLast()) + [2, hidden])
        let gate = shaped[.ellipsis, 0, 0...]
        let up = shaped[.ellipsis, 1, 0...]
        return wo(apply(activations[0], gate) * apply(activations[1], up))
    }
}

/// Decoder attention. Q and K are RMSNormed per head BEFORE RoPE, which is why
/// the SDPA scale is 1.0 rather than 1/sqrt(headDim) — see the spec's Global
/// Constraints. Using the usual scale here produces fluent-sounding garbage.
final class Dia2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numQueryHeads: Int
    let numKVHeads: Int
    let headDim: Int
    private let rope: Dia2RoPE

    init(config: Dia2Config, dim: Int) {
        let dec = config.model.decoder
        numQueryHeads = dec.gqaQueryHeads
        numKVHeads = dec.kvHeads
        headDim = dec.gqaHeadDim
        _qProj.wrappedValue = Linear(dim, numQueryHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numQueryHeads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.model.normalizationLayerEpsilon)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.model.normalizationLayerEpsilon)
        rope = Dia2RoPE(headDim: headDim,
                        minTimescale: config.model.ropeMinTimescale,
                        maxTimescale: config.model.ropeMaxTimescale,
                        maxSeqLen: config.runtime.maxContextSteps + 64)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: KVCacheSimple?) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        var q = qProj(x.asType(.float32)).reshaped(B, T, numQueryHeads, headDim)
        var k = kProj(x.asType(.float32)).reshaped(B, T, numKVHeads, headDim)
        let v = vProj(x.asType(.float32)).reshaped(B, T, numKVHeads, headDim)
        q = qNorm(q)
        k = kNorm(k)
        q = rope(q, positions: positions)
        k = rope(k, positions: positions)

        let out = attentionWithCacheUpdate(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            cache: cache,
            scale: 1.0,
            mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(B, T, numQueryHeads * headDim)
        return oProj(out.asType(.float32)).asType(x.dtype)
    }
}
