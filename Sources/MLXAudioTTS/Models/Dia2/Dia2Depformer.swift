import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Attention whose projection weights are chosen per stage by the config's
/// `weights_schedule`: 31 stages share 5 weight sets, and the stage index also
/// serves as the RoPE position. Weight keys are `in_proj.<id>` / `out_proj.<id>`
/// where `<id>` is the schedule entry, not the stage.
final class Dia2ScheduleAttention: Module {
    // Arrays, not dictionaries keyed by the schedule id. The checkpoint stores
    // these as a ModuleDict whose keys are the decimal ids "0", "1", ...;
    // ModuleParameters.unflattened reads purely numeric keys as array indices,
    // so a [String: Linear] here fails to load against its own weights. The
    // schedule id is mapped through `slot` rather than used directly, so a
    // non-contiguous set of ids stays correct.
    @ModuleInfo(key: "in_proj") var inProj: [Linear]
    @ModuleInfo(key: "out_proj") var outProj: [Linear]
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numQueryHeads: Int
    let numKVHeads: Int
    let headDim: Int
    private let schedule: [Int]
    /// Weight-schedule id -> index into `inProj`/`outProj`.
    private let slot: [Int: Int]
    private let rope: Dia2RoPE?

    init(config: Dia2Config) {
        let dep = config.model.depformer
        schedule = config.runtime.weightsSchedule
        numQueryHeads = dep.gqaQueryHeads
        numKVHeads = dep.kvHeads
        headDim = dep.gqaHeadDim
        let used = Set(schedule).sorted()
        slot = Dictionary(uniqueKeysWithValues: used.enumerated().map { ($1, $0) })
        // Locals: the closures below would otherwise capture self before every
        // stored property is initialised.
        let qkvWidth = 3 * dep.gqaQueryHeads * dep.gqaHeadDim
        let attnWidth = dep.gqaQueryHeads * dep.gqaHeadDim
        _inProj.wrappedValue = used.map { _ in Linear(dep.nEmbd, qkvWidth, bias: false) }
        _outProj.wrappedValue = used.map { _ in Linear(attnWidth, dep.nEmbd, bias: false) }
        let eps = config.model.normalizationLayerEpsilon
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        rope = dep.applyRope
            ? Dia2RoPE(headDim: headDim, minTimescale: config.model.ropeMinTimescale,
                       maxTimescale: config.model.ropeMaxTimescale,
                       maxSeqLen: max(schedule.count, 1) + 8)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, stage: Int, cache: KVCacheSimple?) -> MLXArray {
        let i = slot[schedule[stage]]!
        let (B, T) = (x.dim(0), x.dim(1))
        // One fused projection produces Q, K and V together.
        let proj = inProj[i](x.asType(.float32))
            .reshaped(B, T, 3, numQueryHeads, headDim)
        var q = qNorm(proj[0..., 0..., 0, 0..., 0...])
        var k = kNorm(proj[0..., 0..., 1, 0..., 0...])
        let v = proj[0..., 0..., 2, 0..., 0...]
        if let rope {
            let pos = MLXArray([Int32(stage)]).reshaped([1, 1])
            q = rope(q, positions: pos)
            k = rope(k, positions: pos)
        }
        let out = attentionWithCacheUpdate(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            cache: cache,
            scale: 1.0,
            mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(B, T, numQueryHeads * headDim)
        return outProj[i](out.asType(.float32)).asType(x.dtype)
    }
}

final class Dia2DepformerLayer: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: RMSNorm
    @ModuleInfo(key: "self_attention") var selfAttention: Dia2ScheduleAttention
    @ModuleInfo(key: "mlp") var mlp: Dia2Mlp

    init(config: Dia2Config) {
        let dep = config.model.depformer
        let eps = config.model.normalizationLayerEpsilon
        _preNorm.wrappedValue = RMSNorm(dimensions: dep.nEmbd, eps: eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: dep.nEmbd, eps: eps)
        _selfAttention.wrappedValue = Dia2ScheduleAttention(config: config)
        _mlp.wrappedValue = Dia2Mlp(dim: dep.nEmbd, hidden: dep.nHidden,
                                    activations: dep.mlpActivations)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, stage: Int, cache: KVCacheSimple?) -> MLXArray {
        let h = x + selfAttention(preNorm(x), stage: stage, cache: cache)
        return h + mlp(postNorm(h))
    }
}

/// Expands one frame's hidden state into codebooks 1...31, one stage at a time,
/// each conditioned on the codebook the previous stage produced.
public final class Dia2Depformer: Module {
    @ModuleInfo(key: "audio_embeds") var audioEmbeds: [Embedding]
    @ModuleInfo(key: "text_embed") var textEmbed: MultiStreamEmbedding?
    @ModuleInfo(key: "depformer_in") var depformerIn: [Linear]
    @ModuleInfo(key: "layers") var layers: [Dia2DepformerLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "logits") var logitsHeads: [Linear]

    public let numDepth: Int
    /// Logits are truncated below PAD/BOS so neither can ever be sampled.
    public let audioVocabLimit: Int
    private let config: Dia2Config
    private let schedule: [Int]
    /// Weight-schedule id -> index into `depformerIn`, as in Dia2ScheduleAttention.
    private let slot: [Int: Int]

    public init(config: Dia2Config) {
        self.config = config
        let dep = config.model.depformer
        let data = config.data
        schedule = config.runtime.weightsSchedule
        numDepth = dep.numDepth
        audioVocabLimit = min(data.audioPadTokenID, data.audioBosTokenID)
        _audioEmbeds.wrappedValue = (0 ..< numDepth).map { _ in
            Embedding(embeddingCount: data.audioVocabSize, dimensions: dep.nEmbd)
        }
        _textEmbed.wrappedValue = dep.textEmbedding
            ? MultiStreamEmbedding(vocabSize: data.textVocabSize, dim: dep.nEmbd,
                                   padID: data.textPadTokenID, lowRankDim: nil)
            : nil
        let used = Set(schedule).sorted()
        slot = Dictionary(uniqueKeysWithValues: used.enumerated().map { ($1, $0) })
        _depformerIn.wrappedValue = used.map { _ in
            Linear(config.model.decoder.nEmbd, dep.nEmbd, bias: false)
        }
        _layers.wrappedValue = (0 ..< dep.nLayer).map { _ in Dia2DepformerLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: dep.nEmbd,
                                     eps: config.model.normalizationLayerEpsilon)
        _logitsHeads.wrappedValue = (0 ..< numDepth).map { _ in
            Linear(dep.nEmbd, data.audioVocabSize, bias: false)
        }
        super.init()
    }

    public func makeCache() -> [KVCacheSimple] {
        (0 ..< config.model.depformer.nLayer).map { _ in KVCacheSimple() }
    }

    /// Depth attention spans the stages of ONE frame; the cache is cleared
    /// before every frame, never carried across them.
    public func resetCache(_ cache: [KVCacheSimple]) {
        for c in cache { c.trim(c.offset) }
    }

    public func step(stage: Int, prevAudio: MLXArray, hidden: MLXArray,
                     cache: [KVCacheSimple],
                     mainText: MLXArray?, secondText: MLXArray?) -> MLXArray {
        precondition(stage >= 0 && stage < numDepth, "stage \(stage) out of range")
        var tokenEmb = audioEmbeds[stage](prevAudio.expandedDimensions(axis: -1))
        if stage == 0, let textEmbed, let mainText, let secondText {
            tokenEmb = tokenEmb + textEmbed(mainText.expandedDimensions(axis: -1),
                                            secondText.expandedDimensions(axis: -1))
        }
        var x = depformerIn[slot[schedule[stage]]!](hidden.asType(.float32)) + tokenEmb
        for (i, layer) in layers.enumerated() {
            x = layer(x, stage: stage, cache: cache[i])
        }
        // The reference unsqueezes a stage axis before truncating, so its logits
        // are [B, 1, 1, V]. Match it exactly — the parity fixture is stored in
        // that shape, and a silently rank-3 result would broadcast rather than
        // fail when it reaches the sampler.
        let logits = logitsHeads[stage](norm(x).asType(.float32))
            .expandedDimensions(axis: 1)
        return logits[.ellipsis, 0 ..< audioVocabLimit]
    }

    public func loadWeights(from directory: URL) throws {
        let all = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        let mine = all.reduce(into: [String: MLXArray]()) { out, kv in
            guard kv.key.hasPrefix("depformer.") else { return }
            out[String(kv.key.dropFirst("depformer.".count))] = kv.value
        }
        try update(parameters: ModuleParameters.unflattened(mine), verify: .all)
        eval(self)
    }
}
