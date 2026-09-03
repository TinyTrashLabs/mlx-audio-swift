import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Dia2DecoderLayer: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: Dia2Attention
    @ModuleInfo(key: "mlp") var mlp: Dia2Mlp

    init(config: Dia2Config) {
        let dec = config.model.decoder
        let eps = config.model.normalizationLayerEpsilon
        _preNorm.wrappedValue = RMSNorm(dimensions: dec.nEmbd, eps: eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: dec.nEmbd, eps: eps)
        _attn.wrappedValue = Dia2Attention(config: config, dim: dec.nEmbd)
        _mlp.wrappedValue = Dia2Mlp(dim: dec.nEmbd, hidden: dec.nHidden,
                                    activations: config.model.linear.mlpActivations)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: KVCacheSimple?) -> MLXArray {
        let h = x + attn(preNorm(x), positions: positions, cache: cache)
        return h + mlp(postNorm(h))
    }
}

/// The temporal backbone. One step per Mimi frame; emits the action logits
/// (pad vs new word) and codebook 0 directly, and the hidden state the
/// depformer expands into the remaining 31 codebooks.
public final class Dia2Transformer: Module {
    @ModuleInfo(key: "audio_embeds") var audioEmbeds: [Embedding]
    @ModuleInfo(key: "text_embed") var textEmbed: MultiStreamEmbedding
    @ModuleInfo(key: "layers") var layers: [Dia2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "action_head") var actionHead: Linear
    @ModuleInfo(key: "cb0_head") var cb0Head: Linear

    let config: Dia2Config
    private let numAudioChannels: Int

    public init(config: Dia2Config) {
        self.config = config
        let dec = config.model.decoder
        let data = config.data
        numAudioChannels = data.numAudioChannels
        _audioEmbeds.wrappedValue = (0 ..< numAudioChannels).map { _ in
            Embedding(embeddingCount: data.audioVocabSize, dimensions: dec.nEmbd)
        }
        _textEmbed.wrappedValue = MultiStreamEmbedding(
            vocabSize: data.textVocabSize, dim: dec.nEmbd,
            padID: data.textPadTokenID, lowRankDim: dec.lowRankDim)
        _layers.wrappedValue = (0 ..< dec.nLayer).map { _ in Dia2DecoderLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: dec.nEmbd,
                                     eps: config.model.normalizationLayerEpsilon)
        _actionHead.wrappedValue = Linear(dec.nEmbd, data.actionVocabSize, bias: false)
        _cb0Head.wrappedValue = Linear(dec.nEmbd, data.audioVocabSize, bias: false)
        super.init()
    }

    public func makeCache() -> [KVCacheSimple] {
        (0 ..< config.model.decoder.nLayer).map { _ in KVCacheSimple() }
    }

    /// - Parameter tokens: `[B, channels, 1]` — row 0 main text, row 1 second
    ///   text, rows 2... the audio codebooks for this frame.
    public func step(_ tokens: MLXArray, positions: MLXArray, cache: [KVCacheSimple])
        -> (hidden: MLXArray, action: MLXArray, cb0: MLXArray)
    {
        precondition(tokens.dim(2) == 1, "Dia2 decodes one frame at a time")
        var h = textEmbed(tokens[0..., 0, 0...], tokens[0..., 1, 0...])
        for i in 0 ..< numAudioChannels {
            h = h + audioEmbeds[i](tokens[0..., i + 2, 0...])
        }
        for (i, layer) in layers.enumerated() {
            h = layer(h, positions: positions, cache: cache[i])
        }
        let normed = norm(h)
        let f32 = normed.asType(.float32)
        return (normed, actionHead(f32), cb0Head(f32))
    }
}

public extension Dia2Transformer {
    /// Loads `model.safetensors` from a Dia2 checkout. Keys already match the
    /// PyTorch module paths, so only the depformer's dictionary-keyed modules
    /// need remapping (handled in Dia2Depformer); the transformer is 1:1.
    func loadWeights(from directory: URL) throws {
        let url = directory.appendingPathComponent("model.safetensors")
        let all = try loadArrays(url: url)
        let mine = all.reduce(into: [String: MLXArray]()) { out, kv in
            guard kv.key.hasPrefix("transformer.") else { return }
            out[String(kv.key.dropFirst("transformer.".count))] = kv.value
        }
        try update(parameters: ModuleParameters.unflattened(mine), verify: .all)
        eval(self)
    }
}
