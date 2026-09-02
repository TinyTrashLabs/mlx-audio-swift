import Foundation

/// Decodes Nari Labs' Dia2 `config.json` unchanged, so the file we publish in
/// our converted-weights repos is byte-identical to theirs.
public struct Dia2Config: Codable, Sendable {
    public let data: DataConfig
    public let model: ModelConfig
    public let runtime: RuntimeConfig
    public let assets: AssetsConfig?

    public struct DataConfig: Codable, Sendable {
        /// Total streams: 2 text + N audio codebooks.
        public let channels: Int
        public let textVocabSize: Int
        public let audioVocabSize: Int
        public let actionVocabSize: Int
        public let textPadTokenID: Int
        public let textNewWordTokenID: Int
        public let textZeroTokenID: Int
        public let audioPadTokenID: Int
        public let audioBosTokenID: Int
        public let actionPadTokenID: Int
        public let actionNewWordTokenID: Int
        /// Per-codebook frame offset. Codebook 0 leads; the rest trail it.
        public let delayPattern: [Int]
        public let firstWordMinStart: Int
        public let maxPad: Int
        /// How many words ahead the second text stream is multiplexed.
        public let secondStreamAhead: Int

        public var numAudioChannels: Int { max(0, channels - 2) }

        enum CodingKeys: String, CodingKey {
            case channels
            case textVocabSize = "text_vocab_size"
            case audioVocabSize = "audio_vocab_size"
            case actionVocabSize = "action_vocab_size"
            case textPadTokenID = "text_pad_token_id"
            case textNewWordTokenID = "text_new_word_token_id"
            case textZeroTokenID = "text_zero_token_id"
            case audioPadTokenID = "audio_pad_token_id"
            case audioBosTokenID = "audio_bos_token_id"
            case actionPadTokenID = "action_pad_token_id"
            case actionNewWordTokenID = "action_new_word_token_id"
            case delayPattern = "delay_pattern"
            case firstWordMinStart = "first_word_min_start"
            case maxPad = "max_pad"
            case secondStreamAhead = "second_stream_ahead"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            channels = try c.decode(Int.self, forKey: .channels)
            textVocabSize = try c.decode(Int.self, forKey: .textVocabSize)
            audioVocabSize = try c.decode(Int.self, forKey: .audioVocabSize)
            actionVocabSize = try c.decode(Int.self, forKey: .actionVocabSize)
            textPadTokenID = try c.decode(Int.self, forKey: .textPadTokenID)
            textNewWordTokenID = try c.decode(Int.self, forKey: .textNewWordTokenID)
            textZeroTokenID = try c.decodeIfPresent(Int.self, forKey: .textZeroTokenID) ?? 7
            audioPadTokenID = try c.decodeIfPresent(Int.self, forKey: .audioPadTokenID)
                ?? (audioVocabSize - 1)
            audioBosTokenID = try c.decodeIfPresent(Int.self, forKey: .audioBosTokenID)
                ?? (audioVocabSize - 2)
            actionPadTokenID = try c.decode(Int.self, forKey: .actionPadTokenID)
            actionNewWordTokenID = try c.decode(Int.self, forKey: .actionNewWordTokenID)
            delayPattern = try c.decodeIfPresent([Int].self, forKey: .delayPattern) ?? []
            firstWordMinStart = try c.decodeIfPresent(Int.self, forKey: .firstWordMinStart) ?? 0
            maxPad = try c.decodeIfPresent(Int.self, forKey: .maxPad) ?? 0
            secondStreamAhead = try c.decodeIfPresent(Int.self, forKey: .secondStreamAhead) ?? 0
        }
    }

    public struct DecoderConfig: Codable, Sendable {
        public let nLayer: Int
        public let nEmbd: Int
        public let nHidden: Int
        public let gqaQueryHeads: Int
        public let kvHeads: Int
        public let gqaHeadDim: Int
        public let lowRankDim: Int?

        enum CodingKeys: String, CodingKey {
            case nLayer = "n_layer", nEmbd = "n_embd", nHidden = "n_hidden"
            case gqaQueryHeads = "gqa_query_heads", kvHeads = "kv_heads"
            case gqaHeadDim = "gqa_head_dim", lowRankDim = "low_rank_dim"
        }
    }

    public struct DepformerConfig: Codable, Sendable {
        public let nLayer: Int
        public let nEmbd: Int
        public let nHidden: Int
        public let gqaQueryHeads: Int
        public let kvHeads: Int
        public let gqaHeadDim: Int
        public let applyRope: Bool
        public let textEmbedding: Bool
        public let mlpActivations: [String]
        /// Set by Dia2Config.init once `data.channels` is known.
        public internal(set) var numDepth: Int = 0

        enum CodingKeys: String, CodingKey {
            case nLayer = "n_layer", nEmbd = "n_embd", nHidden = "n_hidden"
            case gqaQueryHeads = "gqa_query_heads", kvHeads = "kv_heads"
            case gqaHeadDim = "gqa_head_dim", applyRope = "apply_rope"
            case textEmbedding = "text_embedding", mlpActivations = "mlp_activations"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            nLayer = try c.decode(Int.self, forKey: .nLayer)
            nEmbd = try c.decode(Int.self, forKey: .nEmbd)
            nHidden = try c.decode(Int.self, forKey: .nHidden)
            gqaQueryHeads = try c.decode(Int.self, forKey: .gqaQueryHeads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            gqaHeadDim = try c.decode(Int.self, forKey: .gqaHeadDim)
            applyRope = try c.decodeIfPresent(Bool.self, forKey: .applyRope) ?? true
            textEmbedding = try c.decodeIfPresent(Bool.self, forKey: .textEmbedding) ?? true
            mlpActivations = try c.decodeIfPresent([String].self, forKey: .mlpActivations)
                ?? ["silu", "linear"]
        }
    }

    public struct LinearHeadConfig: Codable, Sendable {
        public let mlpActivations: [String]
        enum CodingKeys: String, CodingKey { case mlpActivations = "mlp_activations" }
        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            mlpActivations = try c.decodeIfPresent([String].self, forKey: .mlpActivations)
                ?? ["silu", "linear"]
        }
    }

    public struct ModelConfig: Codable, Sendable {
        public let decoder: DecoderConfig
        public internal(set) var depformer: DepformerConfig
        public let linear: LinearHeadConfig
        public let ropeMinTimescale: Int
        public let ropeMaxTimescale: Float
        public let normalizationLayerEpsilon: Float

        enum CodingKeys: String, CodingKey {
            case decoder, depformer, linear
            case ropeMinTimescale = "rope_min_timescale"
            case ropeMaxTimescale = "rope_max_timescale"
            case normalizationLayerEpsilon = "normalization_layer_epsilon"
        }

        public init(from dec: Swift.Decoder) throws {
            let c = try dec.container(keyedBy: CodingKeys.self)
            decoder = try c.decode(DecoderConfig.self, forKey: .decoder)
            depformer = try c.decode(DepformerConfig.self, forKey: .depformer)
            linear = try c.decodeIfPresent(LinearHeadConfig.self, forKey: .linear)
                ?? LinearHeadConfig(mlpActivations: ["silu", "linear"])
            ropeMinTimescale = try c.decodeIfPresent(Int.self, forKey: .ropeMinTimescale) ?? 1
            ropeMaxTimescale = try c.decodeIfPresent(Float.self, forKey: .ropeMaxTimescale) ?? 10000
            normalizationLayerEpsilon =
                try c.decodeIfPresent(Float.self, forKey: .normalizationLayerEpsilon) ?? 1e-5
        }
    }

    public struct RuntimeConfig: Codable, Sendable {
        public let weightsSchedule: [Int]
        public let maxContextSteps: Int
        enum CodingKeys: String, CodingKey {
            case weightsSchedule = "weights_schedule"
            case maxContextSteps = "max_context_steps"
        }
        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            weightsSchedule = try c.decodeIfPresent([Int].self, forKey: .weightsSchedule) ?? []
            maxContextSteps = try c.decodeIfPresent(Int.self, forKey: .maxContextSteps) ?? 1500
        }
    }

    public struct AssetsConfig: Codable, Sendable {
        public let tokenizer: String?
        public let mimi: String?
    }

    enum CodingKeys: String, CodingKey { case data, model, runtime, assets }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        data = try c.decode(DataConfig.self, forKey: .data)
        var m = try c.decode(ModelConfig.self, forKey: .model)
        runtime = try c.decode(RuntimeConfig.self, forKey: .runtime)
        assets = try c.decodeIfPresent(AssetsConfig.self, forKey: .assets)
        // Depth is derived, not declared: one depformer stage per codebook
        // after codebook 0, which the transformer's own head produces.
        m.depformer.numDepth = max(0, data.numAudioChannels - 1)
        model = m
    }
}

extension Dia2Config.LinearHeadConfig {
    init(mlpActivations: [String]) { self.mlpActivations = mlpActivations }
}
