import Foundation
import MLX
import MLXLMCommon
import MLXAudioCodecs



public extension Dia2Runtime {
    /// Builds a runtime from a local Dia2 checkout: config, both networks,
    /// Mimi (downloaded on first use), and the tokenizer.
    static func load(from directory: URL) async throws -> Dia2Runtime {
        let config = try JSONDecoder().decode(
            Dia2Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let transformer = Dia2Transformer(config: config)
        try transformer.loadWeights(from: directory)
        let depformer = Dia2Depformer(config: config)
        try depformer.loadWeights(from: directory)
        let mimi = try await Mimi.fromPretrained(progressHandler: { _ in })
        let tokenizer = try await Dia2Tokenizer(modelFolder: directory)
        let data = config.data
        let ids = Dia2TokenIDs(
            card: data.textVocabSize, newWord: data.textNewWordTokenID, pad: data.textPadTokenID,
            // The reference reads `getattr(tokenizer, "bos_token_id", 1) or 1`.
            // That trailing `or 1` also replaces a ZERO, because 0 is falsy in
            // Python — Swift's `??` only replaces nil. This tokenizer reports 0,
            // so we fed 0 where the reference feeds 1, and the very first token
            // of every generation was wrong.
            bos: Dia2TokenIDs.resolvedBOS(tokenizer.bosTokenID), zero: data.textZeroTokenID,
            spk1: tokenizer.id(of: "[S1]") ?? data.textNewWordTokenID,
            spk2: tokenizer.id(of: "[S2]") ?? data.textNewWordTokenID,
            audioPad: data.audioPadTokenID, audioBos: data.audioBosTokenID)
        return Dia2Runtime(config: config, transformer: transformer, depformer: depformer,
                           mimi: mimi, tokenizer: tokenizer, tokenIDs: ids,
                           delays: data.delayPattern)
    }
}

import Tokenizers

/// Adapter over swift-transformers, plus the added-token lookups Dia2 needs.
public struct Dia2Tokenizer: Dia2TextTokenizing, @unchecked Sendable {
    private let inner: any Tokenizers.Tokenizer
    private let added: [String: Int]

    public init(modelFolder: URL) async throws {
        inner = try await AutoTokenizer.from(modelFolder: modelFolder)
        let addedURL = modelFolder.appendingPathComponent("added_tokens.json")
        added = (try? JSONDecoder().decode([String: Int].self, from: Data(contentsOf: addedURL))) ?? [:]
    }

    public func encode(_ text: String) -> [Int] { inner.encode(text: text) }
    public func id(of token: String) -> Int? { added[token] ?? inner.convertTokenToId(token) }
    public var bosTokenID: Int? { inner.bosTokenId }
    /// The `(laughs)`-style vocabulary, for the Direction chips in the app.
    public var nonverbalTags: [String] {
        added.keys.filter { $0.hasPrefix("(") && $0.hasSuffix(")") }.sorted()
    }
}
