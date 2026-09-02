import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXAudioCodecs
import MLXLMCommon

/// Dia2: streaming two-speaker dialogue TTS. English only, two minutes of
/// context, two speakers maximum.
public final class Dia2Model: @unchecked Sendable {
    public let runtime: Dia2Runtime
    private let tokenizer: Dia2Tokenizer

    init(runtime: Dia2Runtime, tokenizer: Dia2Tokenizer) {
        self.runtime = runtime
        self.tokenizer = tokenizer
    }

    public var sampleRate: Int { runtime.sampleRate }
    /// The `(laughs)`-style vocabulary this checkpoint was trained on.
    public var nonverbalTags: [String] { tokenizer.nonverbalTags }

    public static func load(from directory: URL) async throws -> Dia2Model {
        let runtime = try await Dia2Runtime.load(from: directory)
        guard let tokenizer = runtime.tokenizer as? Dia2Tokenizer else {
            throw NSError(domain: "Dia2Model", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Dia2 runtime built without a Dia2Tokenizer",
            ])
        }
        return Dia2Model(runtime: runtime, tokenizer: tokenizer)
    }

    public static func fromPretrained(
        _ repoId: String = "tinytrashlabs/dia2-2b-mlx-8bit",
        cache: HubCache = .default,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> Dia2Model {
        guard let id = Repo.ID(rawValue: repoId) else {
            throw NSError(domain: "Dia2Model", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Invalid repository ID: \(repoId)",
            ])
        }
        let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: id, requiredExtension: "safetensors", hfToken: hfToken, cache: cache)
        return try await load(from: directory)
    }

    /// Batch: run to completion and return the whole take.
    public func generateDialogue(
        script: [String],
        prefixes: (speaker1: Dia2PrefixInput?, speaker2: Dia2PrefixInput?) = (nil, nil),
        config: Dia2GenerationConfig = Dia2GenerationConfig()
    ) async throws -> (samples: [Float], words: [(String, Double)]) {
        let plan = try Dia2Prefix.plan(speaker1: prefixes.speaker1,
                                       speaker2: prefixes.speaker2, runtime: runtime)
        let session = Dia2Session(runtime: runtime, config: config, prefix: plan)
        await session.append(script)
        await session.finish()
        var samples: [Float] = []
        var words: [(String, Double)] = []
        for try await chunk in session.audio {
            samples.append(contentsOf: chunk.samples)
            words.append(contentsOf: chunk.words)
        }
        return (samples, words)
    }

    /// Streaming: the session accepts more script while audio is flowing.
    public func streamDialogue(
        script: [String] = [],
        prefixes: (speaker1: Dia2PrefixInput?, speaker2: Dia2PrefixInput?) = (nil, nil),
        config: Dia2GenerationConfig = Dia2GenerationConfig()
    ) throws -> Dia2Session {
        let plan = try Dia2Prefix.plan(speaker1: prefixes.speaker1,
                                       speaker2: prefixes.speaker2, runtime: runtime)
        let session = Dia2Session(runtime: runtime, config: config, prefix: plan)
        if !script.isEmpty {
            Task { await session.append(script) }
        }
        return session
    }
}

extension Dia2Model: SpeechGenerationModel {
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 1500, temperature: 0.8, topP: 1.0)
    }

    public func generate(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = (voice, refAudio, refText, language, generationParameters)
        // No prefix here: the protocol has no place for word timings, and a
        // clip without them cannot condition Dia2. EngineKit calls
        // generateDialogue directly when it has an aligned pack.
        let result = try await generateDialogue(script: [text])
        return MLXArray(result.samples)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText,
                       language: language, generationParameters: generationParameters,
                       streamingInterval: 0.5)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = (voice, refAudio, refText, language, generationParameters, streamingInterval)
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { [weak self] in
            guard let self else { return continuation.finish() }
            do {
                let session = try streamDialogue(script: [text])
                await session.finish()
                for try await chunk in session.audio {
                    // AudioGeneration is an enum here (.token/.info/.audio),
                    // not a struct — the sample rate travels on the model, not
                    // on each chunk.
                    continuation.yield(.audio(MLXArray(chunk.samples)))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }
}
