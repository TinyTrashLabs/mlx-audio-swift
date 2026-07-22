// SuperTonic 3 text-to-speech — native MLX port.
//
// Pipeline: text frontend (NFKD + <lang> tags + unicode indexer) -> duration
// predictor -> text encoder -> flow-matching Euler loop (CFG scale 4, 8 steps
// by default) -> WaveNeXt vocoder @ 44100 Hz.
//
// Weights: `Supertone/supertonic-3` ONNX graphs converted to MLX safetensors
// (see the Python spike converter). Model dir layout:
//   config.json (model_type "supertonic")
//   duration_predictor.safetensors / text_encoder.safetensors /
//   vector_estimator.safetensors / vocoder.safetensors
//   unicode_indexer.json
//   voice_styles/{M1..M5,F1..F5}.json
//
// Licensed under the BigScience Open RAIL-M (see LICENSE-SuperTonic + NOTICE
// in this directory). Preset voices only; no cloning, no emotion control.

import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon

public final class SupertonicModel: SpeechGenerationModel, @unchecked Sendable {
    public let sampleRate: Int
    public let defaultGenerationParameters = GenerateParameters(temperature: 0, topP: 1)

    public static let presetVoices = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

    /// Euler integration steps (ONNX default 8).
    public var steps: Int = 8
    /// Classifier-free-guidance scale (baked at 4.0 in the ONNX export; factored out here).
    public var cfgScale: Float = 4.0
    /// Speech-rate multiplier; >1 is faster.
    public var speed: Float = 1.0

    static let chunkSamples = 512 * 6  // samples per latent frame

    let frontend: SupertonicTextFrontend
    let durationPredictor: SupertonicDurationPredictor
    let textEncoder: SupertonicTextEncoder
    let vectorField: SupertonicVectorField
    let vocoder: SupertonicVocoder
    private let modelDirectory: URL
    private var styleCache: [String: (ttl: MLXArray, dp: MLXArray)] = [:]
    private let styleCacheLock = NSLock()

    // MLX on Apple silicon defaults to TF32-style tensor-op GEMMs for fp32
    // (~1e-2 abs error), which audibly degrades this fully-fp32 model. The
    // bundled MLX core reads MLX_ENABLE_TF32 once, lazily, at the first fp32
    // GEMM (mlx/utils.h enable_tf32()), so setting the env var before any
    // matmul restores exact fp32 accumulation process-wide. Best-effort: if
    // another fp32 GEMM already ran, the flag is latched and this is a no-op.
    private static let disableTF32: Void = {
        setenv("MLX_ENABLE_TF32", "0", 0)  // don't clobber an explicit user setting
    }()

    private init(modelDirectory: URL) throws {
        _ = Self.disableTF32
        let fm = FileManager.default
        func file(_ name: String) throws -> URL {
            let url = modelDirectory.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else {
                throw SupertonicError.missingFile(url.path)
            }
            return url
        }
        self.modelDirectory = modelDirectory
        sampleRate = 44100
        frontend = try SupertonicTextFrontend(indexerURL: try file("unicode_indexer.json"))
        durationPredictor = try .fromSafetensors(try file("duration_predictor.safetensors"))
        textEncoder = try .fromSafetensors(try file("text_encoder.safetensors"))
        vectorField = try .fromSafetensors(try file("vector_estimator.safetensors"))
        vocoder = try .fromSafetensors(try file("vocoder.safetensors"))
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> SupertonicModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> SupertonicModel {
        try SupertonicModel(modelDirectory: modelDir)
    }

    // MARK: - Voice styles

    func voiceStyle(_ name: String) throws -> (ttl: MLXArray, dp: MLXArray) {
        styleCacheLock.lock()
        defer { styleCacheLock.unlock() }
        if let cached = styleCache[name] { return cached }
        let url = modelDirectory.appendingPathComponent("voice_styles/\(name).json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SupertonicError.missingFile(url.path)
        }
        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let ttl = Self.styleArray(obj["style_ttl"]),
              let dp = Self.styleArray(obj["style_dp"])
        else {
            throw SupertonicError.invalidInput("malformed voice style \(name)")
        }
        let style = (ttl: ttl, dp: dp)
        styleCache[name] = style
        return style
    }

    private static func styleArray(_ value: Any?) -> MLXArray? {
        guard let d = value as? [String: Any],
              let dims = d["dims"] as? [Int],
              let data = d["data"] as? [Any]
        else { return nil }
        let flat = flatten(data)
        guard flat.count == dims.reduce(1, *) else { return nil }
        return MLXArray(flat).reshaped(dims)
    }

    private static func flatten(_ value: [Any]) -> [Float] {
        var out = [Float]()
        var stack: [Any] = value.reversed()
        while let item = stack.popLast() {
            if let nested = item as? [Any] {
                stack.append(contentsOf: nested.reversed())
            } else if let n = item as? NSNumber {
                out.append(n.floatValue)
            }
        }
        return out
    }

    // MARK: - Synthesis

    /// Core synthesis with optional injected noise (for parity testing).
    func synthesize(
        text: String,
        voice: String,
        language: String = "en",
        steps: Int? = nil,
        cfgScale: Float? = nil,
        speed: Float? = nil,
        noise: MLXArray? = nil
    ) throws -> MLXArray {
        let steps = steps ?? self.steps
        let cfg = cfgScale ?? self.cfgScale
        let speed = speed ?? self.speed
        let style = try voiceStyle(voice)

        let ids = try frontend.encode(text, lang: language)
        guard !ids.isEmpty else {
            throw AudioGenerationError.invalidInput("empty text")
        }
        let idsArr = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let mask = MLXArray.ones([1, 1, ids.count])

        let duration = durationPredictor(idsArr, styleDp: style.dp, textMask: mask)
        let durationSec = duration.item(Float.self) / speed
        let latentLen = max(1, Int((Double(durationSec) * Double(sampleRate) + Double(Self.chunkSamples) - 1) / Double(Self.chunkSamples)))

        let textEmb = textEncoder(idsArr, styleTtl: style.ttl, textMask: mask)

        var x = noise ?? MLXRandom.normal([1, 144, latentLen], type: Float.self)
        guard x.shape == [1, 144, latentLen] else {
            throw AudioGenerationError.invalidInput(
                "noise shape \(x.shape) != [1, 144, \(latentLen)]")
        }
        let latentMask = MLXArray.ones([1, 1, latentLen])
        let total = MLXArray([Float(steps)])
        for step in 0 ..< steps {
            x = vectorField.eulerCFG(
                x, textEmb: textEmb, styleTtl: style.ttl,
                latentMask: latentMask, textMask: mask,
                currentStep: MLXArray([Float(step)]), totalStep: total, cfgScale: cfg
            )
            eval(x)  // MLX.eval: forces lazy tensor evaluation (not code eval)
        }
        let wav = vocoder(x)[0]  // [samples]
        eval(wav)  // MLX.eval: forces lazy tensor evaluation (not code eval)
        return wav
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // Preset voices only in this slice; refAudio/refText are ignored.
        var voiceName = voice ?? "M1"
        if !Self.presetVoices.contains(voiceName) { voiceName = "M1" }
        let lang = SupertonicTextFrontend.availableLangs.contains(language ?? "en")
            ? (language ?? "en") : "en"
        return try synthesize(text: text, voice: voiceName, language: lang)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio,
                    refText: refText, language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}
