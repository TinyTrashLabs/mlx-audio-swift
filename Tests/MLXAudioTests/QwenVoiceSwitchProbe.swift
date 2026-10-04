import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Voice-switch probe (2026-09-09). On the phone, the second DISTINCT voice
/// rendered by one `Qwen3TTSModel` came out as 27 s of near-silence
/// (-55 dBFS), while the same voice rendered first in a fresh process was
/// fine. This renders A, then B, then A again on one model instance in the
/// phone configuration and reports each output's level, so the cause can be
/// bisected on the Mac (flip the statics in `configure()`).
///
/// Needs the mobile checkpoint, the Jeff pack and a second voice folder
/// (`ref.wav` + `meta.json` with `refText`, the iOS voice-store layout);
/// skips otherwise. Environment: `QWEN_MOBILE_WEIGHTS`, `QWEN_JEFF_VOICE_DIR`,
/// `QWEN_SECOND_VOICE_DIR`, `QWEN_VOICE_SWITCH_DIR` (WAV output).
final class QwenVoiceSwitchProbe: XCTestCase {
    static let passage = "The lighthouse keeper had not spoken to anyone in eleven days, and when the supply boat finally came, he found that his voice had gone strange and thin from disuse."

    static let weightsDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_MOBILE_WEIGHTS"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/scratch-mlx/Qwen3-TTS-12Hz-0.6B-Base-4bit-mobile")
    static let jeffDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_JEFF_VOICE_DIR"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/Packs/jeff")
    static let secondDir = ProcessInfo.processInfo.environment["QWEN_SECOND_VOICE_DIR"].map(URL.init(fileURLWithPath:))
    static let outDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_VOICE_SWITCH_DIR"]
        ?? NSTemporaryDirectory()).appendingPathComponent("qwen-switch", isDirectory: true)

    struct Voice {
        let name: String
        let audio: MLXArray
        let text: String
    }

    var model: Qwen3TTSModel!
    var jeff: Voice!
    var second: Voice!

    override func setUp() async throws {
        try await super.setUp()
        guard FileManager.default.fileExists(atPath: Self.weightsDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("mobile Qwen3-TTS checkpoint not at \(Self.weightsDir.path)")
        }
        guard FileManager.default.fileExists(atPath: Self.jeffDir.appendingPathComponent("manifest.json").path) else {
            throw XCTSkip("Jeff voice pack not at \(Self.jeffDir.path)")
        }
        guard let secondDir = Self.secondDir,
              FileManager.default.fileExists(atPath: secondDir.appendingPathComponent("meta.json").path) else {
            throw XCTSkip("set QWEN_SECOND_VOICE_DIR to a folder with ref.wav + meta.json")
        }
        MLX.Device.setDefault(device: Device(.gpu))
        configure()
        model = try await Qwen3TTSModel.fromModelDirectory(Self.weightsDir)

        // Jeff: pack layout (manifest.json -> source.base.{audio,text}).
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: Self.jeffDir.appendingPathComponent("manifest.json"))) as? [String: Any]
        let source = ((manifest?["source"] as? [String: Any])?["base"] as? [String: Any])
        jeff = try load(name: "jeff",
                        url: Self.jeffDir.appendingPathComponent(try XCTUnwrap(source?["audio"] as? String)),
                        text: try XCTUnwrap(source?["text"] as? String))
        // Second voice: iOS voice-store layout (meta.json.refText + ref.wav).
        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: secondDir.appendingPathComponent("meta.json"))) as? [String: Any]
        second = try load(name: (meta?["slug"] as? String) ?? "second",
                          url: secondDir.appendingPathComponent("ref.wav"),
                          text: try XCTUnwrap(meta?["refText"] as? String))
        try FileManager.default.createDirectory(at: Self.outDir, withIntermediateDirectories: true)
    }

    /// The phone configuration (QwenMLXEngine.loaded()).
    func configure() {
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSFusedStep.mode = .hybrid
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
        Memory.cacheLimit = 256 * 1024 * 1024
    }

    override func tearDown() {
        Qwen3TTSModel.fastRope = false
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.eosGreedyStop = false
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        model = nil
        super.tearDown()
    }

    private func load(name: String, url: URL, text: String) throws -> Voice {
        let (_, audio) = try loadAudioArray(from: url, sampleRate: model.sampleRate)
        eval(audio) // materialise the lazy array; runs no model code
        return Voice(name: name, audio: audio, text: text)
    }

    /// One render; returns (seconds, dBFS) and writes the WAV.
    @discardableResult
    func render(_ voice: Voice, tag: String) async throws -> (seconds: Double, dbfs: Double) {
        MLXRandom.seed(7)
        var samples: [Float] = []
        let stream = model.generateStream(
            text: Self.passage, voice: nil, refAudio: voice.audio, refText: voice.text, language: nil,
            generationParameters: model.defaultGenerationParameters, streamingInterval: 1.0)
        for try await event in stream {
            if case .audio(let chunk) = event, chunk.size > 1 {
                samples.append(contentsOf: chunk.asArray(Float.self))
            }
        }
        Memory.clearCache()
        let sr = model.sampleRate
        let wav = Self.outDir.appendingPathComponent("switch-\(tag)-\(voice.name).wav")
        try QwenStopVarianceProbe.writeWav(samples, sampleRate: sr, to: wav)
        let rms = sqrt(samples.reduce(0) { $0 + Double($1 * $1) } / Double(max(1, samples.count)))
        let dbfs = 20 * log10(max(rms, 1e-9))
        let secs = Double(samples.count) / Double(sr)
        print(String(format: "[switch] %@ %@: %.1f s, %.1f dBFS -> %@", tag, voice.name, secs, dbfs, wav.path))
        return (secs, dbfs)
    }

    func testSecondVoiceInOneProcessIsNotSilent() async throws {
        let a1 = try await render(jeff, tag: "1")
        let b = try await render(second, tag: "2")
        let a2 = try await render(jeff, tag: "3")
        XCTAssertGreaterThan(a1.dbfs, -30, "Jeff first render is silent")
        XCTAssertGreaterThan(b.dbfs, -30, "\(second.name) rendered second is silent (the phone bug)")
        XCTAssertGreaterThan(a2.dbfs, -30, "Jeff rendered third is silent")
    }
}
