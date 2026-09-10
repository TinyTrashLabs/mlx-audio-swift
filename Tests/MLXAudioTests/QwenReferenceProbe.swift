import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Reference-clip probe (2026-09-09). One voice-store folder (`ref.wav` +
/// `meta.json` with `refText`) rendered N times with different seeds in the
/// phone configuration; per render: audio seconds, stop reason, frames,
/// level, and whether it is speech (> -35 dBFS) or hiss. Answers "does this
/// recording work with Qwen" with numbers. Environment: `QWEN_MOBILE_WEIGHTS`,
/// `QWEN_PROBE_VOICE_DIR` (required), `QWEN_PROBE_SEEDS` (default 1…6),
/// `QWEN_PROBE_TEXT`, `QWEN_PROBE_OUT` (WAVs).
final class QwenReferenceProbe: XCTestCase {
    static let weightsDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_MOBILE_WEIGHTS"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/scratch-mlx/Qwen3-TTS-12Hz-0.6B-Base-4bit-mobile")
    static let voiceDir = ProcessInfo.processInfo.environment["QWEN_PROBE_VOICE_DIR"].map(URL.init(fileURLWithPath:))
    static let seeds: [UInt64] = (ProcessInfo.processInfo.environment["QWEN_PROBE_SEEDS"] ?? "1,2,3,4,5,6")
        .split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }
    static let text = ProcessInfo.processInfo.environment["QWEN_PROBE_TEXT"]
        ?? "Type anything here and hear it read back in your own voice."
    static let outDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_PROBE_OUT"] ?? NSTemporaryDirectory())
        .appendingPathComponent("qwen-ref-probe", isDirectory: true)

    func testRenderTheReferenceAcrossSeeds() async throws {
        guard let voiceDir = Self.voiceDir else { throw XCTSkip("set QWEN_PROBE_VOICE_DIR") }
        guard FileManager.default.fileExists(atPath: Self.weightsDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("no mobile checkpoint at \(Self.weightsDir.path)")
        }
        MLX.Device.setDefault(device: Device(.gpu))
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSFusedStep.mode = .hybrid
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
        defer {
            Qwen3TTSModel.fastRope = false; Qwen3TTSModel.greedySubCodes = false; Qwen3TTSModel.fusedLayers = false
            Qwen3TTSModel.eosGreedyStop = false; Qwen3TTSModel.trailingSilenceStopSeconds = 0
        }
        let model = try await Qwen3TTSModel.fromModelDirectory(Self.weightsDir)
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: voiceDir.appendingPathComponent("meta.json"))) as? [String: Any]
        let refText = try XCTUnwrap(meta?["refText"] as? String)
        let (_, refAudio) = try loadAudioArray(from: voiceDir.appendingPathComponent("ref.wav"), sampleRate: model.sampleRate)
        eval(refAudio)   // materialise the lazy array; runs no model code
        try FileManager.default.createDirectory(at: Self.outDir, withIntermediateDirectories: true)
        let name = voiceDir.lastPathComponent
        var hiss = 0
        for seed in Self.seeds {
            MLXRandom.seed(seed)
            var samples: [Float] = []
            let stream = model.generateStream(
                text: Self.text, voice: nil, refAudio: refAudio, refText: refText, language: nil,
                generationParameters: model.defaultGenerationParameters, streamingInterval: 1.0)
            for try await event in stream {
                if case .audio(let chunk) = event, chunk.size > 1 { samples.append(contentsOf: chunk.asArray(Float.self)) }
            }
            Memory.clearCache()
            let sr = model.sampleRate
            let rms = sqrt(samples.reduce(0) { $0 + Double($1 * $1) } / Double(max(1, samples.count)))
            let dbfs = 20 * log10(max(rms, 1e-9))
            let secs = Double(samples.count) / Double(sr)
            let verdict = dbfs > -35 ? "speech" : "HISS"
            if dbfs <= -35 { hiss += 1 }
            let wav = Self.outDir.appendingPathComponent("\(name)-seed\(seed).wav")
            try QwenStopVarianceProbe.writeWav(samples, sampleRate: sr, to: wav)
            print(String(format: "[refprobe] %@ seed %d: %.1f s, %.1f dBFS, stop %@ after %d frames -> %@",
                         name, seed, secs, dbfs, Qwen3TTSModel.lastStopReason?.rawValue ?? "none",
                         Qwen3TTSModel.lastFrameCount, verdict))
        }
        print("[refprobe] \(name): \(hiss)/\(Self.seeds.count) renders were hiss")
    }
}
