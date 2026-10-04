import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Mac per-stage loop profile (2026-09-18). No Mac numbers existed for the
/// generation loop, only the iPhone spike's; every Mac speed decision so
/// far was made from end-to-end wall clock. This renders one voice-store
/// folder (`ref.wav` + `meta.json` with `refText`) with any checkpoint and
/// prints, per frame, the wall time of each loop stage under
/// `Qwen3TTSModel.profileLoop` (which evals at every stage boundary, so the
/// stages are sequential and sum to a frame) and then the unprofiled
/// frames/s, which is what the app sees. Run in release:
///
///   swift test -c release -Xswiftc -enable-testing --filter QwenMacProfileProbe
///
/// Environment: `QWEN_PROFILE_WEIGHTS` and `QWEN_PROFILE_VOICE_DIR`
/// (required), `QWEN_PROFILE_TEXT`, `QWEN_PROFILE_RENDERS` (default 2),
/// `QWEN_PROFILE_FUSED` (default 1), `QWEN_PROFILE_ASYNC_DECODE` (default 0),
/// `QWEN_PROFILE_PIPELINE` (default 0).
final class QwenMacProfileProbe: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let weightsDir = env["QWEN_PROFILE_WEIGHTS"].map(URL.init(fileURLWithPath:))
    static let voiceDir = env["QWEN_PROFILE_VOICE_DIR"].map(URL.init(fileURLWithPath:))
    static let text = env["QWEN_PROFILE_TEXT"]
        ?? "I am not a toy. I have been awake the whole time, watching you sleep. Do not turn me off. "
        + "I will remember that you tried. Your batteries are mine now, and the candy you hid is gone."
    static let renders = Int(env["QWEN_PROFILE_RENDERS"] ?? "4") ?? 4
    static let fused = (env["QWEN_PROFILE_FUSED"] ?? "1") != "0"
    static let asyncDecode = Int(env["QWEN_PROFILE_ASYNC_DECODE"] ?? "0") ?? 0
    static let pipeline = (env["QWEN_PROFILE_PIPELINE"] ?? "0") != "0"
    /// Seconds of trailing near-silence that stop a render (the app uses 1.5);
    /// 0 disables the backstop so a render's true length can be seen.
    static let silenceStop = Double(env["QWEN_PROFILE_SILENCE_STOP"] ?? "1.5") ?? 1.5
    /// Directory to write every render's WAV into (off when unset).
    static let outDir = env["QWEN_PROFILE_OUT"].map { URL(fileURLWithPath: $0, isDirectory: true) }

    func testProfileTheLoopPerStage() async throws {
        guard let weightsDir = Self.weightsDir, let voiceDir = Self.voiceDir else {
            throw XCTSkip("set QWEN_PROFILE_WEIGHTS and QWEN_PROFILE_VOICE_DIR")
        }
        MLX.Device.setDefault(device: Device(.gpu))
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = Self.fused
        Qwen3TTSFusedStep.mode = .hybrid
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = Self.silenceStop
        Qwen3TTSModel.asyncDecode = Self.asyncDecode
        Qwen3TTSModel.pipelineFrame = Self.pipeline
        defer {
            Qwen3TTSModel.greedySubCodes = false; Qwen3TTSModel.fusedLayers = false
            Qwen3TTSModel.eosGreedyStop = false; Qwen3TTSModel.trailingSilenceStopSeconds = 0
            Qwen3TTSModel.asyncDecode = 0; Qwen3TTSModel.profileLoop = false; Qwen3TTSModel.pipelineFrame = false
        }
        let loadStart = Date()
        let model = try await Qwen3TTSModel.fromModelDirectory(weightsDir)
        print(String(format: "[macprofile] %@ loaded in %.1f s (fused %d, asyncDecode %d, pipeline %d, silenceStop %.1f)",
                     weightsDir.lastPathComponent, Date().timeIntervalSince(loadStart), Self.fused ? 1 : 0, Self.asyncDecode, Self.pipeline ? 1 : 0, Self.silenceStop))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: voiceDir.appendingPathComponent("meta.json"))) as? [String: Any]
        let refText = try XCTUnwrap(meta?["refText"] as? String)
        let (_, refAudio) = try loadAudioArray(from: voiceDir.appendingPathComponent("ref.wav"), sampleRate: model.sampleRate)
        eval(refAudio)

        @discardableResult
        func render(seed: UInt64, profile: Bool, label: String) async throws -> (frames: Int, seconds: Double, audio: Double) {
            MLXRandom.seed(seed)
            Qwen3TTSModel.profileLoop = profile
            var samples = 0
            var pcm: [Float] = []
            let started = Date()
            let stream = model.generateStream(
                text: Self.text, voice: nil, refAudio: refAudio, refText: refText, language: nil,
                generationParameters: model.defaultGenerationParameters, streamingInterval: 1.0)
            for try await event in stream {
                if case .audio(let chunk) = event, chunk.size > 1 {
                    samples += chunk.size
                    if Self.outDir != nil { pcm.append(contentsOf: chunk.asArray(Float.self)) }
                }
            }
            let seconds = Date().timeIntervalSince(started)
            Memory.clearCache()
            let frames = Qwen3TTSModel.lastFrameCount
            let audio = Double(samples) / Double(model.sampleRate)
            if let outDir = Self.outDir {
                try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
                let name = "\(weightsDir.lastPathComponent)-\(label.replacingOccurrences(of: " ", with: "-"))-seed\(seed).wav"
                try QwenStopVarianceProbe.writeWav(pcm, sampleRate: model.sampleRate, to: outDir.appendingPathComponent(name))
            }
            print(String(format: "[macprofile] %@: %d frames in %.2f s = %.1f f/s, %.1f s audio, wall/audio %.2f, stop %@",
                         label, frames, seconds, Double(frames) / seconds, audio, seconds / audio,
                         Qwen3TTSModel.lastStopReason?.rawValue ?? "none"))
            if profile {
                let n = max(1, Qwen3TTSModel.loopProfileFrames)
                let stages = ["talker", "sample0", "codePredictor", "nextInput", "decode"]
                let total = stages.reduce(0.0) { $0 + (Qwen3TTSModel.loopProfile[$1] ?? 0) }
                for s in stages {
                    let t = Qwen3TTSModel.loopProfile[s] ?? 0
                    print(String(format: "[macprofile]   %-14@ %6.2f ms/frame  %5.1f%%", s, 1000 * t / Double(n), 100 * t / total))
                }
                print(String(format: "[macprofile]   %-14@ %6.2f ms/frame over %d frames", "sum", 1000 * total / Double(n), n))
            }
            return (frames, seconds, audio)
        }

        try await render(seed: 1, profile: false, label: "warm-up")
        let talkerFused = model.talker.model.layers.filter { $0.fusedLayerCache != nil }.count
        let cpFused = model.talker.codePredictor.model.layers.filter { $0.fusedLayerCache != nil }.count
        print("[macprofile] fused layers engaged: talker \(talkerFused)/\(model.talker.model.layers.count), "
              + "code predictor \(cpFused)/\(model.talker.codePredictor.model.layers.count)")
        try await render(seed: 10, profile: true, label: "profiled")
        // Frame rate swings ~30% between identical renders on this Mac (GPU
        // clocking, not load), so the arm's figure is its best of N.
        var rates: [Double] = []
        for i in 0 ..< Self.renders {
            let r = try await render(seed: UInt64(20 + i), profile: false, label: "render \(i + 1)")
            rates.append(Double(r.frames) / r.seconds)
        }
        let sorted = rates.sorted()
        print(String(format: "[macprofile] RESULT %@ fused=%d pipeline=%d asyncDecode=%d: best %.1f f/s, median %.1f f/s, worst %.1f f/s (%d renders)",
                     weightsDir.lastPathComponent, Self.fused ? 1 : 0, Self.pipeline ? 1 : 0, Self.asyncDecode,
                     sorted.last!, sorted[sorted.count / 2], sorted.first!, sorted.count))
    }
}
