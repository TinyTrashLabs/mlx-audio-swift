import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Stop-variance probe (2026-09-09). The phone renders the same 250-character
/// passage with the Jeff reference voice to anywhere between 11 and 25 s, and
/// the long renders sound wrong at the end. This renders it eight times on the
/// phone configuration (fused layers, fast RoPE, greedy sub-codes, 2 s
/// streaming chunks) with seeds 1…8 and records, per run: frames, audio
/// seconds, whether EOS or the token cap ended the loop, the frame at which
/// the trailing text ran out, where speech actually ends (last 0.5 s window
/// above -40 dBFS) and the log P(EOS) trajectory. WAVs land in
/// `$QWEN_STOP_PROBE_DIR` (default: the temp dir) for listening.
///
/// Needs the mobile checkpoint and the Jeff pack on disk; skips otherwise.
/// Environment overrides: `QWEN_MOBILE_WEIGHTS`, `QWEN_JEFF_VOICE_DIR`,
/// `QWEN_STOP_PROBE_DIR`, `QWEN_STOP_PROBE_SEEDS` (comma list).
final class QwenStopVarianceProbe: XCTestCase {
    static let passage = "Good evening, night owls. You're locked into the midnight frequency, where the coffee is strong and the tempo stays low. Tonight we start slow, with a record that sounds like rain on a tin roof, then climb toward something brighter as the hours pass."

    static let weightsDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_MOBILE_WEIGHTS"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/scratch-mlx/Qwen3-TTS-12Hz-0.6B-Base-4bit-mobile")
    static let voiceDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_JEFF_VOICE_DIR"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/Packs/jeff")
    static let outDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["QWEN_STOP_PROBE_DIR"]
        ?? NSTemporaryDirectory()).appendingPathComponent("qwen", isDirectory: true)
    static let seeds: [UInt64] = (ProcessInfo.processInfo.environment["QWEN_STOP_PROBE_SEEDS"] ?? "1,2,3,4,5,6,7,8")
        .split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }

    // Loaded once for the whole suite; the checkpoint takes a few seconds.
    nonisolated(unsafe) static var sharedModel: Qwen3TTSModel?
    nonisolated(unsafe) static var sharedRef: (audio: MLXArray, text: String)?

    struct Run {
        let seed: UInt64
        let frames: Int
        let audioSeconds: Double
        let stop: Qwen3TTSModel.StopReason
        let cap: Int
        let textExhaustedFrame: Int?
        let speechEndSeconds: Double
        let charsPerSecond: Double
        let firstEosAbove1pct: Int?
        let eosNearMisses: Int
        let eosLogProbTail: [Float]
        let wav: URL
    }

    override func setUp() async throws {
        try await super.setUp()
        guard FileManager.default.fileExists(atPath: Self.weightsDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("mobile Qwen3-TTS checkpoint not at \(Self.weightsDir.path)")
        }
        guard FileManager.default.fileExists(atPath: Self.voiceDir.appendingPathComponent("manifest.json").path) else {
            throw XCTSkip("Jeff voice pack not at \(Self.voiceDir.path)")
        }
        MLX.Device.setDefault(device: Device(.gpu))
        // The phone configuration.
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSFusedStep.mode = .hybrid
        Memory.cacheLimit = 512 * 1024 * 1024
        if Self.sharedModel == nil {
            let model = try await Qwen3TTSModel.fromModelDirectory(Self.weightsDir)
            let manifest = try JSONSerialization.jsonObject(
                with: Data(contentsOf: Self.voiceDir.appendingPathComponent("manifest.json"))) as? [String: Any]
            let source = ((manifest?["source"] as? [String: Any])?["base"] as? [String: Any])
            let refText = try XCTUnwrap(source?["text"] as? String)
            let refPath = try XCTUnwrap(source?["audio"] as? String)
            let (_, refAudio) = try loadAudioArray(
                from: Self.voiceDir.appendingPathComponent(refPath), sampleRate: model.sampleRate)
            eval(refAudio) // MLX eval(_:) materialises the lazy array; it runs no code.
            Self.sharedModel = model
            Self.sharedRef = (refAudio, refText)
        }
        try FileManager.default.createDirectory(at: Self.outDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        Qwen3TTSModel.fastRope = false
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.stopProbe = nil
        Qwen3TTSModel.eosLogitBias = 0
        Qwen3TTSModel.eosGreedyStop = false
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        super.tearDown()
    }

    // MARK: - One render

    func render(seed: UInt64, tag: String) async throws -> Run {
        let model = try XCTUnwrap(Self.sharedModel)
        let ref = try XCTUnwrap(Self.sharedRef)
        var eosLogProbs: [Float] = []
        Qwen3TTSModel.stopProbe = { _, lp, _ in eosLogProbs.append(lp) }
        MLXRandom.seed(seed)

        var samples: [Float] = []
        let stream = model.generateStream(
            text: Self.passage, voice: nil, refAudio: ref.audio, refText: ref.text, language: nil,
            generationParameters: model.defaultGenerationParameters, streamingInterval: 2.0)
        for try await event in stream {
            if case .audio(let chunk) = event, chunk.size > 1 {
                samples.append(contentsOf: chunk.asArray(Float.self))
            }
        }
        Qwen3TTSModel.stopProbe = nil

        let sr = model.sampleRate
        let wav = Self.outDir.appendingPathComponent("stop-probe-\(tag)\(seed).wav")
        try Self.writeWav(samples, sampleRate: sr, to: wav)

        let profile = Self.energyProfile(samples, sampleRate: sr, window: 0.5)
        let lastLoud = profile.lastIndex(where: { $0 > -40 })
        let speechEnd = lastLoud.map { Double($0 + 1) * 0.5 } ?? 0
        let audioSeconds = Double(samples.count) / Double(sr)
        return Run(
            seed: seed,
            frames: Qwen3TTSModel.lastFrameCount,
            audioSeconds: audioSeconds,
            stop: Qwen3TTSModel.lastStopReason ?? .maxTokens,
            cap: Qwen3TTSModel.lastEffectiveMaxTokens,
            textExhaustedFrame: Qwen3TTSModel.lastTextExhaustedFrame,
            speechEndSeconds: min(speechEnd, audioSeconds),
            charsPerSecond: speechEnd > 0 ? Double(Self.passage.count) / speechEnd : 0,
            firstEosAbove1pct: eosLogProbs.firstIndex(where: { $0 > log(0.01) }),
            eosNearMisses: eosLogProbs.dropLast().filter { $0 > log(0.3) }.count,
            eosLogProbTail: Array(eosLogProbs.suffix(12)),
            wav: wav
        )
    }

    func report(_ runs: [Run], title: String) {
        var lines = ["", "== \(title) ==",
                     "seed | frames | cap | stop | textEnd | audio s | speech end s | silence tail s | chars/s | 1st P(EOS)>1% frame | P(EOS)>30% misses | tail log P(EOS)"]
        for r in runs {
            let tail = r.eosLogProbTail.map { String(format: "%.1f", $0) }.joined(separator: " ")
            lines.append(String(
                format: "%4d | %6d | %3d | %@ | %@ | %7.2f | %12.2f | %14.2f | %7.2f | %@ | %3d | %@",
                Int(r.seed), r.frames, r.cap, r.stop == .eos ? "EOS " : (r.stop == .maxTokens ? "CAP " : "SIL "),
                r.textExhaustedFrame.map { String(format: "%7d", $0) } ?? "      -",
                r.audioSeconds, r.speechEndSeconds, r.audioSeconds - r.speechEndSeconds, r.charsPerSecond,
                r.firstEosAbove1pct.map { String(format: "%4d", $0) } ?? "   -", r.eosNearMisses, tail))
        }
        let secs = runs.map(\.audioSeconds)
        let ends = runs.map(\.speechEndSeconds)
        lines.append(String(format: "audio s: min %.2f max %.2f mean %.2f | speech end s: min %.2f max %.2f mean %.2f | EOS stops %d/%d",
                            secs.min() ?? 0, secs.max() ?? 0, secs.reduce(0, +) / Double(max(secs.count, 1)),
                            ends.min() ?? 0, ends.max() ?? 0, ends.reduce(0, +) / Double(max(ends.count, 1)),
                            runs.filter { $0.stop == .eos }.count, runs.count))
        lines.append("wavs: " + runs.map(\.wav.path).joined(separator: " "))
        let text = lines.joined(separator: "\n")
        print(text)
        try? (text + "\n").appendLine(to: Self.outDir.appendingPathComponent("stop-probe-report.txt"))
    }

    // MARK: - Tests

    func testBaselineEightSeeds() async throws {
        var runs: [Run] = []
        for seed in Self.seeds { runs.append(try await render(seed: seed, tag: "")) }
        report(runs, title: "baseline (phone configuration, defaults)")
    }

    /// Forces the sampler to miss the talker's exit (EOS logit -8) to see
    /// what the talker does after a missed EOS: recover within a few frames,
    /// or hiss to the token cap like the phone's 25.44 s render.
    func testForcedMissEightSeeds() async throws {
        Qwen3TTSModel.eosLogitBias = -8
        var runs: [Run] = []
        for seed in Self.seeds { runs.append(try await render(seed: seed, tag: "miss-")) }
        report(runs, title: "forced miss (eosLogitBias -8)")
    }

    /// Same forced miss, with the greedy-EOS stop: the run ends the first time
    /// EOS is the talker's top token, whatever the sampler drew.
    func testForcedMissWithGreedyStop() async throws {
        Qwen3TTSModel.eosLogitBias = -8
        Qwen3TTSModel.eosGreedyStop = true
        var runs: [Run] = []
        for seed in Self.seeds { runs.append(try await render(seed: seed, tag: "greedy-")) }
        report(runs, title: "forced miss + eosGreedyStop")
        for r in runs { XCTAssertEqual(r.stop, .eos, "seed \(r.seed)") }
    }

    /// Same forced miss, with the trailing-silence stop only (1.5 s below
    /// -35 dBFS after speech): bounds the tail whatever the talker's logits do.
    func testForcedMissWithSilenceStop() async throws {
        Qwen3TTSModel.eosLogitBias = -8
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
        var runs: [Run] = []
        for seed in Self.seeds { runs.append(try await render(seed: seed, tag: "silence-")) }
        report(runs, title: "forced miss + trailingSilenceStop 1.5 s")
        for r in runs {
            XCTAssertNotEqual(r.stop, .maxTokens, "seed \(r.seed) ran to the cap")
            XCTAssertLessThan(r.audioSeconds - r.speechEndSeconds, 4.0, "seed \(r.seed) tail")
        }
    }

    /// The proposed shipping configuration, on the unbiased sampler: both
    /// stops on, checking that ordinary renders are unaffected (still EOS,
    /// same length band as the baseline).
    func testBothStopsUnbiasedEightSeeds() async throws {
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
        var runs: [Run] = []
        for seed in Self.seeds { runs.append(try await render(seed: seed, tag: "fixed-")) }
        report(runs, title: "eosGreedyStop + trailingSilenceStop 1.5 s, unbiased")
        for r in runs {
            XCTAssertNotEqual(r.stop, .maxTokens, "seed \(r.seed) ran to the cap")
            XCTAssertLessThan(r.audioSeconds - r.speechEndSeconds, 2.5, "seed \(r.seed) tail")
        }
    }

    // MARK: - Helpers

    /// RMS per window in dBFS.
    static func energyProfile(_ samples: [Float], sampleRate: Int, window: Double) -> [Float] {
        let n = max(1, Int(Double(sampleRate) * window))
        var out: [Float] = []
        var i = 0
        while i < samples.count {
            let end = min(samples.count, i + n)
            var sum: Double = 0
            for j in i ..< end { sum += Double(samples[j]) * Double(samples[j]) }
            let rms = sqrt(sum / Double(end - i))
            out.append(rms > 0 ? Float(20 * log10(rms)) : -120)
            i = end
        }
        return out
    }

    /// Minimal 16-bit PCM mono WAV.
    static func writeWav(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let byteCount = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8)); put(UInt32(36 + byteCount))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); put(UInt32(16)); put(UInt16(1)); put(UInt16(1))
        put(UInt32(sampleRate)); put(UInt32(sampleRate * 2)); put(UInt16(2)); put(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); put(UInt32(byteCount))
        for s in samples { put(Int16(max(-1, min(1, s)) * 32767)) }
        try data.write(to: url)
    }
}

private extension String {
    func appendLine(to url: URL) throws {
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data(utf8))
        } else {
            try write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
