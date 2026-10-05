import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Join-context probe (2026-09-24). A stitched render streams each part
/// through a freshly reset decoder, and every part after the first opens
/// with a ~60 ms pop. `streamPrimeCodes` warms the decoder on the previous
/// part's tail; on the phone the first primed part came out as not-words.
///
/// Renders part A, then part B twice from the same seed -- cold and primed
/// with A's tail. The talker never sees the priming, so both B renders
/// share their codes; only the decoder differs. Reports each B's level
/// envelope and how closely the two track, and writes the WAVs to
/// `$QWEN_STOP_PROBE_DIR/qwen` for listening.
final class QwenJoinContextProbe: XCTestCase {
    static let partA = "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting each of the one hundred and twelve steps under his breath."
    static let partB = "At the top, he wiped the salt from the great lens, trimmed the wick, and waited for the last light to leave the sky."

    override func setUp() async throws {
        try await super.setUp()
        let probe = QwenStopVarianceProbe()
        try await probe.setUp()
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
    }

    override func tearDown() {
        Qwen3TTSModel.streamPrimeCodes = nil
        Qwen3TTSModel.fastRope = false
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.eosGreedyStop = false
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        super.tearDown()
    }

    func stream(_ text: String, seed: UInt64, prime: MLXArray?, trimRefSeconds: Double = 0) async throws -> (samples: [Float], tail: MLXArray?) {
        let model = try XCTUnwrap(QwenStopVarianceProbe.sharedModel)
        var ref = try XCTUnwrap(QwenStopVarianceProbe.sharedRef)
        if trimRefSeconds > 0 {
            ref.audio = ref.audio[0 ..< (ref.audio.dim(0) - Int(trimRefSeconds * Double(model.sampleRate)))]
            eval(ref.audio)
        }
        Qwen3TTSModel.streamPrimeCodes = prime
        defer { Qwen3TTSModel.streamPrimeCodes = nil }
        MLXRandom.seed(seed)
        var samples: [Float] = []
        for try await event in model.generateStream(
            text: text, voice: nil, refAudio: ref.audio, refText: ref.text, language: nil,
            generationParameters: model.defaultGenerationParameters, streamingInterval: 2.0) {
            if case .audio(let chunk) = event, chunk.size > 1 { samples.append(contentsOf: chunk.asArray(Float.self)) }
        }
        return (samples, Qwen3TTSModel.lastCodesTail)
    }

    func testPrimedPartMatchesColdPartAfterTheSeam() async throws {
        let model = try XCTUnwrap(QwenStopVarianceProbe.sharedModel)
        let sr = model.sampleRate
        let dir = QwenStopVarianceProbe.outDir
        for seed: UInt64 in [1, 2, 3] {
            let a = try await stream(Self.partA, seed: 100 + seed, prime: nil)
            let tailA = try XCTUnwrap(a.tail)
            let cold = try await stream(Self.partB, seed: seed, prime: nil)
            let coldTail = cold.tail.map { $0.asArray(Int32.self) }
            let primed = try await stream(Self.partB, seed: seed, prime: tailA)
            let primedTail = primed.tail.map { $0.asArray(Int32.self) }

            try QwenStopVarianceProbe.writeWav(a.samples, sampleRate: sr, to: dir.appendingPathComponent("join-A\(seed).wav"))
            try QwenStopVarianceProbe.writeWav(cold.samples, sampleRate: sr, to: dir.appendingPathComponent("join-B\(seed)-cold.wav"))
            try QwenStopVarianceProbe.writeWav(primed.samples, sampleRate: sr, to: dir.appendingPathComponent("join-B\(seed)-primed.wav"))

            let ec = QwenStopVarianceProbe.energyProfile(cold.samples, sampleRate: sr, window: 0.05)
            let ep = QwenStopVarianceProbe.energyProfile(primed.samples, sampleRate: sr, window: 0.05)
            let n = min(ec.count, ep.count)
            let meanAbsDiff = n > 10 ? zip(ec[10 ..< n], ep[10 ..< n]).map { abs($0 - $1) }.reduce(0, +) / Float(n - 10) : -1
            print(String(format: "seed %d | A tail frames %d | codes equal %@ | cold %.2f s primed %.2f s | mean |dB diff| after 0.5 s %.1f",
                         Int(seed), tailA.dim(2), coldTail == primedTail ? "yes" : "NO",
                         Double(cold.samples.count) / Double(sr), Double(primed.samples.count) / Double(sr), meanAbsDiff))
            print("  cold  :", ec.prefix(12).map { Int($0) }, "...", stride(from: 0, to: ec.count, by: 10).map { Int(ec[$0]) })
            print("  primed:", ep.prefix(12).map { Int($0) }, "...", stride(from: 0, to: ep.count, by: 10).map { Int(ep[$0]) })
        }
        print("wavs in", dir.path)
    }

    /// Every part of the phone's test script, seeds 1...8, cold -- WAVs only;
    /// scripts transcribe them to count parts that come out as no words.
    func testEveryPartEightSeeds() async throws {
        let model = try XCTUnwrap(QwenStopVarianceProbe.sharedModel)
        let parts = [Self.partA, Self.partB,
                     "Ships he would never meet passed in the dark, trusting a flame he tended alone.",
                     "In the winter the storms came sideways off the water, and the tower hummed like a struck bell.",
                     "He wrote the weather in a ledger no one read, because someone, someday, might need to know that the wind turned east at four."]
        for (p, text) in parts.enumerated() {
            for seed: UInt64 in 1...8 {
                let r = try await stream(text, seed: seed, prime: nil)
                try QwenStopVarianceProbe.writeWav(r.samples, sampleRate: model.sampleRate,
                    to: QwenStopVarianceProbe.outDir.appendingPathComponent("part\(p + 1)-s\(seed).wav"))
                print(String(format: "part %d seed %d: %.2f s, stop %@", p + 1, Int(seed),
                             Double(r.samples.count) / Double(model.sampleRate),
                             Qwen3TTSModel.lastStopReason?.rawValue ?? "none"))
            }
        }
    }

    /// Jeff's ref.wav ends on the onset of a word it cuts off (-66 dB, then
    /// -17 dB in the last 60 ms). Parts 2 and 3 again, the reference with
    /// its last 100 ms removed.
    func testTrimmedReference() async throws {
        let model = try XCTUnwrap(QwenStopVarianceProbe.sharedModel)
        let parts = [(2, Self.partB), (3, "Ships he would never meet passed in the dark, trusting a flame he tended alone.")]
        for (p, text) in parts {
            for seed: UInt64 in 1...8 {
                let r = try await stream(text, seed: seed, prime: nil, trimRefSeconds: 0.1)
                try QwenStopVarianceProbe.writeWav(r.samples, sampleRate: model.sampleRate,
                    to: QwenStopVarianceProbe.outDir.appendingPathComponent("trim-part\(p)-s\(seed).wav"))
                print(String(format: "trim part %d seed %d: %.2f s, stop %@", p, Int(seed),
                             Double(r.samples.count) / Double(model.sampleRate),
                             Qwen3TTSModel.lastStopReason?.rawValue ?? "none"))
            }
        }
    }

    /// Rate and clipping sweep for any reference (2026-09-24, Morgan's
    /// window: a slow speaker over a -35 dB bed). QWEN_REF_WAV +
    /// QWEN_REF_TEXT_FILE name the reference; prints per render the chars/s
    /// RenderCheck computes (text / whole audio) and the clipped fraction.
    func testVoiceSweep() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let wavPath = env["QWEN_REF_WAV"], let textPath = env["QWEN_REF_TEXT_FILE"] else {
            throw XCTSkip("set QWEN_REF_WAV and QWEN_REF_TEXT_FILE")
        }
        let model = try XCTUnwrap(QwenStopVarianceProbe.sharedModel)
        let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: wavPath), sampleRate: model.sampleRate)
        eval(audio)
        let refText = try String(contentsOfFile: textPath, encoding: .utf8)
        let texts = ["Type anything here and hear it read back in your own voice.",
                     "Type anything here and hear it read back in your own voice. I don't know if this will always work",
                     Self.partA, Self.partB,
                     "Ships he would never meet passed in the dark, trusting a flame he tended alone."]
        for (t, text) in texts.enumerated() {
            for seed: UInt64 in 1...6 {
                MLXRandom.seed(seed)
                var samples: [Float] = []
                for try await event in model.generateStream(
                    text: text, voice: nil, refAudio: audio, refText: refText, language: nil,
                    generationParameters: model.defaultGenerationParameters, streamingInterval: 2.0) {
                    if case .audio(let chunk) = event, chunk.size > 1 { samples.append(contentsOf: chunk.asArray(Float.self)) }
                }
                let secs = Double(samples.count) / Double(model.sampleRate)
                let clipped = Double(samples.filter { abs($0) >= 0.999 }.count) / Double(max(1, samples.count))
                try QwenStopVarianceProbe.writeWav(samples, sampleRate: model.sampleRate,
                    to: QwenStopVarianceProbe.outDir.appendingPathComponent("sweep-t\(t + 1)-s\(seed).wav"))
                print(String(format: "sweep text %d seed %d: %.2f s, %.1f chars/s, clipped %.4f, stop %@",
                             t + 1, Int(seed), secs, Double(text.count) / max(secs, 0.01), clipped,
                             Qwen3TTSModel.lastStopReason?.rawValue ?? "none"))
            }
        }
    }
}
