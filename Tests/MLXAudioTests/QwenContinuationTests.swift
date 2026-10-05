import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// One read, many parts (2026-10-04): `continuing(_:previousText:previousCodes:)`
/// appends the previous part's transcript and codec frames after the voice
/// reference, `lastGeneratedCodes` hands a take's frames back, and a
/// `RandomState` scoped around `generateStream` gives one sampler stream for
/// the whole read. Mirrors gloam-voice-studio's `QwenTalkSession` (#93) on MLX.
///
/// Needs the mobile checkpoint and the Jeff pack on disk; skips otherwise
/// (`QWEN_MOBILE_WEIGHTS`, `QWEN_JEFF_VOICE_DIR`). The A/B probe at the end
/// runs only with `QWEN_CARRY_PROBE=1` (WAVs in `$QWEN_CARRY_PROBE_DIR`).
final class QwenContinuationTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let weightsDir = URL(fileURLWithPath: env["QWEN_MOBILE_WEIGHTS"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/scratch-mlx/Qwen3-TTS-12Hz-0.6B-Base-4bit-mobile")
    static let voiceDir = URL(fileURLWithPath: env["QWEN_JEFF_VOICE_DIR"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/Packs/jeff")

    nonisolated(unsafe) static var sharedModel: Qwen3TTSModel?
    nonisolated(unsafe) static var sharedRef: (audio: MLXArray, text: String)?

    override func setUp() async throws {
        try await super.setUp()
        guard FileManager.default.fileExists(atPath: Self.weightsDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("mobile Qwen3-TTS checkpoint not at \(Self.weightsDir.path)")
        }
        guard FileManager.default.fileExists(atPath: Self.voiceDir.appendingPathComponent("manifest.json").path) else {
            throw XCTSkip("Jeff voice pack not at \(Self.voiceDir.path)")
        }
        MLX.Device.setDefault(device: Device(.gpu))
        // The phone configuration (QwenMLXEngine.loaded()).
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSModel.pipelineFrame = true
        Qwen3TTSModel.asyncDecode = 1
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
        Memory.cacheLimit = 256 * 1024 * 1024
        if Self.sharedModel == nil {
            let model = try await Qwen3TTSModel.fromModelDirectory(Self.weightsDir)
            let manifest = try JSONSerialization.jsonObject(
                with: Data(contentsOf: Self.voiceDir.appendingPathComponent("manifest.json"))) as? [String: Any]
            let source = ((manifest?["source"] as? [String: Any])?["base"] as? [String: Any])
            let refText = try XCTUnwrap(source?["text"] as? String)
            let refPath = try XCTUnwrap(source?["audio"] as? String)
            let (_, refAudio) = try loadAudioArray(
                from: Self.voiceDir.appendingPathComponent(refPath), sampleRate: model.sampleRate)
            eval(refAudio)
            Self.sharedModel = model
            Self.sharedRef = (refAudio, refText)
        }
    }

    override func tearDown() {
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.pipelineFrame = false
        Qwen3TTSModel.asyncDecode = 0
        Qwen3TTSModel.eosGreedyStop = false
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        super.tearDown()
    }

    private func base() throws -> (Qwen3TTSModel, Qwen3TTSModel.Qwen3TTSReferenceConditioning) {
        let model = try XCTUnwrap(Self.sharedModel)
        let ref = try XCTUnwrap(Self.sharedRef)
        return (model, try model.prepareReferenceConditioning(refAudio: ref.audio, refText: ref.text, language: "english"))
    }

    struct Take {
        var samples: [Float]
        var codes: MLXArray?
        var frames: Int
        var stop: Qwen3TTSModel.StopReason?
        var firstTokenSeconds: Double
        var seconds: Double
    }

    private func take(_ model: Qwen3TTSModel, _ text: String, _ c: Qwen3TTSModel.Qwen3TTSReferenceConditioning,
                      rng: MLXRandom.RandomState) async throws -> Take {
        let started = Date()
        var firstToken: Double?
        var samples: [Float] = []
        // The stream's task is created inside the scope, so it inherits the
        // task-local random state: every sample of this take comes from `rng`.
        let stream = withRandomState(rng) {
            model.generateStream(text: text, conditioning: c, generationParameters: model.defaultGenerationParameters,
                                 streamingInterval: 1.0)
        }
        for try await event in stream {
            switch event {
            case .token: if firstToken == nil { firstToken = Date().timeIntervalSince(started) }
            case .audio(let chunk): if chunk.size > 1 { samples += chunk.asArray(Float.self) }
            default: break
            }
        }
        return Take(samples: samples, codes: Qwen3TTSModel.lastGeneratedCodes, frames: Qwen3TTSModel.lastFrameCount,
                    stop: Qwen3TTSModel.lastStopReason, firstTokenSeconds: firstToken ?? 0,
                    seconds: Date().timeIntervalSince(started))
    }

    // MARK: - API

    func testPromptRowsCountWhatThePrefillHolds() throws {
        let (model, c) = try base()
        let line = "Then the rain came, and nobody minded."
        XCTAssertEqual(model.promptRows(text: line, conditioning: c),
                       try model.prepareICLGenerationInputs(text: line, conditioning: c).0.dim(1))

        let prevCodes = MLXArray.zeros([1, c.referenceSpeechCodes.dim(1), 20], type: Int32.self)
        let next = try model.continuing(c, previousText: "Good evening, night owls.", previousCodes: prevCodes)
        XCTAssertEqual(model.promptRows(text: line, conditioning: next),
                       try model.prepareICLGenerationInputs(text: line, conditioning: next).0.dim(1))
    }

    func testFrameBudgetFollowsTheLineNotTheContext() throws {
        let (model, _) = try base()
        let line = String(repeating: "A slow and careful sentence, read with long pauses. ", count: 3)
        let six = model.frameBudget(text: line, maxTokens: 4096)
        XCTAssertGreaterThan(six, 75)
        XCTAssertEqual(six % 6, 0)
        Qwen3TTSModel.framesPerTextToken = 8
        defer { Qwen3TTSModel.framesPerTextToken = 6 }
        XCTAssertEqual(model.frameBudget(text: line, maxTokens: 4096), six / 6 * 8)
        XCTAssertEqual(model.frameBudget(text: line, maxTokens: 100), 100)
        XCTAssertEqual(model.frameBudget(text: "Hi.", maxTokens: 4096), 75)
    }

    func testContinuingAppendsThePreviousPartAfterTheReference() throws {
        let (model, c) = try base()
        let groups = c.referenceSpeechCodes.dim(1)
        let prevCodes = MLXArray(Array(repeating: Int32(7), count: groups * 5)).reshaped(1, groups, 5)
        let next = try model.continuing(c, previousText: "Good evening.", previousCodes: prevCodes)

        let refFrames = c.referenceSpeechCodes.dim(2)
        XCTAssertEqual(next.referenceSpeechCodes.dim(2), refFrames + 5)
        XCTAssertEqual(next.referenceSpeechCodes[0..., 0..., refFrames...].asArray(Int32.self), prevCodes.asArray(Int32.self))
        XCTAssertEqual(next.referenceSpeechCodes[0..., 0..., ..<refFrames].asArray(Int32.self),
                       c.referenceSpeechCodes.asArray(Int32.self))
        let refIds = c.referenceTextTokenIDs.asArray(Int32.self)
        let ids = next.referenceTextTokenIDs.asArray(Int32.self)
        XCTAssertEqual(Array(ids.prefix(refIds.count)), refIds)
        XCTAssertGreaterThan(ids.count, refIds.count)
        XCTAssertEqual(next.resolvedLanguage, c.resolvedLanguage)
        XCTAssertEqual(next.codecLanguageID, c.codecLanguageID)

        XCTAssertThrowsError(try model.continuing(c, previousText: "x", previousCodes: MLXArray.zeros([1, 3, 4], type: Int32.self)))
    }

    func testATakeHandsBackItsFramesAndOneSeedReproducesTheRead() async throws {
        let (model, c) = try base()
        let parts = ["Good evening, night owls.", "The coffee is strong tonight."]
        func read(seed: UInt64) async throws -> [[Int32]] {
            let rng = MLXRandom.RandomState(seed: seed)
            var cond = c
            var out: [[Int32]] = []
            for part in parts {
                let t = try await take(model, part, cond, rng: rng)
                let codes = try XCTUnwrap(t.codes)
                XCTAssertEqual(codes.shape, [1, c.referenceSpeechCodes.dim(1), t.frames])
                XCTAssertEqual(codes.dtype, .int32)
                out.append(codes.asArray(Int32.self))
                cond = try model.continuing(c, previousText: part, previousCodes: codes)
            }
            return out
        }
        let a = try await read(seed: 42)
        let b = try await read(seed: 42)
        XCTAssertEqual(a, b, "one RandomState per read must make the read reproducible")
        // A cancelled or not-yet-run generation reports no frames.
        XCTAssertNotNil(Qwen3TTSModel.lastGeneratedCodes)
    }

    // MARK: - A/B probe (QWEN_CARRY_PROBE=1)

    /// Renders a four-part read with and without the carry, several seeds
    /// each, and prints per seam: the level jump (dB, the last second of
    /// speech before the seam vs the first after) and the pitch jump
    /// (semitones, median F0 on each side), plus RTF and time to first frame
    /// (prefill) per part.
    func testCarryProbe() async throws {
        guard Self.env["QWEN_CARRY_PROBE"] == "1" else { throw XCTSkip("set QWEN_CARRY_PROBE=1") }
        let (model, c) = try base()
        let parts = [
            "Good evening, night owls. You're locked into the midnight frequency, where the coffee is strong and the tempo stays low.",
            "Tonight we start slow, with a record that sounds like rain on a tin roof, then climb toward something brighter.",
            "Before that, a word about the weather: fog on the coast until morning, and a cold wind coming down off the hills.",
            "So pull the blanket a little closer, turn the dial a little louder, and stay with me until the sun comes up.",
        ]
        let seeds = (Self.env["QWEN_CARRY_PROBE_SEEDS"] ?? "1,2,3").split(separator: ",").compactMap { UInt64($0) }
        let outDir = URL(fileURLWithPath: Self.env["QWEN_CARRY_PROBE_DIR"] ?? NSTemporaryDirectory())
            .appendingPathComponent("qwen-carry", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let sr = model.sampleRate
        _ = try await take(model, "Ready.", c, rng: MLXRandom.RandomState(seed: 0)) // warm

        var report: [String] = ["", "mode | seed | seam level jumps dB | seam pitch jumps st | read spread | part RTF | first frame s | prompt rows | stops"]
        var summary: [String: (level: [Double], pitch: [Double], rtf: [Double], ff: [Double],
                                f0Spread: [Double], dbSpread: [Double], redraws: Int)] = [:]
        for carry in [false, true] {
            let mode = carry ? "carry" : "plain"
            for seed in seeds {
                let rng = MLXRandom.RandomState(seed: seed)
                var cond = c
                var takes: [Take] = []
                var rows: [Int] = []
                var redraws = 0
                for part in parts {
                    rows.append(model.promptRows(text: part, conditioning: cond))
                    // As the app does when not streaming: redraw a derailed take
                    // (cap hit, or > 2 s of silence inside the line) up to twice.
                    var t = try await take(model, part, cond, rng: rng)
                    var draws = 1
                    while Self.derailed(t, sr: sr), draws <= 2 {
                        t = try await take(model, part, cond, rng: rng); draws += 1; redraws += 1
                    }
                    takes.append(t)
                    if carry, let codes = t.codes, !Self.derailed(t, sr: sr) {
                        cond = try model.continuing(c, previousText: part, previousCodes: codes)
                    } else {
                        cond = c
                    }
                }
                var levels: [Double] = [], pitches: [Double] = []
                for i in 0 ..< takes.count - 1 {
                    let a = Self.speechEdge(takes[i].samples, sr: sr, atEnd: true)
                    let b = Self.speechEdge(takes[i + 1].samples, sr: sr, atEnd: false)
                    levels.append(abs(Self.db(a) - Self.db(b)))
                    if let fa = Self.medianF0(a, sr: sr), let fb = Self.medianF0(b, sr: sr) {
                        pitches.append(abs(12 * log2(fb / fa)))
                    }
                }
                // Consistency across the read: spread of each part's median F0
                // (semitones) and speech level (dB).
                let partF0 = takes.compactMap { Self.medianF0(Self.speechOnly($0.samples, sr: sr), sr: sr) }
                let partDb = takes.map { Self.db(Self.speechOnly($0.samples, sr: sr)) }
                let f0Spread = Self.std(partF0.map { 12 * log2($0 / partF0[0]) })
                let dbSpread = Self.std(partDb)
                let rtf = takes.map { $0.seconds / max(0.01, Double($0.samples.count) / Double(sr)) }
                let ff = takes.map(\.firstTokenSeconds)
                var s = summary[mode] ?? ([], [], [], [], [], [], 0)
                s.level += levels; s.pitch += pitches; s.rtf += rtf; s.ff += ff
                s.f0Spread.append(f0Spread); s.dbSpread.append(dbSpread); s.redraws += redraws
                summary[mode] = s
                func f(_ xs: [Double]) -> String { xs.map { String(format: "%.2f", $0) }.joined(separator: " ") }
                report.append("\(mode) | \(seed) | \(f(levels)) | \(f(pitches)) | F0 spread \(f([f0Spread])) st, level spread \(f([dbSpread])) dB, redraws \(redraws) | \(f(rtf)) | \(f(ff)) | \(rows.map(String.init).joined(separator: " ")) | \(takes.map { $0.stop?.rawValue ?? "-" }.joined(separator: " "))")
                let gap = [Float](repeating: 0, count: Int(0.3 * Double(sr)))
                var joined: [Float] = []
                for (i, t) in takes.enumerated() { joined += (i > 0 ? gap : []) + t.samples }
                try Self.writeWav(joined, sampleRate: sr, to: outDir.appendingPathComponent("\(mode)-seed\(seed).wav"))
            }
        }
        func mean(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
        for mode in ["plain", "carry"] {
            let s = summary[mode]!
            report.append(String(format: "%@ mean: seam level jump %.2f dB, seam pitch jump %.2f st, F0 spread %.2f st, level spread %.2f dB, redraws %d, RTF %.3f, first frame %.3f s (parts 2+: %.3f s)",
                                 mode, mean(s.level), mean(s.pitch), mean(s.f0Spread), mean(s.dbSpread), s.redraws, mean(s.rtf), mean(s.ff),
                                 mean(s.ff.enumerated().filter { $0.offset % parts.count != 0 }.map(\.element))))
        }
        report.append("WAVs: \(outDir.path)")
        print(report.joined(separator: "\n"))
    }

    // MARK: - Signal helpers

    /// The app's derail rule (QwenTalkSession.derailed): the cap was hit, or a
    /// pause of more than 2 s sits between two stretches of speech.
    static func derailed(_ t: Take, sr: Int) -> Bool {
        t.stop == .maxTokens || longestInnerPause(t.samples, sr: sr) > 2
    }

    static func longestInnerPause(_ x: [Float], sr: Int) -> Double {
        let w = sr / 50
        let loud = (0 ..< x.count / w).map { i in db(Array(x[i * w ..< (i + 1) * w])) > -40 }
        guard let first = loud.firstIndex(of: true), let last = loud.lastIndex(of: true) else { return 0 }
        var longest = 0, run = 0
        for i in first ... last { if loud[i] { run = 0 } else { run += 1; longest = max(longest, run) } }
        return Double(longest) * 0.02
    }

    static func speechOnly(_ x: [Float], sr: Int) -> [Float] {
        let w = sr / 50
        return (0 ..< x.count / w).flatMap { i -> [Float] in
            let f = Array(x[i * w ..< (i + 1) * w]); return db(f) > -40 ? f : []
        }
    }

    static func std(_ xs: [Double]) -> Double {
        guard xs.count > 1 else { return 0 }
        let m = xs.reduce(0, +) / Double(xs.count)
        return (xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(xs.count - 1)).squareRoot()
    }

    static func db(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return -120 }
        let ms = x.reduce(0.0) { $0 + Double($1 * $1) } / Double(x.count)
        return 10 * log10(max(ms, 1e-12))
    }

    /// The first (or last) second of speech: 20 ms windows above -40 dBFS.
    static func speechEdge(_ x: [Float], sr: Int, atEnd: Bool) -> [Float] {
        let w = sr / 50
        let n = x.count / w
        let loud = (0 ..< n).filter { i in db(Array(x[i * w ..< (i + 1) * w])) > -40 }
        guard let first = loud.first, let last = loud.last else { return [] }
        let span = sr / w // windows per second
        let lo = atEnd ? max(first, last - span + 1) : first
        let hi = atEnd ? last : min(last, first + span - 1)
        return Array(x[lo * w ..< (hi + 1) * w])
    }

    /// Median F0 (Hz) over voiced 40 ms frames, by normalised autocorrelation in 70-350 Hz.
    static func medianF0(_ x: [Float], sr: Int) -> Double? {
        let w = sr / 25
        var f0s: [Double] = []
        var i = 0
        while i + w <= x.count {
            let f = Array(x[i ..< i + w])
            i += w / 2
            guard db(f) > -40 else { continue }
            let e0 = f.reduce(0.0) { $0 + Double($1 * $1) }
            var best = 0.0, lag = 0
            for l in (sr / 350) ... (sr / 70) where l < w {
                var acc = 0.0
                for k in 0 ..< (w - l) { acc += Double(f[k] * f[k + l]) }
                let r = acc / e0
                if r > best { best = r; lag = l }
            }
            if best > 0.5, lag > 0 { f0s.append(Double(sr) / Double(lag)) }
        }
        guard !f0s.isEmpty else { return nil }
        return f0s.sorted()[f0s.count / 2]
    }

    static func writeWav(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(bytes)
        for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
        try data.write(to: url)
    }
}
