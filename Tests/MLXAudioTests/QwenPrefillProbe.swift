import Foundation
import MLX
import MLXAudioCore
import XCTest

@testable import MLXAudioTTS

/// Part start, stage by stage (2026-10-05). On the phone every part of a read
/// spent 1.0-1.6 s between `generateStream` and its first frame. This times
/// what that span holds -- reference conditioning, the carry (`continuing`),
/// the prompt build, decoder priming, the talker prefill and frame 0 -- for a
/// fresh line with the reference and for a part that continues the previous
/// one, then the unprofiled time to the first frame in the app's
/// configuration, and checks that `fastPrefill` leaves a seeded render's codes
/// bit-identical. Run in release:
///
///   swift test -c release -Xswiftc -enable-testing --filter QwenPrefillProbe
///
/// Needs the mobile checkpoint and the Jeff pack (`QWEN_MOBILE_WEIGHTS`,
/// `QWEN_JEFF_VOICE_DIR`); skips otherwise.
final class QwenPrefillProbe: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let weightsDir = URL(fileURLWithPath: env["QWEN_MOBILE_WEIGHTS"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/scratch-mlx/Qwen3-TTS-12Hz-0.6B-Base-4bit-mobile")
    static let voiceDir = URL(fileURLWithPath: env["QWEN_JEFF_VOICE_DIR"]
        ?? "/Users/david/projects/gloam.fm/gloam-voice-studio-ios/Packs/jeff")
    static let runs = Int(env["QWEN_PREFILL_RUNS"] ?? "5") ?? 5

    static let first = "Good evening, night owls. The rain has been falling since noon, and the streets are quiet."
    static let second = "Then the wind came up off the river, and every window on the block began to sing along with it."

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
        Self.phoneConfiguration()
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

    /// The phone configuration (QwenMLXEngine.loaded()).
    static func phoneConfiguration() {
        Qwen3TTSModel.fastRope = true
        Qwen3TTSModel.greedySubCodes = true
        Qwen3TTSModel.fusedLayers = true
        Qwen3TTSModel.pipelineFrame = true
        Qwen3TTSModel.asyncDecode = 1
        Qwen3TTSModel.eosGreedyStop = true
        Qwen3TTSModel.trailingSilenceStopSeconds = 1.5
    }

    override func tearDown() {
        Qwen3TTSModel.greedySubCodes = false
        Qwen3TTSModel.fusedLayers = false
        Qwen3TTSModel.pipelineFrame = false
        Qwen3TTSModel.asyncDecode = 0
        Qwen3TTSModel.eosGreedyStop = false
        Qwen3TTSModel.trailingSilenceStopSeconds = 0
        Qwen3TTSModel.profileStart = false
        Qwen3TTSModel.fastPrefill = true
        super.tearDown()
    }

    struct Take {
        var codes: MLXArray?
        var firstToken: Double
        var firstAudio: Double
        var audio: [Float]
    }

    private func take(_ model: Qwen3TTSModel, _ text: String, _ c: Qwen3TTSModel.Qwen3TTSReferenceConditioning,
                      seed: UInt64, maxTokens: Int) async throws -> Take {
        var params = model.defaultGenerationParameters
        params.maxTokens = maxTokens
        let rng = MLXRandom.RandomState(seed: seed)
        let started = Date()
        var firstToken: Double?
        var firstAudio: Double?
        var audio: [Float] = []
        let stream = withRandomState(rng) {
            model.generateStream(text: text, conditioning: c, generationParameters: params, streamingInterval: 1.0)
        }
        for try await event in stream {
            switch event {
            case .token: if firstToken == nil { firstToken = Date().timeIntervalSince(started) }
            case .audio(let chunk) where chunk.size > 1:
                if firstAudio == nil { firstAudio = Date().timeIntervalSince(started) }
                audio += chunk.asArray(Float.self)
            default: break
            }
        }
        return Take(codes: Qwen3TTSModel.lastGeneratedCodes, firstToken: firstToken ?? 0, firstAudio: firstAudio ?? 0,
                    audio: audio)
    }

    private func conditionings() async throws
        -> (Qwen3TTSModel, fresh: Qwen3TTSModel.Qwen3TTSReferenceConditioning,
            carried: Qwen3TTSModel.Qwen3TTSReferenceConditioning) {
        let model = try XCTUnwrap(Self.sharedModel)
        let ref = try XCTUnwrap(Self.sharedRef)
        let base = try model.prepareReferenceConditioning(refAudio: ref.audio, refText: ref.text, language: "english")
        let part1 = try await take(model, Self.first, base, seed: 7, maxTokens: 400)
        let prev = try XCTUnwrap(part1.codes)
        let carried = try model.continuing(base, previousText: Self.first, previousCodes: prev)
        return (model, base, carried)
    }

    private static func ms(_ s: Double) -> String { String(format: "%.1f", s * 1000) }

    func testPartStartStageBreakdown() async throws {
        let (model, fresh, carried) = try await conditionings()
        let ref = try XCTUnwrap(Self.sharedRef)
        for fast in [false, true] {
            Qwen3TTSModel.fastPrefill = fast
            for (name, c) in [("fresh", fresh), ("carried", carried)] {
                let rows = model.promptRows(text: Self.second, conditioning: c)
                var stages: [String: Double] = [:]
                var condSeconds = 0.0, carrySeconds = 0.0
                // One warm-up, then `runs` measured.
                for run in 0 ... Self.runs {
                    var t = Date()
                    let base = try model.prepareReferenceConditioning(
                        refAudio: ref.audio, refText: ref.text, language: "english")
                    eval(base.referenceSpeechCodes)
                    let cs = Date().timeIntervalSince(t)
                    t = Date()
                    if name == "carried" {
                        let n = carried.referenceSpeechCodes.dim(2) - base.referenceSpeechCodes.dim(2)
                        let next = try model.continuing(base, previousText: Self.first,
                                                        previousCodes: carried.referenceSpeechCodes[0..., 0..., (-n)...])
                        eval(next.referenceSpeechCodes, next.referenceTextTokenIDs)
                    }
                    let ks = Date().timeIntervalSince(t)
                    Qwen3TTSModel.profileStart = true
                    _ = try await take(model, Self.second, c, seed: 11, maxTokens: 2)
                    Qwen3TTSModel.profileStart = false
                    guard run > 0 else { continue }
                    condSeconds += cs
                    carrySeconds += ks
                    for (stage, s) in Qwen3TTSModel.lastStartProfile { stages[stage, default: 0] += s }
                }
                let n = Double(Self.runs)
                let order = ["prompt", "prime", "prefill", "frame0"]
                let parts = order.map { "\($0) \(Self.ms((stages[$0] ?? 0) / n))" }.joined(separator: "  ")
                print("[prefill-probe] fast=\(fast) \(name) rows \(rows): conditioning \(Self.ms(condSeconds / n))  "
                      + "carry \(Self.ms(carrySeconds / n))  \(parts)  sum "
                      + Self.ms(order.reduce(0) { $0 + (stages[$1] ?? 0) } / n) + " ms")

                // Unprofiled, the app's overlap: time to the first frame and
                // to the first 1 s chunk of audio (12 frames).
                var first = 0.0, audio = 0.0
                for run in 0 ... Self.runs {
                    let tk = try await take(model, Self.second, c, seed: 11, maxTokens: 14)
                    if run > 0 { first += tk.firstToken; audio += tk.firstAudio }
                }
                print("[prefill-probe] fast=\(fast) \(name) rows \(rows): first frame \(Self.ms(first / n)) ms, "
                      + "first audio \(Self.ms(audio / n)) ms (unprofiled)")
            }
        }
    }

    /// Seeded renders: `fastPrefill` on and off give the same codes, fresh and carried.
    func testFastPrefillCodesMatchTheModulePath() async throws {
        let (model, fresh, carried) = try await conditionings()
        for (name, c) in [("fresh", fresh), ("carried", carried)] {
            Qwen3TTSModel.fastPrefill = false
            let oldTake = try await take(model, Self.second, c, seed: 23, maxTokens: 60)
            let old = try XCTUnwrap(oldTake.codes)
            Qwen3TTSModel.fastPrefill = true
            let newTake = try await take(model, Self.second, c, seed: 23, maxTokens: 60)
            let new = try XCTUnwrap(newTake.codes)
            XCTAssertEqual(old.shape, new.shape, name)
            let same = old.shape == new.shape && old.asArray(Int32.self) == new.asArray(Int32.self)
            print("[prefill-probe] parity \(name): \(old.dim(2)) frames, codes identical: \(same)")
            XCTAssertTrue(same, "\(name): codes differ between the module prefill and fastPrefill")
        }
    }

    /// Where decoder priming spends its time: the transformer over every
    /// context frame, or the upsampler/decoder tail.
    func testPrimeSplit() async throws {
        let (model, fresh, carried) = try await conditionings()
        let dec = try XCTUnwrap(model.speechTokenizer).decoder
        print("[prefill-probe] ref frames \(fresh.referenceSpeechCodes.dim(2)), ref text tokens "
              + "\(fresh.referenceTextTokenIDs.dim(1)), carried frames \(carried.referenceSpeechCodes.dim(2))")
        for (name, c) in [("fresh", fresh), ("carried", carried)] {
            let codes = c.referenceSpeechCodes
            var full = 0.0, tail1 = 0.0, last12 = 0.0
            for run in 0 ... Self.runs {
                dec.resetStreamingState(); var t = Date()
                dec.primeStreaming(codes); let a = Date().timeIntervalSince(t)
                dec.resetStreamingState(); t = Date()
                dec.primeStreaming(codes, tailFrames: 1); let b = Date().timeIntervalSince(t)
                dec.resetStreamingState(); t = Date()
                dec.primeStreaming(codes[0..., 0..., (-12)...]); let d = Date().timeIntervalSince(t)
                if run > 0 { full += a; tail1 += b; last12 += d }
            }
            let n = Double(Self.runs)
            print("[prefill-probe] prime \(name): full \(Self.ms(full / n))  tail=1 \(Self.ms(tail1 / n))  "
                  + "only last 12 frames \(Self.ms(last12 / n)) ms")
        }
        dec.resetStreamingState()
    }

    /// Per-layer wall time of a prime's decoder tail (eval at each layer).
    func testPrimeLayers() async throws {
        let (model, fresh, _) = try await conditionings()
        let dec = try XCTUnwrap(model.speechTokenizer).decoder
        let codes = fresh.referenceSpeechCodes
        for tail in [12, 1] {
            var acc: [String: Double] = [:]
            var order: [String] = []
            for run in 0 ... Self.runs {
                dec.resetStreamingState()
                var t = Date()
                func mark(_ k: String, _ x: MLXArray) {
                    eval(x)
                    let now = Date()
                    if run > 0 { if acc[k] == nil { order.append(k) }; acc[k, default: 0] += now.timeIntervalSince(t) }
                    t = now
                }
                var h = dec.quantizer.decode(codes); mark("quantizer", h)
                h = dec.preConv.step(h); mark("preConv", h)
                dec.transformerCache = dec.preTransformer.makeCache()
                h = dec.preTransformer(h.transposed(0, 2, 1), cache: dec.transformerCache).transposed(0, 2, 1)
                mark("transformer", h)
                let n = h.dim(2)
                h = h[0..., 0..., (n - tail)...]
                for (j, layer) in dec.upsample.enumerated() { h = layer.step(h); mark("upsample\(j)", h) }
                for (j, layer) in dec.decoder.enumerated() {
                    if let l = layer as? DecoderInitialConv { h = l.step(h) }
                    else if let l = layer as? DecoderBlock { h = l.step(h) }
                    else if let l = layer as? DecoderOutputSnake { h = l(h) }
                    else if let l = layer as? DecoderOutputConv { h = l.step(h) }
                    mark("decoder\(j) \(h.shape)", h)
                }
            }
            print("[prefill-probe] prime layers tail \(tail): "
                  + order.map { "\($0) \(Self.ms(acc[$0]! / Double(Self.runs)))" }.joined(separator: " | "))
        }
        dec.resetStreamingState()
    }

    /// Inside the decoder blocks: snake, transposed-conv upsample, residual units.
    func testDecoderBlockParts() async throws {
        let (model, _, _) = try await conditionings()
        let dec = try XCTUnwrap(model.speechTokenizer).decoder
        for tail in [1, 12] {
            var acc: [String: Double] = [:]
            var order: [String] = []
            for run in 0 ... Self.runs {
                dec.resetStreamingState()
                var h = MLXRandom.normal([1, 1536, 4 * tail]).asType(.float32)
                eval(h)
                var t = Date()
                func mark(_ k: String, _ x: MLXArray) {
                    eval(x)
                    let now = Date()
                    if run > 0 { if acc[k] == nil { order.append(k) }; acc[k, default: 0] += now.timeIntervalSince(t) }
                    t = now
                }
                for (j, layer) in dec.decoder.enumerated() {
                    guard let b = layer as? DecoderBlock else { continue }
                    h = (b.block[0] as! SnakeBeta)(h); mark("b\(j).snake", h)
                    h = (b.block[1] as! DecoderBlockUpsample).step(h); mark("b\(j).up\(h.shape)", h)
                    for (k, r) in b.block.dropFirst(2).enumerated() { h = (r as! DecoderResidualUnit).step(h); mark("b\(j).res\(k)", h) }
                }
            }
            print("[prefill-probe] blocks tail \(tail): "
                  + order.map { "\($0) \(Self.ms(acc[$0]! / Double(Self.runs)))" }.joined(separator: " | "))
        }
        let w = (dec.decoder[1] as! DecoderBlock).block[1] as! DecoderBlockUpsample
        print("[prefill-probe] upsample weight \(w.conv.weight.shape) \(w.conv.weight.dtype)")
        dec.resetStreamingState()
    }

    /// The trimmed prime leaves the decoder in the state the full 12-frame
    /// tail does, up to float reordering, and a restored snapshot decodes the
    /// next chunk bit-identically.
    func testTrimmedPrimeAndRestoreMatchTheFullTail() async throws {
        let (model, fresh, carried) = try await conditionings()
        let dec = try XCTUnwrap(model.speechTokenizer).decoder
        let next = carried.referenceSpeechCodes[0..., 0..., (-12)...]
        func flat(_ st: Qwen3TTSSpeechTokenizerDecoder.StreamingState) -> [MLXArray] {
            st.kv.flatMap { $0 } + st.buffers.compactMap { $0 }
        }
        for (name, c) in [("fresh", fresh), ("carried", carried)] {
            let codes = c.referenceSpeechCodes
            dec.resetStreamingState(); dec.primeStreaming(codes)
            let full = dec.streamingState()
            let fullChunk = dec.streamingStep(next); eval(fullChunk)
            dec.resetStreamingState(); dec.primeStreaming(codes, trimmed: true)
            let trimmed = dec.streamingState()
            let trimmedChunk = dec.streamingStep(next); eval(trimmedChunk)
            dec.resetStreamingState(); dec.restoreStreamingState(trimmed)
            let restoredChunk = dec.streamingStep(next); eval(restoredChunk)
            let (a, b) = (flat(full), flat(trimmed))
            XCTAssertEqual(a.count, b.count)
            var maxDiff: Float = 0
            var identical = a.count == b.count
            for (j, (x, y)) in zip(a, b).enumerated() {
                if x.shape != y.shape { identical = false; print("[prefill-probe]   #\(j) shape \(x.shape) vs \(y.shape)"); continue }
                let d = abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                if d > 0, Self.env["QWEN_PREFILL_VERBOSE"] != nil {
                    print("[prefill-probe]   #\(j) \(x.shape) \(x.dtype) diff \(d) max|x| \(abs(x).max().item(Float.self))")
                }
                maxDiff = max(maxDiff, abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self))
                if !arrayEqual(x, y).item(Bool.self) { identical = false }
            }
            let chunkDiff = abs(fullChunk - trimmedChunk).max().item(Float.self)
            let restoreSame = arrayEqual(trimmedChunk, restoredChunk).item(Bool.self)
            print("[prefill-probe] trimmed prime \(name): \(a.count) state arrays, identical \(identical), "
                  + "max diff \(maxDiff); next chunk max diff \(chunkDiff); restore identical \(restoreSame)")
            // Not bit-identical: shorter convs/matmuls round differently.
            // Bound it: the state within 1e-2, the next chunk below -60 dBFS.
            _ = identical
            XCTAssertLessThan(maxDiff, 1e-2, "\(name): trimmed prime state drifted")
            XCTAssertLessThan(chunkDiff, 1e-3, "\(name): next chunk audio drifted")
            XCTAssertTrue(restoreSame, "\(name)")
        }
        dec.resetStreamingState()
    }
}
