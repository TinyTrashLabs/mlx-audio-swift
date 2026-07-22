// SuperTonic 3 Swift-port parity tests.
//
// Golden fixtures were dumped from the validated Python-MLX spike
// (supertonic-mlx-spike/dump_fixtures.py) as safetensors archives under
// Tests/Fixtures/supertonic/. Sub-model tolerance: 1e-4; e2e wav: 1e-3.
//
// The weight-dependent tests need the converted model directory (the four
// safetensors + unicode_indexer.json + voice_styles/). Set
// SUPERTONIC_MODEL_DIR, or place it at <repo>/weights-repo (default);
// otherwise those tests skip.

import Foundation
import MLX
import XCTest

@testable import MLXAudioTTS

final class SupertonicTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        // fp32 parity is unreachable with TF32 tensor-op GEMMs (~1e-2 error).
        // The bundled MLX core latches MLX_ENABLE_TF32 at the first fp32 GEMM.
        setenv("MLX_ENABLE_TF32", "0", 1)
    }

    // MARK: - Helpers

    static func fixturesDir() -> URL? {
        if let sub = Bundle.module.url(forResource: "supertonic", withExtension: nil, subdirectory: "Fixtures") {
            return sub
        }
        // Fallback: source-tree relative
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/supertonic")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static func modelDir() -> URL? {
        if let env = ProcessInfo.processInfo.environment["SUPERTONIC_MODEL_DIR"] {
            return URL(fileURLWithPath: env)
        }
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights-repo")
        return FileManager.default.fileExists(
            atPath: url.appendingPathComponent("vocoder.safetensors").path) ? url : nil
    }

    func fixture(_ name: String) throws -> [String: MLXArray] {
        guard let dir = Self.fixturesDir() else {
            throw XCTSkip("Supertonic fixtures not found")
        }
        return try loadArrays(url: dir.appendingPathComponent("\(name).safetensors"))
    }

    func weights(_ name: String) throws -> [String: MLXArray] {
        guard let dir = Self.modelDir() else {
            throw XCTSkip("Supertonic model dir not found (set SUPERTONIC_MODEL_DIR)")
        }
        return try loadArrays(url: dir.appendingPathComponent("\(name).safetensors"))
    }

    func requireModelDir() throws -> URL {
        guard let dir = Self.modelDir() else {
            throw XCTSkip("Supertonic model dir not found (set SUPERTONIC_MODEL_DIR)")
        }
        return dir
    }

    @discardableResult
    func assertParity(
        _ got: MLXArray, _ ref: MLXArray, tol: Float, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) -> Float {
        XCTAssertEqual(got.shape, ref.shape, "\(label): shape mismatch", file: file, line: line)
        guard got.shape == ref.shape else { return .infinity }
        let err = abs(got - ref).max().item(Float.self)
        XCTAssertLessThanOrEqual(err, tol, "\(label): max-abs-err \(err) > \(tol)", file: file, line: line)
        print("[parity] \(label): max-abs-err \(err)")
        return err
    }

    // MARK: - B6: Text frontend (no weights needed beyond indexer)

    func testFrontendEncodeHi() throws {
        let dir = try requireModelDir()
        let fe = try SupertonicTextFrontend(indexerURL: dir.appendingPathComponent("unicode_indexer.json"))
        // "<en>hi.</en>"
        XCTAssertEqual(try fe.encode("hi", lang: "en"),
                       [29, 64, 73, 31, 67, 68, 15, 29, 16, 64, 73, 31])
        XCTAssertEqual(try fe.preprocess("The quick brown fox jumps over the lazy dog."),
                       "<en>The quick brown fox jumps over the lazy dog.</en>")
        // trailing period auto-appended
        XCTAssertEqual(try fe.preprocess("hello world"), "<en>hello world.</en>")
        // NFKD decomposition: é -> e + combining acute (2 ids, no UNK leak)
        let eAcute = try fe.encode("é", lang: "fr")
        XCTAssertEqual(eAcute.count, "<fr>e\u{0301}.</fr>".unicodeScalars.count)
        XCTAssertFalse(eAcute.contains(-1))
        // invalid language throws
        XCTAssertThrowsError(try fe.encode("hi", lang: "xx"))
    }

    // MARK: - B1: Shared modules

    func testModulesParity() throws {
        let fx = try fixture("modules")
        let wDp = try weights("duration_predictor")
        let wTe = try weights("text_encoder")
        let wVe = try weights("vector_estimator")

        let mask = MLXArray.ones([1, 20, 1])
        let convnext = try SupertonicConvNeXtStack(wDp, "sentence_encoder.convnext.convnext", 6)
        assertParity(convnext(fx["convnext_x"]!, mask: mask), fx["convnext_out"]!, tol: 1e-4, "convnext")

        let vits = try SupertonicVITSEncoder(wDp, "sentence_encoder.attn_encoder", nLayers: 2, heads: 2)
        assertParity(vits(fx["vits_x"]!, mask: mask), fx["vits_out"]!, tol: 1e-4, "vits_encoder")

        let gst = try SupertonicGSTCrossAttention(wTe, "speech_prompted_text_encoder.attention1", heads: 2)
        let key = broadcast(wTe["style_encoder.style_token_layer.style_key"]!, to: [1, 50, 256])
        assertParity(
            gst(fx["gst_x"]!, keyIn: key, valIn: fx["gst_val"]!, qMask: mask),
            fx["gst_out"]!, tol: 1e-4, "gst_attention")

        let rot = try SupertonicRotaryCrossAttention(wVe, "vector_field.main_blocks.3.attn")
        assertParity(
            rot(fx["rot_x"]!, kv: fx["rot_kv"]!,
                qMask: MLXArray.ones([1, 12, 1]), kvMask: MLXArray.ones([1, 20, 1])),
            fx["rot_out"]!, tol: 1e-4, "rotary_attention")

        let time = try SupertonicTimeEncoder(wVe, "vector_field.time_encoder")
        assertParity(time(fx["time_t"]!), fx["time_out"]!, tol: 1e-4, "time_encoder")
    }

    // MARK: - B2: Duration predictor

    func testDurationPredictorParity() throws {
        let fx = try fixture("dp")
        let dp = try SupertonicDurationPredictor(try weights("duration_predictor"))
        assertParity(
            dp(fx["rand_ids"]!.asType(.int32), styleDp: fx["rand_style"]!, textMask: fx["rand_mask"]!),
            fx["rand_out"]!, tol: 1e-4, "dp_random")
        assertParity(
            dp(fx["m1_ids"]!.asType(.int32), styleDp: fx["m1_style"]!, textMask: fx["m1_mask"]!),
            fx["m1_out"]!, tol: 1e-4, "dp_m1")
    }

    // MARK: - B3: Text encoder

    func testTextEncoderParity() throws {
        let fx = try fixture("te")
        let te = try SupertonicTextEncoder(try weights("text_encoder"))
        assertParity(
            te(fx["rand_ids"]!.asType(.int32), styleTtl: fx["rand_ttl"]!, textMask: fx["rand_mask"]!),
            fx["rand_out"]!, tol: 1e-4, "te_random")
        assertParity(
            te(fx["m1_ids"]!.asType(.int32), styleTtl: fx["m1_ttl"]!, textMask: fx["m1_mask"]!),
            fx["m1_out"]!, tol: 1e-4, "te_m1")
    }

    // MARK: - B4: Vector field

    func testVectorFieldParity() throws {
        let fx = try fixture("ve")
        let vf = try SupertonicVectorField(try weights("vector_estimator"))
        assertParity(
            vf.eulerCFG(
                fx["rand_noisy"]!, textEmb: fx["rand_temb"]!, styleTtl: fx["rand_ttl"]!,
                latentMask: fx["rand_lmask"]!, textMask: fx["rand_tmask"]!,
                currentStep: fx["cur"]!, totalStep: fx["tot"]!),
            fx["rand_euler_out"]!, tol: 1e-4, "ve_euler_random")
        assertParity(
            vf.velocity(
                fx["rand_noisy"]!, textEmb: fx["rand_temb"]!, styleTtl: fx["rand_ttl"]!,
                latentMask: fx["rand_lmask"]!, textMask: fx["rand_tmask"]!,
                t: fx["cur"]! / fx["tot"]!),
            fx["rand_velocity_out"]!, tol: 1e-4, "ve_velocity_random")
        assertParity(
            vf.eulerCFG(
                fx["m1_noisy"]!, textEmb: fx["m1_temb"]!, styleTtl: fx["m1_ttl"]!,
                latentMask: fx["rand_lmask"]!, textMask: fx["m1_tmask"]!,
                currentStep: fx["cur"]!, totalStep: fx["tot"]!),
            fx["m1_euler_out"]!, tol: 1e-4, "ve_euler_m1")
    }

    // MARK: - B5: Vocoder

    func testVocoderParity() throws {
        let fx = try fixture("vocoder")
        let voc = try SupertonicVocoder(try weights("vocoder"))
        assertParity(voc(fx["rand_latent"]!), fx["rand_wav"]!, tol: 1e-4, "vocoder_random")
    }

    // MARK: - B7: Model registration + load

    func testLoadModelAndGenerate() async throws {
        let dir = try requireModelDir()
        let model = try await TTS.loadModel(modelRepo: dir.path)
        guard let supertonic = model as? SupertonicModel else {
            XCTFail("expected SupertonicModel, got \(type(of: model))")
            return
        }
        XCTAssertEqual(supertonic.sampleRate, 44100)
        let audio = try await supertonic.generate(
            text: "hi", voice: "M1", refAudio: nil, refText: nil, language: "en",
            generationParameters: supertonic.defaultGenerationParameters)
        XCTAssertGreaterThan(audio.dim(0), 0)
        let samples = audio.asArray(Float.self)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.01, "silent output")
    }

    // MARK: - B8: End-to-end parity vs the Python-MLX spike

    func testEndToEndParity() async throws {
        let dir = try requireModelDir()
        let model = try await SupertonicModel.fromModelDirectory(dir)
        let text = "The quick brown fox jumps over the lazy dog."
        for preset in ["M1", "F3"] {
            let fx = try fixture("e2e_\(preset)")
            // frontend must reproduce the exact spike token ids
            let ids = try model.frontend.encode(text, lang: "en")
            XCTAssertEqual(ids.map { Int32($0) }, fx["ids"]![0].asArray(Int32.self),
                           "\(preset): frontend ids mismatch")
            let wav = try model.synthesize(text: text, voice: preset, noise: fx["noise"]!)
            let err = assertParity(
                wav.reshaped([1, -1]), fx["wav"]!, tol: 1e-3, "e2e_\(preset)")
            print("[e2e] \(preset): swift-vs-python max-abs-err \(err)")
            try Self.writeWav(
                wav.asArray(Float.self), sampleRate: model.sampleRate,
                to: URL(fileURLWithPath: #filePath)
                    .deletingLastPathComponent().deletingLastPathComponent()
                    .appendingPathComponent("out_swift_\(preset).wav"))
        }
    }

    // MARK: - WAV writer (16-bit PCM mono)

    static func writeWav(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func put(_ s: String) { data.append(s.data(using: .ascii)!) }
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let pcm = samples.map { Int16(max(-32768, min(32767, Int(($0 * 32767).rounded(.towardZero))))) }
        put("RIFF"); put32(UInt32(36 + pcm.count * 2)); put("WAVE")
        put("fmt "); put32(16); put16(1); put16(1)
        put32(UInt32(sampleRate)); put32(UInt32(sampleRate * 2)); put16(2); put16(16)
        put("data"); put32(UInt32(pcm.count * 2))
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }
}
