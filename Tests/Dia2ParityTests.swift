import Foundation
import MLX
import XCTest
@testable import MLXAudioTTS

/// Parity against Tools/dia2-parity-dump.py. Skips when the fixture or the
/// weights are absent, so a clean checkout still passes.
final class Dia2ParityTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        setenv("MLX_ENABLE_TF32", "0", 1)
    }

    static func fixture() throws -> [String: MLXArray] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/dia2/parity-1b.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("run Tools/dia2-parity-dump.py first")
        }
        return try loadArrays(url: url)
    }

    /// Directory holding Dia2-1B's config.json + model.safetensors. Any
    /// precision: loading and generating must work for every tier we publish.
    static func modelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["DIA2_MODEL_DIR"] else {
            throw XCTSkip("set DIA2_MODEL_DIR to a Dia2-1B checkout")
        }
        return URL(fileURLWithPath: path)
    }

    /// The same directory, but only when the weights are float32 — the
    /// precision Tools/dia2-parity-dump.py dumped the fixture at.
    ///
    /// Element-wise parity against a torch dump is a statement about the PORT,
    /// not about a checkpoint's precision. bf16 misses the 2e-3 tolerance by
    /// an order of magnitude (0.03 on hidden states) and 8-bit by more, for
    /// reasons that have nothing to do with this code. Running it anyway
    /// produced 271 failures that read exactly like a broken port.
    static func fullPrecisionModelDir() throws -> URL {
        let dir = try modelDir()
        let weights = try loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        guard weights.values.allSatisfy({ $0.dtype == .float32 }) else {
            throw XCTSkip(
                "parity needs the float32 checkpoint the fixture was dumped from; "
                    + "DIA2_MODEL_DIR holds converted weights")
        }
        return dir
    }

    func assertParity(_ got: MLXArray, _ ref: MLXArray, tol: Float, _ label: String,
                      file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.shape, ref.shape, "\(label): shape", file: file, line: line)
        guard got.shape == ref.shape else { return }
        let err = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(err, tol, "\(label): max-abs-err \(err)", file: file, line: line)
    }

    func testTransformerMatchesTheReference() throws {
        let fx = try Self.fixture()
        let dir = try Self.fullPrecisionModelDir()
        let config = try JSONDecoder().decode(
            Dia2Config.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let model = Dia2Transformer(config: config)
        try model.loadWeights(from: dir)

        let tokens = fx["tokens"]!
        let cache = model.makeCache()
        for t in 0 ..< 8 {
            let step = tokens[0..., 0..., t ..< (t + 1)]
            let pos = MLXArray([Int32(t)]).reshaped([1, 1])
            let out = model.step(step, positions: pos, cache: cache)
            assertParity(out.action, fx["action_\(t)"]!, tol: 2e-3, "action t=\(t)")
            assertParity(out.cb0, fx["cb0_\(t)"]!, tol: 2e-3, "cb0 t=\(t)")
            assertParity(out.hidden, fx["hidden_\(t)"]!, tol: 2e-3, "hidden t=\(t)")
        }
    }
    func testDepformerMatchesTheReference() throws {
        let fx = try Self.fixture()
        let dir = try Self.fullPrecisionModelDir()
        let config = try JSONDecoder().decode(
            Dia2Config.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))

        let transformer = Dia2Transformer(config: config)
        try transformer.loadWeights(from: dir)
        let depformer = Dia2Depformer(config: config)
        try depformer.loadWeights(from: dir)

        let tokens = fx["tokens"]!
        let channels = config.data.channels
        let depCache = depformer.makeCache()

        for t in 0 ..< 8 {
            // The reference feeds its own hidden state in; use the fixture's so
            // a transformer regression cannot masquerade as a depformer pass.
            let hidden = fx["hidden_\(t)"]!
            depformer.resetCache(depCache)
            var prev = tokens[0..., 2, t]
            for stage in 0 ..< depformer.numDepth {
                let logits = depformer.step(
                    stage: stage, prevAudio: prev, hidden: hidden, cache: depCache,
                    mainText: stage == 0 ? tokens[0..., 0, t] : nil,
                    secondText: stage == 0 ? tokens[0..., 1, t] : nil)
                assertParity(logits, fx["dep_\(t)_\(stage)"]!, tol: 2e-3, "dep t=\(t) stage=\(stage)")
                prev = (3 + stage) < channels ? tokens[0..., 3 + stage, t] : prev
            }
        }
    }
    func testGeneratesAudioAndStopsWithoutHittingTheContextCap() async throws {
        let dir = try Self.modelDir()
        let runtime = try await Dia2Runtime.load(from: dir)
        var config = Dia2GenerationConfig()
        config.audioTemperature = 0.0   // greedy, so the test is deterministic
        config.textTemperature = 0.0
        config.cfgScale = 1.0           // single branch: faster, still exercises the loop

        let session = Dia2Session(runtime: runtime, config: config, prefix: nil)
        await session.append(["[S1] Hello there. [S2] Hi."])
        await session.finish()

        var samples: [Float] = []
        for try await chunk in session.audio { samples.append(contentsOf: chunk.samples) }

        XCTAssertGreaterThan(samples.count, 12_000, "expected at least ~0.5s of audio")
        XCTAssertLessThan(samples.count, 24_000 * 30, "must stop well before the 2-minute cap")
        XCTAssertLessThanOrEqual(samples.map(abs).max() ?? 0, 1.0, "output must be in range")
        XCTAssertGreaterThan(samples.map(abs).max() ?? 0, 0.01, "output must not be silence")
    }
    func testFacadeExposesTagsAndSampleRate() async throws {
        let dir = try Self.modelDir()
        let model = try await Dia2Model.load(from: dir)
        XCTAssertEqual(model.sampleRate, 24000)
        XCTAssertTrue(model.nonverbalTags.contains("(laughs)"))
        XCTAssertTrue(model.nonverbalTags.contains("(sighs)"))
        XCTAssertFalse(model.nonverbalTags.contains("[S1]"), "speaker tags are not nonverbals")
    }
}
