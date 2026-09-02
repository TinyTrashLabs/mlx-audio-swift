import Foundation
import XCTest
@testable import MLXAudioTTS

final class Dia2RuntimeTests: XCTestCase {
    static func fixtureURL(_ name: String) throws -> URL {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/dia2/\(name)")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("dia2 fixture missing: \(name)")
        }
        return url
    }

    func testConfigDecodesTheShippedTwoBConfig() throws {
        let data = try Data(contentsOf: Self.fixtureURL("config-2b.json"))
        let config = try JSONDecoder().decode(Dia2Config.self, from: data)

        XCTAssertEqual(config.data.channels, 34)
        XCTAssertEqual(config.data.numAudioChannels, 32)
        XCTAssertEqual(config.model.depformer.numDepth, 31)
        XCTAssertEqual(config.data.textVocabSize, 49280)
        XCTAssertEqual(config.data.audioVocabSize, 2050)
        XCTAssertEqual(config.data.audioBosTokenID, 2048)
        XCTAssertEqual(config.data.audioPadTokenID, 2049)
        XCTAssertEqual(config.data.secondStreamAhead, 2)
        XCTAssertEqual(config.data.maxPad, 8)
        XCTAssertEqual(config.data.firstWordMinStart, 3)
        XCTAssertEqual(config.data.delayPattern.first, 16)
        XCTAssertEqual(config.data.delayPattern.count, 32)
        XCTAssertEqual(Set(config.data.delayPattern.dropFirst()), [18])
        XCTAssertEqual(config.model.decoder.nLayer, 28)
        XCTAssertEqual(config.model.decoder.nEmbd, 2048)
        XCTAssertEqual(config.model.decoder.nHidden, 6144)
        XCTAssertEqual(config.model.decoder.gqaQueryHeads, 16)
        XCTAssertEqual(config.model.decoder.kvHeads, 8)
        XCTAssertEqual(config.model.decoder.gqaHeadDim, 128)
        XCTAssertEqual(config.model.depformer.nLayer, 4)
        XCTAssertEqual(config.model.depformer.nEmbd, 1024)
        XCTAssertEqual(config.model.depformer.gqaQueryHeads, 8)
        XCTAssertEqual(config.model.depformer.textEmbedding, false)
        XCTAssertEqual(config.model.depformer.applyRope, true)
        // Absent from the shipped config, so this pins the documented default.
        XCTAssertEqual(config.model.depformer.mlpActivations, ["silu", "linear"])
        XCTAssertEqual(config.model.linear.mlpActivations, ["silu", "linear"])
        XCTAssertEqual(config.runtime.weightsSchedule.count, 31)
        XCTAssertEqual(config.runtime.weightsSchedule.first, 0)
        XCTAssertEqual(config.runtime.weightsSchedule.last, 4)
        XCTAssertEqual(config.runtime.maxContextSteps, 1500)
        XCTAssertEqual(config.model.ropeMinTimescale, 1)
        XCTAssertEqual(config.model.ropeMaxTimescale, 10000, accuracy: 1e-6)
        XCTAssertEqual(config.model.normalizationLayerEpsilon, 1e-6, accuracy: 1e-12)
        // The shipped config carries no assets block; it must stay optional.
        XCTAssertNil(config.assets)
    }

    /// numDepth and numAudioChannels are derived from `channels`, not declared,
    /// so a different channel count must move them.
    func testDerivedCountsFollowChannels() throws {
        let json = """
        {"data":{"channels":10,"text_vocab_size":8,"audio_vocab_size":2050,
          "action_vocab_size":2,"text_pad_token_id":3,"text_new_word_token_id":2,
          "action_pad_token_id":0,"action_new_word_token_id":1},
         "model":{"decoder":{"n_layer":1,"n_embd":8,"n_hidden":8,"gqa_query_heads":1,
            "kv_heads":1,"gqa_head_dim":8},
          "depformer":{"n_layer":1,"n_embd":8,"n_hidden":8,"gqa_query_heads":1,
            "kv_heads":1,"gqa_head_dim":8}},
         "runtime":{}}
        """
        let config = try JSONDecoder().decode(Dia2Config.self, from: Data(json.utf8))
        XCTAssertEqual(config.data.numAudioChannels, 8)
        XCTAssertEqual(config.model.depformer.numDepth, 7)
        // Omitted ids fall back to the top of the audio vocab.
        XCTAssertEqual(config.data.audioPadTokenID, 2049)
        XCTAssertEqual(config.data.audioBosTokenID, 2048)
        XCTAssertEqual(config.data.textZeroTokenID, 7)
        XCTAssertEqual(config.runtime.maxContextSteps, 1500)
        XCTAssertTrue(config.data.delayPattern.isEmpty)
    }
}
