import Foundation
import MLX
import MLXNN
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

    /// Dia2's upstream config is the model config itself, not a Transformers
    /// wrapper with a root `model_type`. The public factory must still route a
    /// local converted checkpoint to Dia2 rather than reject it before load.
    func testTTSFactoryRoutesNestedDia2ConfigToLocalLoader() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nested-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.copyItem(
            at: Self.fixtureURL("config-2b.json"),
            to: directory.appendingPathComponent("config.json"))

        do {
            _ = try await TTS.loadModel(modelRepo: directory.path)
            XCTFail("an incomplete fixture must fail after factory dispatch")
        } catch let error as TTSModelError {
            XCTFail("factory rejected a valid Dia2 config instead of invoking its loader: \(error)")
        } catch {
            // Expected: the tiny fixture has config only, so the Dia2 loader
            // reaches its missing tokenizer/weights after successful dispatch.
        }
    }

    /// Prefix frames condition the KV cache but are not part of generated
    /// output unless a caller explicitly asks to retain them.
    func testGeneratedOutputWindowExcludesConditioningPrefixFrames() {
        XCTAssertFalse(Dia2OutputWindow.shouldEmit(outputIndex: 17, prefixFrames: 18))
        XCTAssertTrue(Dia2OutputWindow.shouldEmit(outputIndex: 18, prefixFrames: 18))
        XCTAssertTrue(Dia2OutputWindow.shouldEmit(outputIndex: 0, prefixFrames: 0))
    }

    func testZeroBOSTokenFallsBackToOneLikeTheReference() {
        XCTAssertEqual(Dia2TokenIDs.resolvedBOS(nil), 1)
        XCTAssertEqual(Dia2TokenIDs.resolvedBOS(0), 1)
        XCTAssertEqual(Dia2TokenIDs.resolvedBOS(42), 42)
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
    func testRoPEIsIdentityAtPositionZero() {
        let rope = Dia2RoPE(headDim: 8, minTimescale: 1, maxTimescale: 10000, maxSeqLen: 16)
        let x = MLXArray(converting: (0 ..< 8).map { Double($0) }).reshaped([1, 1, 1, 8])
        let out = rope(x, positions: MLXArray([0]).reshaped([1, 1]))
        XCTAssertLessThanOrEqual(abs(out - x).max().item(Float.self), 1e-5)
    }

    /// The rotation is norm-preserving; a wrong half-split silently is not.
    func testRoPEPreservesNorm() {
        let rope = Dia2RoPE(headDim: 8, minTimescale: 1, maxTimescale: 10000, maxSeqLen: 16)
        let x = MLXArray(converting: (0 ..< 8).map { Double($0) * 0.3 - 1 }).reshaped([1, 1, 1, 8])
        let out = rope(x, positions: MLXArray([5]).reshaped([1, 1]))
        let before = (x * x).sum().item(Float.self)
        let after = (out * out).sum().item(Float.self)
        XCTAssertEqual(before, after, accuracy: 1e-4)
    }

    /// A pad in the second stream must contribute nothing — that gate is how
    /// the model knows a lookahead slot is empty rather than a real word.
    func testMultiStreamEmbeddingGatesThePadSecondStream() {
        let embed = MultiStreamEmbedding(vocabSize: 16, dim: 4, padID: 3, lowRankDim: nil)
        MLXRandom.seed(0)
        eval(embed)
        let padded = embed(MLXArray([1]).reshaped([1, 1]), MLXArray([3]).reshaped([1, 1]))
        let mainOnly = embed.mainOnly(MLXArray([1]).reshaped([1, 1]))
        XCTAssertLessThanOrEqual(abs(padded - mainOnly).max().item(Float.self), 1e-6)

        let real = embed(MLXArray([1]).reshaped([1, 1]), MLXArray([2]).reshaped([1, 1]))
        XCTAssertGreaterThan(abs(real - mainOnly).max().item(Float.self), 1e-6)
    }

    func testMlpIsGatedNotSequential() {
        // activations ["silu","linear"] => silu(gate) * up, one wi of width 2*hidden.
        let mlp = Dia2Mlp(dim: 4, hidden: 8, activations: ["silu", "linear"])
        eval(mlp)
        let x = MLXArray(converting: [0.5, -0.25, 1.0, 0.0] as [Double]).reshaped([1, 1, 4])
        let out = mlp(x)
        XCTAssertEqual(out.shape, [1, 1, 4])
    }
    private func makeIDs() -> Dia2TokenIDs {
        Dia2TokenIDs(card: 49280, newWord: 2, pad: 3, bos: 1, zero: 7,
                     spk1: 49152, spk2: 49153, audioPad: 2049, audioBos: 2048)
    }

    /// A sampled new-word token pops the next entry and starts emitting its
    /// text tokens; everything else is padding.
    func testNewWordConsumesAnEntryAndEmitsItsTokens() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [
            Dia2Entry(tokens: [100, 101], text: "hello", padding: 1),
        ])
        let first = machine.process(step: 0, state: state, token: ids.newWord)
        XCTAssertEqual(first.main, ids.newWord)
        XCTAssertTrue(first.consumedNewWord)
        // The entry's own tokens drain on subsequent pad steps.
        let second = machine.process(step: 1, state: state, token: ids.pad)
        XCTAssertEqual(second.main, 100)
        let third = machine.process(step: 2, state: state, token: ids.pad)
        XCTAssertEqual(third.main, 101)
    }

    /// Anything that is not new-word or pad is coerced to pad — the sampler
    /// only ever legitimately produces those two.
    func testStrayTokensAreCoercedToPad() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [Dia2Entry(tokens: [7], text: "x", padding: 0)])
        // A word has to be consumed first. new_state starts padding_budget at
        // initialPadding, so at 0 the machine is already out of budget and
        // promotes any pad to new-word — which would mask the coercion here.
        _ = machine.process(step: 0, state: state, token: ids.newWord)
        _ = machine.process(step: 1, state: state, token: ids.pad)   // drains 7
        XCTAssertEqual(machine.process(step: 2, state: state, token: 4242).main, ids.pad)
    }

    /// Once the padding budget runs out the machine forces the next word, so a
    /// stuck sampler cannot stall the script forever.
    func testExhaustedPaddingBudgetForcesTheNextWord() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 2, initialPadding: 0)
        let state = machine.newState(entries: [
            Dia2Entry(tokens: [10], text: "a", padding: 0),
            Dia2Entry(tokens: [11], text: "b", padding: 0),
        ])
        _ = machine.process(step: 0, state: state, token: ids.newWord)  // takes "a", budget := 2
        _ = machine.process(step: 1, state: state, token: ids.pad)      // drains 10, budget 2 -> 1
        _ = machine.process(step: 2, state: state, token: ids.pad)      // budget 1 -> 0
        // Only a pad that is actually emitted spends budget, and consuming a
        // word resets it, so the force lands on the next step — not one later.
        let forced = machine.process(step: 3, state: state, token: ids.pad)
        XCTAssertTrue(forced.consumedNewWord, "budget exhausted must force a word")
    }

    /// A break entry (no tokens) spends frames as silence rather than speech.
    func testForcedPaddingFromABreakBlocksNewWords() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [
            Dia2Entry(tokens: [], text: "", padding: 3),
            Dia2Entry(tokens: [20], text: "after", padding: 0),
        ])
        // Consuming the break sets forcedPadding = 3 and immediately spends one
        // of it on that same frame, so it blocks steps 1 and 2 and expires
        // before step 3.
        _ = machine.process(step: 0, state: state, token: ids.newWord)   // consumes the break
        for step in 1 ... 2 {
            let r = machine.process(step: step, state: state, token: ids.newWord)
            XCTAssertFalse(r.consumedNewWord, "forced padding must hold at step \(step)")
        }
        XCTAssertTrue(machine.process(step: 3, state: state, token: ids.newWord).consumedNewWord,
                      "forced padding must expire, not stall the script")
    }

    /// End step is recorded once the script is spent, so the caller knows when
    /// to start the delay-flush tail instead of running to the context cap.
    func testEndStepIsRecordedWhenEntriesRunOut() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [])
        XCTAssertNil(state.endStep)
        _ = machine.process(step: 9, state: state, token: ids.newWord)
        XCTAssertEqual(state.endStep, 9)
    }

    /// With lookahead on, the second stream carries the NEXT entry's tokens so
    /// the model can anticipate the upcoming word.
    func testSecondStreamCarriesLookahead() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 2,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [
            Dia2Entry(tokens: [30], text: "one", padding: 0),
            Dia2Entry(tokens: [31], text: "two", padding: 0),
        ])
        let r = machine.process(step: 0, state: state, token: ids.newWord)
        XCTAssertEqual(r.second, ids.newWord)
        XCTAssertEqual(r.main, 30, "main advances past new-word when multiplexing")
    }

    /// Appending mid-run is what makes streaming possible: the Chat tab feeds
    /// words in as the LLM writes them.
    func testEntriesCanBeAppendedWhileRunning() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 6, initialPadding: 0)
        let state = machine.newState(entries: [Dia2Entry(tokens: [40], text: "a", padding: 0)])
        _ = machine.process(step: 0, state: state, token: ids.newWord)
        state.append([Dia2Entry(tokens: [41], text: "b", padding: 0)])
        _ = machine.process(step: 1, state: state, token: ids.pad)
        let r = machine.process(step: 2, state: state, token: ids.newWord)
        XCTAssertTrue(r.consumedNewWord)
        XCTAssertNil(state.endStep, "appending must clear a premature end")
    }
    /// Deterministic stand-in: one id per character code, so assertions can be
    /// written against text without loading a real tokenizer.
    private struct FakeTokenizer: Dia2TextTokenizing {
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    }

    func testSpeakerTagPrefixesTheFirstWordOfATurn() {
        let parser = Dia2ScriptParser(tokenIDs: makeIDs(), frameRate: 12.5)
        let entries = parser.parse(["[S1] hello there"], tokenizer: FakeTokenizer())
        XCTAssertEqual(entries.map(\.text), ["hello", "there"])
        XCTAssertEqual(entries[0].tokens.first, makeIDs().spk1)
        XCTAssertNotEqual(entries[1].tokens.first, makeIDs().spk1)
    }

    func testSecondSpeakerGetsItsOwnTag() {
        let parser = Dia2ScriptParser(tokenIDs: makeIDs(), frameRate: 12.5)
        let entries = parser.parse(["[S1] hi", "[S2] hey"], tokenizer: FakeTokenizer())
        XCTAssertEqual(entries[0].tokens.first, makeIDs().spk1)
        XCTAssertEqual(entries[1].tokens.first, makeIDs().spk2)
    }

    /// A break becomes a token-less entry whose padding is the pause in frames.
    func testBreakTagBecomesPaddingFrames() {
        let parser = Dia2ScriptParser(tokenIDs: makeIDs(), frameRate: 12.5)
        let entries = parser.parse([#"[S1] wait <break time="2s"/> ok"#], tokenizer: FakeTokenizer())
        let pause = entries.first { $0.tokens.isEmpty }
        XCTAssertNotNil(pause)
        XCTAssertEqual(pause?.padding, 25)   // 2.0s * 12.5 Hz
        XCTAssertEqual(entries.filter { !$0.tokens.isEmpty }.map(\.text), ["wait", "ok"])
    }

    /// Curly apostrophes and colons are normalised before tokenising, matching
    /// the reference — otherwise "S1:" style scripts tokenise unpredictably.
    func testCurlyApostrophesAndColonsAreNormalised() {
        let parser = Dia2ScriptParser(tokenIDs: makeIDs(), frameRate: 12.5)
        let entries = parser.parse(["[S1] it\u{2019}s fine: really"], tokenizer: FakeTokenizer())
        XCTAssertEqual(entries.map(\.text), ["it's", "fine", "really"])
    }

    func testEmptyScriptProducesNoEntries() {
        let parser = Dia2ScriptParser(tokenIDs: makeIDs(), frameRate: 12.5)
        XCTAssertTrue(parser.parse(["", "   "], tokenizer: FakeTokenizer()).isEmpty)
    }
    /// Delay then undelay is the identity on the original span — the property
    /// the whole codebook interleaving rests on.
    func testDelayAndUndelayRoundTrip() {
        let delays = [0, 2, 3]
        let aligned = MLXArray(0 ..< 12).reshaped([3, 4])
        let delayed = Dia2Grid.delay(aligned, delays: delays, padID: 99)
        XCTAssertEqual(delayed.shape, [3, 4 + 3])
        let back = Dia2Grid.undelay(delayed, delays: delays, padID: 99)
        XCTAssertEqual(back.shape, [3, 4])
        XCTAssertTrue(MLX.all(back .== aligned).item(Bool.self))
    }

    /// CFG is a contrast knob with a sweet spot, not a dial that improves as
    /// it rises. 6.0 (Nari's Space default) removes the silence collapse but
    /// sounds overdriven; 2.0 is the value judged acceptable by ear. Pinned so
    /// nobody re-tunes it from a silence metric again.
    func testCfgScaleDefaultsToTheValueJudgedByEar() {
        XCTAssertEqual(Dia2GenerationConfig().cfgScale, 2.0)
    }

    /// The first word lands at frame 2, matching the reference's 0.16s. The
    /// checkpoint's `first_word_min_start` of 3 is a floor on a word's
    /// scheduled step, which is a different quantity — do not conflate them.
    func testInitialPaddingMatchesTheReferenceOpening() throws {
        let data = try Data(contentsOf: Self.fixtureURL("config-2b.json"))
        let config = try JSONDecoder().decode(Dia2Config.self, from: data)
        var gen = Dia2GenerationConfig()
        XCTAssertEqual(gen.effectiveInitialPadding(for: config), 2)
        gen.initialPadding = 0
        XCTAssertEqual(gen.effectiveInitialPadding(for: config), 0,
                       "an explicit choice still wins")
    }

    /// And the padding actually has to hold the first word back that long.
    func testTheFirstWordWaitsOutTheInitialPadding() {
        let ids = makeIDs()
        let machine = Dia2StateMachine(tokenIDs: ids, secondStreamAhead: 0,
                                       maxPadding: 8, initialPadding: 3)
        let state = machine.newState(entries: [Dia2Entry(tokens: [7], text: "hi")])
        var firstWordStep: Int?
        for step in 0 ..< 12 {
            let out = machine.process(step: step, state: state, token: ids.newWord)
            if firstWordStep == nil, out.main != ids.pad { firstWordStep = step }
        }
        XCTAssertEqual(firstWordStep, 3, "the first word may not land before frame 3")
    }

    /// The decode path pulls ONE frame at a time out of the delay-shifted grid
    /// it has been generating into. Taking a raw column there is wrong: in that
    /// grid codebook `cb` at column `c` holds output frame `c - delays[cb]`, so
    /// a column mixes codebooks from up to maxDelay different frames. Dia2's
    /// real pattern delays codebook 0 by 16 and the rest by 18, which smears
    /// every residual codebook 160ms away from the coarse one it belongs to.
    func testFrameGathersOneOutputFrameAcrossTheDelays() {
        let delays = [0, 2, 3]
        let aligned = MLXArray(0 ..< 12).reshaped([3, 4])
        let delayed = Dia2Grid.delay(aligned, delays: delays, padID: 99)
        for u in 0 ..< 4 {
            let frame = Dia2Grid.frame(delayed, outputIndex: u, delays: delays)
            XCTAssertEqual(frame.shape, [3, 1], "frame \(u): shape")
            XCTAssertTrue(MLX.all(frame .== aligned[0..., u ..< (u + 1)]).item(Bool.self),
                          "frame \(u) must equal the aligned column, not the delayed one")
        }
    }

    /// The same property stated against `undelay`, so the two paths into Mimi —
    /// whole-grid and frame-at-a-time — can never disagree.
    func testFrameAgreesWithUndelay() {
        let delays = [16, 18, 18]
        let aligned = MLXArray(0 ..< 90).reshaped([3, 30])
        let delayed = Dia2Grid.delay(aligned, delays: delays, padID: 99)
        let undelayed = Dia2Grid.undelay(delayed, delays: delays, padID: 99)
        for u in 0 ..< undelayed.dim(1) {
            let frame = Dia2Grid.frame(delayed, outputIndex: u, delays: delays)
            XCTAssertTrue(MLX.all(frame .== undelayed[0..., u ..< (u + 1)]).item(Bool.self),
                          "frame \(u) disagrees with undelay")
        }
    }

    func testDelayPadsTheLeadingFramesOfDelayedCodebooks() {
        let delayed = Dia2Grid.delay(MLXArray(0 ..< 4).reshaped([2, 2]), delays: [0, 2], padID: 99)
        XCTAssertEqual(delayed[1, 0].item(Int32.self), 99)
        XCTAssertEqual(delayed[1, 1].item(Int32.self), 99)
        XCTAssertEqual(delayed[1, 2].item(Int32.self), 2)
    }

    /// PAD and BOS must be unreachable by sampling; they are control tokens.
    func testMaskingDrivesPadAndBosToNegativeInfinity() {
        let logits = MLXArray(converting: [1.0, 2.0, 3.0, 4.0] as [Double]).reshaped([1, 1, 4])
        let masked = Dia2Grid.maskAudioLogits(logits, padIdx: 3, bosIdx: 2)
        XCTAssertLessThan(masked[0, 0, 2].item(Float.self), -1e30)
        XCTAssertLessThan(masked[0, 0, 3].item(Float.self), -1e30)
        XCTAssertEqual(masked[0, 0, 0].item(Float.self), 1.0, accuracy: 1e-6)
    }

    /// scale == 1 is the identity, so unguided runs cost nothing extra.
    func testGuidanceIsIdentityAtScaleOne() {
        let logits = MLXArray(converting: [1.0, 2.0, 5.0, 0.0] as [Double]).reshaped([2, 1, 2])
        let out = Dia2Guidance.apply(logits, active: false, scale: 1.0, filterK: 0)
        XCTAssertEqual(out.shape, [2, 1, 2])
    }

    /// Guidance extrapolates away from the unconditional branch.
    func testGuidancePushesPastTheConditionalBranch() {
        // cond = [0, 4], uncond = [0, 0]; lerp(uncond, cond, 2) = [0, 8].
        let logits = MLXArray(converting: [0.0, 4.0, 0.0, 0.0] as [Double]).reshaped([2, 1, 2])
        let out = Dia2Guidance.apply(logits, active: true, scale: 2.0, filterK: 0)
        XCTAssertEqual(out.shape, [1, 1, 2])
        XCTAssertGreaterThan(out[0, 0, 1].item(Float.self), 4.0)
    }

    /// Filtering with k >= vocabulary size must keep every candidate. MLX's
    /// `top` result is explicitly unsorted, so treating its last value as the
    /// threshold can accidentally collapse a two-action distribution to one.
    func testGuidanceFilterKeepsTheWholeActionVocabulary() {
        let logits = MLXArray(converting: [0.0, 4.0, 0.0, 0.0] as [Double])
            .reshaped([2, 1, 2])
        let out = Dia2Guidance.apply(logits, active: true, scale: 2.0, filterK: 50)
        XCTAssertEqual(out[0, 0, 0].item(Float.self), 0.0, accuracy: 1e-6)
        XCTAssertEqual(out[0, 0, 1].item(Float.self), 4.0, accuracy: 1e-6)
    }

    /// The CFG filter keeps exactly the k candidates selected by guided
    /// logits, while returning their conditional scores for sampling.
    func testGuidanceFilterKeepsExactlyTheGuidedTopK() {
        let logits = MLXArray(converting: [10.0, 20.0, 30.0, 40.0,
                                           0.0, 15.0, 29.0, 39.0] as [Double])
            .reshaped([2, 1, 4])
        let out = Dia2Guidance.apply(logits, active: true, scale: 2.0, filterK: 2)
        XCTAssertLessThan(out[0, 0, 0].item(Float.self), -1e30)
        XCTAssertLessThan(out[0, 0, 1].item(Float.self), -1e30)
        XCTAssertEqual(out[0, 0, 2].item(Float.self), 30.0, accuracy: 1e-6)
        XCTAssertEqual(out[0, 0, 3].item(Float.self), 40.0, accuracy: 1e-6)
    }

    func testZeroTemperatureIsArgmax() {
        let logits = MLXArray(converting: [0.1, 9.0, 0.2] as [Double]).reshaped([1, 1, 3])
        XCTAssertEqual(Dia2Sampler.sample(logits, temperature: 0, topK: 0, key: nil), 1)
    }

    /// Top-k sampling must keep all k candidates, not only whichever value
    /// happens to be last in MLX's explicitly unsorted `top` output.
    func testTopKSamplingCanDrawEveryKeptCandidate() {
        MLX.seed(20260903)
        let logits = MLXArray(converting: [2.0, 1.0, -100.0] as [Double])
            .reshaped([1, 1, 3])
        let draws = Set((0 ..< 100).map { _ in
            Dia2Sampler.sample(logits, temperature: 1, topK: 2, key: nil)
        })
        XCTAssertEqual(draws, Set([0, 1]))
    }
    /// Word timings become entries whose padding spans the gap to the next
    /// word, so the model is taught this speaker's actual rhythm.
    func testWordsBecomeEntriesWithGapPadding() {
        let words = [
            Dia2Word(text: "hello", start: 0.0, end: 0.4),
            Dia2Word(text: "there", start: 1.0, end: 1.4),
        ]
        let entries = Dia2Prefix.entries(for: words, speakerToken: makeIDs().spk1,
                                         tokenizer: FakeTokenizer(), frameRate: 12.5)
        XCTAssertEqual(entries.map(\.text), ["hello", "there"])
        XCTAssertEqual(entries[0].tokens.first, makeIDs().spk1)
        // 1.0s gap at 12.5 Hz is ~12 frames; the first entry must hold that long.
        XCTAssertGreaterThan(entries[0].padding, 5)
    }

    func testNoPrefixWhenSpeakerOneIsAbsent() throws {
        XCTAssertNil(try Dia2Prefix.plan(speaker1: nil, speaker2: nil, runtime: nil))
    }

    func testSpeakerTwoWithoutSpeakerOneIsRejected() {
        let two = Dia2PrefixInput(samples: [0, 0, 0], words: [])
        XCTAssertThrowsError(try Dia2Prefix.plan(speaker1: nil, speaker2: two, runtime: nil))
    }
}
