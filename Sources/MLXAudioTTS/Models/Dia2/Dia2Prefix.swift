import Foundation
import MLX
import MLXLMCommon

public struct Dia2Word: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    /// Which speaker says this word: 1, 2, or nil for "whoever was already
    /// speaking". A recorded reference clip is one person, so nil throughout
    /// is the ordinary case and behaves exactly as before.
    ///
    /// It is set when the clip is a two-speaker exchange — which is what a
    /// CONTINUATION prefix is: the tail of the previous pass, both voices in
    /// it, handed back as conditioning so the next pass carries on from how
    /// the take actually sounds rather than resetting to the reference.
    public let speaker: Int?
    public init(text: String, start: Double, end: Double, speaker: Int? = nil) {
        self.text = text; self.start = start; self.end = end; self.speaker = speaker
    }
}

/// A speaker's conditioning clip: mono samples at Mimi's rate, plus word
/// timings. Transcription happens in the app (WhisperKit); the port only
/// consumes timings.
public struct Dia2PrefixInput: Sendable {
    public let samples: [Float]
    public let words: [Dia2Word]
    public init(samples: [Float], words: [Dia2Word]) {
        self.samples = samples; self.words = words
    }
}

public struct Dia2PrefixPlan: @unchecked Sendable {
    public let entries: [Dia2Entry]
    public let newWordSteps: [Int]
    public let alignedTokens: MLXArray
    public let alignedFrames: Int
}

public enum Dia2PrefixError: Error, LocalizedError {
    case secondSpeakerWithoutFirst
    public var errorDescription: String? {
        "A second speaker prefix requires a first speaker prefix."
    }
}

public enum Dia2Prefix {
    /// Word timings to scheduling entries. Padding spans to the next word's
    /// start so silence between words is reproduced, not compressed away.
    ///
    /// Returns the new-word steps alongside the entries, derived from the same
    /// clamped start. Recomputing them from the raw word timings instead —
    /// which the port used to do — drops the `current + 1` clamp, so two words
    /// can be forced onto one frame and a force can land while the previous
    /// word's tokens are still pending. `enforce` pads those away, the text
    /// stream stops matching the audio being teacher-forced, and the speaker
    /// never binds to its voice.
    /// - Parameter spk2Token: the `[S2]` id, so `speakerToken` can be named for
    ///   the tokenizer. Defaults to nil, which reads every clip as `[S1]`.
    /// - Parameter spk1Token: the `[S1]` id. Only needed for a clip whose words
    ///   carry their own `speaker`; a single-voice reference never changes
    ///   speaker, so it can be left nil.
    public static func entries(for words: [Dia2Word], speakerToken: Int,
                               tokenizer: any Dia2TextTokenizing,
                               frameRate: Double,
                               spk2Token: Int? = nil,
                               spk1Token: Int? = nil) -> (entries: [Dia2Entry], newWordSteps: [Int]) {
        var entries: [Dia2Entry] = []
        var newWordSteps: [Int] = []
        var current = 0
        var running: Int?
        for (index, word) in words.enumerated() {
            // Who says this word. A single-voice clip leaves `speaker` nil
            // throughout, so `owner` never moves off `speakerToken` and this
            // behaves exactly as the single-speaker path always has.
            let owner: Int
            switch word.speaker {
            case 2: owner = spk2Token ?? speakerToken
            case 1: owner = spk1Token ?? speakerToken
            default: owner = running ?? speakerToken
            }
            // A tag opens the clip, and opens every turn after a speaker
            // change -- the same rule `Dia2ScriptParser` applies to a script.
            // Without it a continuation prefix reads as one long monologue and
            // the model loses the exchange it is meant to be carrying on.
            let needsTag = index == 0 || owner != running
            // The reference encodes the first word together with its speaker
            // tag -- `encode("[S1] We")` -- so the word keeps the leading space
            // the BPE vocabulary expects after a tag. Encoding "We" on its own
            // and inserting the tag in front yields a different, space-less
            // token for every prefix's opening word. `Dia2ScriptParser` already
            // does this correctly; the prefix path did not.
            let tag = (spk2Token != nil && owner == spk2Token!) ? "[S2]" : "[S1]"
            var tokens = needsTag
                ? tokenizer.encode("\(tag) \(word.text)")
                : tokenizer.encode(word.text)
            if needsTag, tokens.first != owner { tokens.insert(owner, at: 0) }
            running = owner
            let start = max(current + 1, Int((word.start * frameRate).rounded()))
            newWordSteps.append(max(0, start - 1))
            let end = start + tokens.count
            let nextStart: Int = index < words.count - 1
                ? max(end + 1, Int((words[index + 1].start * frameRate).rounded()))
                : max(end + 1, Int((word.end * frameRate).rounded()))
            entries.append(Dia2Entry(tokens: tokens, text: word.text,
                                     padding: max(0, nextStart - start - 1)))
            current = end
        }
        return (entries, newWordSteps)
    }

    /// Speaker 2's clip is concatenated after speaker 1's, so the model hears a
    /// two-person exchange before it is asked to continue one.
    public static func plan(speaker1: Dia2PrefixInput?, speaker2: Dia2PrefixInput?,
                            runtime: Dia2Runtime?) throws -> Dia2PrefixPlan? {
        guard let speaker1 else {
            if speaker2 != nil { throw Dia2PrefixError.secondSpeakerWithoutFirst }
            return nil
        }
        guard let runtime else { return nil }

        func encode(_ input: Dia2PrefixInput) -> MLXArray {
            let wave = MLXArray(input.samples).reshaped([1, 1, input.samples.count])
            return runtime.mimi.encode(wave)[0].asType(.int32)   // [C, T]
        }

        let first = entries(for: speaker1.words, speakerToken: runtime.tokenIDs.spk1,
                            tokenizer: runtime.tokenizer, frameRate: runtime.mimi.frameRate,
                            spk2Token: runtime.tokenIDs.spk2,
                            spk1Token: runtime.tokenIDs.spk1)
        var entries = first.entries
        var tokens = encode(speaker1)
        // Matches the reference's BOS/PAD offset before the first prefix word.
        var steps = first.newWordSteps.map { $0 + 3 }

        if let speaker2 {
            let frames = tokens.dim(1)
            let second = Self.entries(for: speaker2.words, speakerToken: runtime.tokenIDs.spk2,
                                      tokenizer: runtime.tokenizer,
                                      frameRate: runtime.mimi.frameRate,
                                      spk2Token: runtime.tokenIDs.spk2,
                                      spk1Token: runtime.tokenIDs.spk1)
            entries += second.entries
            steps += second.newWordSteps.map { $0 + frames }
            tokens = concatenated([tokens, encode(speaker2)], axis: 1)
        }
        return Dia2PrefixPlan(entries: entries, newWordSteps: steps,
                              alignedTokens: tokens, alignedFrames: tokens.dim(1))
    }

    /// Teacher-forces the prefix through the transformer so its KV cache holds
    /// the conditioning context. Returns the step the real generation starts at.
    /// - Parameter stepTokens: the SAME buffer the generation loop goes on to
    ///   use. The reference builds it once (`build_initial_state`: BOS on the
    ///   conditional branch, ZERO on the unconditional one), lets warm-up
    ///   mutate it, and carries it straight into `run_generation_loop`. The
    ///   port used to build a fresh all-PAD buffer here and a second, fresh
    ///   BOS buffer afterwards, so it diverged twice: the prefix's first frame
    ///   saw PAD where the model expects BOS, and generation's first frame saw
    ///   BOS -- "this is the start of a sequence" -- at the very moment it is
    ///   meant to be continuing the exchange it has just been shown. Sharing
    ///   one buffer is what makes the prefix/generation boundary continuous.
    static func warmUp(_ plan: Dia2PrefixPlan, runtime: Dia2Runtime,
                       machine: Dia2StateMachine, state: Dia2State,
                       cache: [KVCacheSimple], branches: Int,
                       stepTokens: MLXArray) throws -> Int {
        let ids = runtime.tokenIDs
        let forcedSteps = Set(plan.newWordSteps)
        var stepTokens = stepTokens

        for t in 0 ..< plan.alignedFrames {
            for cb in 0 ..< plan.alignedTokens.dim(0) {
                let delay = cb < runtime.delays.count ? runtime.delays[cb] : 0
                let index = t - delay
                let value = index >= 0
                    ? plan.alignedTokens[cb, index].asType(.int32)
                    : MLXArray(Int32(ids.audioBos))
                for b in 0 ..< branches { stepTokens[b, cb + 2, 0] = value }
            }
            let positions = repeated(MLXArray([Int32(t)]).reshaped([1, 1]), count: branches, axis: 0)
            _ = runtime.transformer.step(stepTokens, positions: positions, cache: cache)

            let forced = forcedSteps.contains(t) ? ids.newWord : ids.pad
            let processed = machine.process(step: t, state: state, token: forced, isForced: true)
            stepTokens[0, 0, 0] = MLXArray(Int32(processed.main))
            stepTokens[0, 1, 0] = MLXArray(Int32(processed.second))
            if branches > 1 {
                stepTokens[1, 0, 0] = MLXArray(Int32(ids.zero))
                stepTokens[1, 1, 0] = MLXArray(Int32(ids.pad))
            }
        }
        // Teacher-forcing does not guarantee every prefix entry was consumed;
        // anything left would be spoken as the start of the generated turn.
        state.drainPrefix(through: plan.entries.count, at: plan.alignedFrames)
        return max(plan.alignedFrames - 1, 0)
    }
}
