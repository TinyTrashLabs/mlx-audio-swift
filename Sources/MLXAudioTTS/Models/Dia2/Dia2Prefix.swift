import Foundation
import MLX
import MLXLMCommon

public struct Dia2Word: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    public init(text: String, start: Double, end: Double) {
        self.text = text; self.start = start; self.end = end
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
    public static func entries(for words: [Dia2Word], speakerToken: Int,
                               tokenizer: any Dia2TextTokenizing,
                               frameRate: Double) -> [Dia2Entry] {
        var entries: [Dia2Entry] = []
        var current = 0
        for (index, word) in words.enumerated() {
            var tokens = index == 0
                ? tokenizer.encode(word.text)
                : tokenizer.encode(word.text)
            if index == 0, tokens.first != speakerToken { tokens.insert(speakerToken, at: 0) }
            let start = max(current + 1, Int((word.start * frameRate).rounded()))
            let end = start + tokens.count
            let nextStart: Int = index < words.count - 1
                ? max(end + 1, Int((words[index + 1].start * frameRate).rounded()))
                : max(end + 1, Int((word.end * frameRate).rounded()))
            entries.append(Dia2Entry(tokens: tokens, text: word.text,
                                     padding: max(0, nextStart - start - 1)))
            current = end
        }
        return entries
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

        var entries = entries(for: speaker1.words, speakerToken: runtime.tokenIDs.spk1,
                              tokenizer: runtime.tokenizer, frameRate: runtime.mimi.frameRate)
        var tokens = encode(speaker1)
        // Matches the reference's BOS/PAD offset before the first prefix word.
        var steps = stepStarts(speaker1.words, frameRate: runtime.mimi.frameRate).map { $0 + 3 }

        if let speaker2 {
            let frames = tokens.dim(1)
            entries += Self.entries(for: speaker2.words, speakerToken: runtime.tokenIDs.spk2,
                               tokenizer: runtime.tokenizer, frameRate: runtime.mimi.frameRate)
            steps += stepStarts(speaker2.words, frameRate: runtime.mimi.frameRate).map { $0 + frames }
            tokens = concatenated([tokens, encode(speaker2)], axis: 1)
        }
        return Dia2PrefixPlan(entries: entries, newWordSteps: steps,
                              alignedTokens: tokens, alignedFrames: tokens.dim(1))
    }

    private static func stepStarts(_ words: [Dia2Word], frameRate: Double) -> [Int] {
        words.map { max(0, Int(($0.start * frameRate).rounded()) - 1) }
    }

    /// Teacher-forces the prefix through the transformer so its KV cache holds
    /// the conditioning context. Returns the step the real generation starts at.
    static func warmUp(_ plan: Dia2PrefixPlan, runtime: Dia2Runtime,
                       machine: Dia2StateMachine, state: Dia2State,
                       cache: [KVCacheSimple], branches: Int) throws -> Int {
        let ids = runtime.tokenIDs
        let channels = runtime.config.data.channels
        let forcedSteps = Set(plan.newWordSteps)
        var stepTokens = MLXArray.full([branches, channels, 1],
                                       values: MLXArray(Int32(ids.pad)), type: Int32.self)

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
