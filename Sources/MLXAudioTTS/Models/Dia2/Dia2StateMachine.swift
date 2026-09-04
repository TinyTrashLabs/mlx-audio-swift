import Foundation

public struct Dia2TokenIDs: Sendable, Equatable {
    public var card: Int
    public var newWord: Int
    public var pad: Int
    public var bos: Int
    public var zero: Int
    public var spk1: Int
    public var spk2: Int
    public var audioPad: Int
    public var audioBos: Int
    public var ungenerated: Int = -2

    public init(card: Int, newWord: Int, pad: Int, bos: Int, zero: Int,
                spk1: Int, spk2: Int, audioPad: Int, audioBos: Int) {
        self.card = card; self.newWord = newWord; self.pad = pad; self.bos = bos
        self.zero = zero; self.spk1 = spk1; self.spk2 = spk2
        self.audioPad = audioPad; self.audioBos = audioBos
    }

    /// Mirrors Python's `candidate or 1`: nil and zero both fall back to 1.
    static func resolvedBOS(_ candidate: Int?) -> Int {
        guard let candidate, candidate != 0 else { return 1 }
        return candidate
    }
}

/// One scheduling unit: a word's text tokens, plus how many frames of silence
/// follow it. A break is an entry with no tokens and nonzero padding.
public struct Dia2Entry: Sendable, Equatable {
    public var tokens: [Int]
    public var text: String
    public var padding: Int
    public init(tokens: [Int], text: String, padding: Int = 0) {
        self.tokens = tokens; self.text = text; self.padding = padding
    }
}

/// Mutable per-generation state. A class because the generation loop mutates it
/// in place every frame and streaming appends to it from outside.
public final class Dia2State {
    var entries: [Dia2Entry]
    var head: Int = 0
    var paddingBudget: Int
    var forcedPadding: Int
    var pendingTokens: [Int] = []
    var pendingHead: Int = 0
    var lookaheadTokens: [Int] = []
    var lookaheadHead: Int = 0
    public private(set) var endStep: Int?
    public private(set) var transcript: [(String, Int)] = []
    /// Frames on which a word was consumed. Prefix warmup replays these.
    public private(set) var consumptionTimes: [Int] = []

    init(entries: [Dia2Entry], paddingBudget: Int, forcedPadding: Int) {
        self.entries = entries
        self.paddingBudget = paddingBudget
        self.forcedPadding = forcedPadding
    }

    var hasEntries: Bool { head < entries.count }
    var hasPending: Bool { pendingHead < pendingTokens.count }

    /// Streaming: more script arrived. Clearing `endStep` matters — the loop
    /// stops at end plus the flush tail, and new words must cancel that.
    public func append(_ more: [Dia2Entry]) {
        guard !more.isEmpty else { return }
        entries.append(contentsOf: more)
        endStep = nil
    }

    /// Consumes any prefix entries warm-up left queued.
    ///
    /// Warm-up forces one new-word per scheduled frame, but `enforce` pads a
    /// forced new-word away while an earlier word's tokens are still pending,
    /// and two words can round onto the same frame. Either way the head lags,
    /// and whatever is still queued when free generation starts is spoken as
    /// the opening of the generated turn — the reference clip's own last
    /// words. The lag grows with the prefix, so longer reference audio leaks
    /// more of itself.
    ///
    /// Draining keeps the drained words in `transcript`, because they are
    /// still prefix: their timings belong to conditioning, not to output.
    func drainPrefix(through count: Int, at step: Int) {
        while head < count {
            let entry = entries[head]
            head += 1
            if !entry.tokens.isEmpty { note(entry.text, at: step) }
            noteConsumption(step)
        }
        pendingTokens.removeAll(); pendingHead = 0
        lookaheadTokens.removeAll(); lookaheadHead = 0
        forcedPadding = 0
    }

    func recordEnd(_ step: Int) { if endStep == nil { endStep = step } }
    func note(_ text: String, at step: Int) { transcript.append((text, step)) }
    func noteConsumption(_ step: Int) { consumptionTimes.append(step) }
}

/// Turns the model's binary action stream (pad / new word) into the two text
/// streams it is fed on the next frame. Ported from dia2/runtime/state_machine.py.
public struct Dia2StateMachine: Sendable {
    public let tokenIDs: Dia2TokenIDs
    public let secondStreamAhead: Int
    public let maxPadding: Int
    public let initialPadding: Int

    public init(tokenIDs: Dia2TokenIDs, secondStreamAhead: Int,
                maxPadding: Int = 6, initialPadding: Int = 0) {
        self.tokenIDs = tokenIDs
        self.secondStreamAhead = secondStreamAhead
        self.maxPadding = maxPadding
        self.initialPadding = initialPadding
    }

    public func newState(entries: [Dia2Entry]) -> Dia2State {
        Dia2State(entries: entries, paddingBudget: initialPadding, forcedPadding: initialPadding)
    }

    @discardableResult
    public func process(step: Int, state: Dia2State, token rawToken: Int,
                        isForced: Bool = false) -> (main: Int, second: Int, consumedNewWord: Bool) {
        var token = sanitize(rawToken)
        token = enforce(state: state, token: token, isForced: isForced)
        let (afterWord, consumed) = handleNewWord(step: step, state: state, token: token)
        let output = select(state: state, token: afterWord)
        let (main, second) = multiplex(state: state, output: output)
        return (main, second, consumed)
    }

    private func sanitize(_ token: Int) -> Int {
        // The reference maps the action head's 0/1 onto the text ids first.
        if token == 1 { return tokenIDs.newWord }
        if token == 0 { return tokenIDs.pad }
        return (token == tokenIDs.newWord || token == tokenIDs.pad) ? token : tokenIDs.pad
    }

    private func enforce(state: Dia2State, token: Int, isForced: Bool) -> Int {
        if state.hasPending { return tokenIDs.pad }
        if isForced { return token }
        if state.forcedPadding > 0 { return tokenIDs.pad }
        if state.paddingBudget <= 0 && token != tokenIDs.newWord { return tokenIDs.newWord }
        return token
    }

    private func handleNewWord(step: Int, state: Dia2State, token: Int) -> (Int, Bool) {
        guard token == tokenIDs.newWord else { return (token, false) }
        guard state.hasEntries else {
            state.recordEnd(step)
            // With lookahead the stream keeps ticking new-word so the second
            // channel stays aligned while the tail flushes.
            return (secondStreamAhead > 0 && state.endStep == step ? tokenIDs.newWord : tokenIDs.pad,
                    false)
        }
        let entry = state.entries[state.head]
        state.head += 1
        state.noteConsumption(step)
        var out = token
        if entry.tokens.isEmpty {
            out = tokenIDs.pad
        } else {
            state.note(entry.text, at: step)
            state.pendingTokens.append(contentsOf: entry.tokens)
            if secondStreamAhead > 0 {
                state.lookaheadTokens.append(contentsOf: peek(state: state, count: secondStreamAhead))
            }
            state.paddingBudget = maxPadding
        }
        state.forcedPadding = entry.padding
        return (out, true)
    }

    private func peek(state: Dia2State, count: Int) -> [Int] {
        var remaining = count
        var index = state.head
        while index < state.entries.count {
            let entry = state.entries[index]
            if !entry.tokens.isEmpty {
                remaining -= 1
                if remaining == 0 { return entry.tokens }
            }
            index += 1
        }
        return []
    }

    private func select(state: Dia2State, token: Int) -> Int {
        if token == tokenIDs.pad {
            if state.paddingBudget > 0 { state.paddingBudget -= 1 }
            if state.forcedPadding > 0 { state.forcedPadding -= 1 }
            if state.hasPending {
                let next = state.pendingTokens[state.pendingHead]
                state.pendingHead += 1
                if !state.hasPending { state.pendingTokens.removeAll(); state.pendingHead = 0 }
                return next
            }
            return tokenIDs.pad
        }
        return token
    }

    private func multiplex(state: Dia2State, output: Int) -> (Int, Int) {
        guard secondStreamAhead > 0 else { return (output, output) }
        if output == tokenIDs.newWord {
            var main = tokenIDs.pad
            if state.hasPending {
                main = state.pendingTokens[state.pendingHead]
                state.pendingHead += 1
                if !state.hasPending { state.pendingTokens.removeAll(); state.pendingHead = 0 }
            }
            return (main, tokenIDs.newWord)
        }
        if state.lookaheadHead < state.lookaheadTokens.count {
            let second = state.lookaheadTokens[state.lookaheadHead]
            state.lookaheadHead += 1
            if state.lookaheadHead == state.lookaheadTokens.count {
                state.lookaheadTokens.removeAll(); state.lookaheadHead = 0
            }
            return (output, second)
        }
        return (output, tokenIDs.pad)
    }
}
