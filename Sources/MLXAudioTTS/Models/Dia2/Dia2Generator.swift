import Foundation
import MLX
import MLXAudioCodecs
import MLXLMCommon
import MLXNN

public struct Dia2GenerationConfig: Sendable {
    public var textTemperature: Float = 0.6
    public var textTopK: Int = 50
    public var audioTemperature: Float = 0.8
    public var audioTopK: Int = 50
    /// How hard generation is pushed toward the script and away from an
    /// unconditioned pass. Nari's Space defaults to 6.0 and offers 1–8, and
    /// raising it does cure the silence collapse seen at 2.0 — but it is a
    /// contrast knob, not a correctness one, and past its sweet spot the
    /// output turns harsh. 2.0 is what we have actually judged by ear.
    /// Expose it; do not tune it by a silence metric.
    public var cfgScale: Float = 2.0
    public var cfgFilterK: Int = 50
    /// Frames of silence before the first word may start.
    ///
    /// 2, not the checkpoint's `first_word_min_start` of 3: the reference
    /// implementation puts its first word at 0.16s, which is frame 2. Those
    /// are different quantities — a floor on the first word's scheduled step
    /// is not the same as the padding budget the state machine opens with.
    public var initialPadding: Int? = 2
    /// How many frames the model may wait between words before one is forced.
    /// This is Dia2's own pacing control: raising it lets the delivery breathe,
    /// lowering it presses on. nil keeps the checkpoint's own `max_pad`.
    ///
    /// Prefer this over resampling the output. Resampling stretches the
    /// waveform, so it drops the pitch with the pace and the voice sounds
    /// wrong; padding changes only how long the model waits between words.
    public var maxPadding: Int?
    public init() {}

    public func effectiveInitialPadding(for config: Dia2Config) -> Int {
        initialPadding ?? config.data.firstWordMinStart
    }

    func effectiveMaxPadding(for config: Dia2Config) -> Int {
        max(maxPadding ?? config.data.maxPad, 1)
    }
}

/// One emission: decoded samples plus any words that began inside it.
public struct Dia2Chunk: Sendable {
    public let samples: [Float]
    public let words: [(String, Double)]
}

/// Everything a session needs that is expensive to build.
public struct Dia2Runtime: @unchecked Sendable {
    public let config: Dia2Config
    public let transformer: Dia2Transformer
    public let depformer: Dia2Depformer
    public let mimi: Mimi
    public let tokenizer: any Dia2TextTokenizing
    public let tokenIDs: Dia2TokenIDs
    public let delays: [Int]
    public var frameRate: Double { mimi.frameRate }
    public var sampleRate: Int { Int(mimi.sampleRate) }
}

/// Streaming generator. Text can arrive while audio is already being produced,
/// which is what lets the Chat tab speak a reply the LLM is still writing.
public actor Dia2Session {
    private let runtime: Dia2Runtime
    private let genConfig: Dia2GenerationConfig
    private let machine: Dia2StateMachine
    private let state: Dia2State
    private let parser: Dia2ScriptParser
    private let prefix: Dia2PrefixPlan?

    private let continuation: AsyncThrowingStream<Dia2Chunk, Error>.Continuation
    public nonisolated let audio: AsyncThrowingStream<Dia2Chunk, Error>
    private var task: Task<Void, Never>?
    private var finished = false

    public init(runtime: Dia2Runtime, config: Dia2GenerationConfig, prefix: Dia2PrefixPlan?) {
        self.runtime = runtime
        self.genConfig = config
        self.prefix = prefix
        let data = runtime.config.data
        machine = Dia2StateMachine(tokenIDs: runtime.tokenIDs,
                                   secondStreamAhead: data.secondStreamAhead,
                                   maxPadding: config.effectiveMaxPadding(for: runtime.config),
                                   initialPadding:
                                       config.effectiveInitialPadding(for: runtime.config))
        state = machine.newState(entries: prefix?.entries ?? [])
        parser = Dia2ScriptParser(tokenIDs: runtime.tokenIDs, frameRate: runtime.mimi.frameRate)
        (audio, continuation) = AsyncThrowingStream<Dia2Chunk, Error>.makeStream()
    }

    /// Adds script. Safe before or during generation; the first call starts it.
    public func append(_ lines: [String]) {
        state.append(parser.parse(lines, tokenizer: runtime.tokenizer))
        if task == nil { start() }
    }

    /// No more text is coming. The loop still runs until the model stops.
    public func finish() { finished = true }

    public func cancel() {
        task?.cancel()
        continuation.finish()
    }

    private func start() {
        // The loop takes exclusive ownership of the model, the state machine and
        // its state for its whole lifetime; the session hands them over here and
        // only ever calls back through `isComplete`. That is what makes the
        // unchecked Sendable box sound — none of it is touched from two places
        // at once, though the compiler cannot see that.
        let handoff = Dia2LoopHandoff(runtime: runtime, genConfig: genConfig,
                                      machine: machine, state: state, prefix: prefix)
        task = Task.detached(priority: .userInitiated) { [handoff, continuation] in
            do {
                try Dia2Loop.run(runtime: handoff.runtime, genConfig: handoff.genConfig,
                                 machine: handoff.machine,
                                 state: handoff.state, prefix: handoff.prefix,
                                 isComplete: { [weak self] in
                                     guard let self else { return true }
                                     return await self.isFinished()
                                 },
                                 emit: { continuation.yield($0) })
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    private func isFinished() -> Bool { finished }
}

enum Dia2Loop {
    /// Runs until the script is spent and the delay tail has flushed, or the
    /// context cap is reached. `isComplete` reports whether more text may still
    /// arrive; while it is false the loop keeps stepping even with no entries
    /// left, so a streaming reply does not cut off between sentences.
    static func run(runtime: Dia2Runtime,
                    genConfig: Dia2GenerationConfig,
                    machine: Dia2StateMachine,
                    state: Dia2State,
                    prefix: Dia2PrefixPlan?,
                    isComplete: @escaping @Sendable () async -> Bool,
                    emit: @Sendable (Dia2Chunk) -> Void) throws {
        let cfg = runtime.config
        let ids = runtime.tokenIDs
        let delays = runtime.delays
        let channels = cfg.data.channels
        let audioChannels = cfg.data.numAudioChannels
        let depth = runtime.depformer.numDepth
        let maxDelay = delays.max() ?? 0
        let flushTail = maxDelay + max(cfg.data.maxPad, 1)
        let cfgActive = genConfig.cfgScale != 1.0
        let branches = cfgActive ? 2 : 1

        let transformerCache = runtime.transformer.makeCache()
        let depCache = runtime.depformer.makeCache()
        let decoder = MimiStreamingDecoder(runtime.mimi)

        var startStep = 0
        if let prefix {
            startStep = try Dia2Prefix.warmUp(prefix, runtime: runtime, machine: machine,
                                              state: state, cache: transformerCache,
                                              branches: branches)
        }

        // step tokens: [branches, channels, 1]
        var stepTokens = MLXArray.full([branches, channels, 1],
                                       values: MLXArray(Int32(ids.pad)), type: Int32.self)
        stepTokens[0, 0, 0] = MLXArray(Int32(ids.bos))
        if branches > 1 { stepTokens[1, 0, 0] = MLXArray(Int32(ids.zero)) }

        let totalSteps = cfg.runtime.maxContextSteps + startStep + 1
        var audioBuf = MLXArray.full([audioChannels, totalSteps],
                                     values: MLXArray(Int32(ids.ungenerated)), type: Int32.self)
        if let prefix {
            let delayed = Dia2Grid.delay(prefix.alignedTokens, delays: delays, padID: ids.audioPad)
            audioBuf[0..., 0 ..< delayed.dim(1)] = delayed
        }

        var eosCutoff: Int?
        var emittedWords = 0

        for offset in 0 ..< cfg.runtime.maxContextSteps {
            let t = startStep + offset
            if let cutoff = eosCutoff, t >= cutoff { break }
            if t + 1 >= totalSteps { break }
            if Task.isCancelled { break }

            runtime.depformer.resetCache(depCache)
            let positions = MLXArray([Int32(t)]).reshaped([1, 1])

            // Audio channels for this frame come from the buffer, honouring
            // each codebook's delay; anything not yet reached is BOS.
            for cb in 0 ..< audioChannels {
                let delay = cb < delays.count ? delays[cb] : 0
                let value = (delay > t || t >= totalSteps)
                    ? MLXArray(Int32(ids.audioBos))
                    : audioBuf[cb, t].asType(.int32)
                for b in 0 ..< branches { stepTokens[b, cb + 2, 0] = value }
            }
            if branches > 1 {
                stepTokens[1, 0, 0] = MLXArray(Int32(ids.zero))
                stepTokens[1, 1, 0] = MLXArray(Int32(ids.pad))
            }

            let out = runtime.transformer.step(stepTokens,
                                               positions: repeated(positions, count: branches, axis: 0),
                                               cache: transformerCache)

            let guidedAction = Dia2Guidance.apply(out.action, active: cfgActive,
                                                  scale: genConfig.cfgScale,
                                                  filterK: genConfig.cfgFilterK)
            let actionToken = Dia2Sampler.sample(guidedAction,
                                                 temperature: genConfig.textTemperature,
                                                 topK: genConfig.textTopK, key: nil)
            let processed = machine.process(step: t, state: state, token: actionToken)
            for b in 0 ..< branches {
                stepTokens[b, 0, 0] = MLXArray(Int32(processed.main))
                stepTokens[b, 1, 0] = MLXArray(Int32(processed.second))
            }

            let guidedCb0 = Dia2Guidance.apply(out.cb0, active: cfgActive,
                                               scale: genConfig.cfgScale,
                                               filterK: genConfig.cfgFilterK)
            let maskedCb0 = Dia2Grid.maskAudioLogits(guidedCb0, padIdx: ids.audioPad,
                                                     bosIdx: ids.audioBos)
            var previous = Dia2Sampler.sample(maskedCb0, temperature: genConfig.audioTemperature,
                                              topK: genConfig.audioTopK, key: nil)
            audioBuf[0, t + 1] = MLXArray(Int32(previous))

            let mainText = MLXArray([Int32(processed.main)])
            let secondText = MLXArray([Int32(processed.second)])
            for stage in 0 ..< depth {
                let logits = runtime.depformer.step(
                    stage: stage,
                    prevAudio: repeated(MLXArray([Int32(previous)]), count: branches, axis: 0),
                    hidden: out.hidden, cache: depCache,
                    mainText: stage == 0 ? mainText : nil,
                    secondText: stage == 0 ? secondText : nil)
                let guided = Dia2Guidance.apply(logits, active: cfgActive,
                                                scale: genConfig.cfgScale,
                                                filterK: genConfig.cfgFilterK)
                previous = Dia2Sampler.sample(guided, temperature: genConfig.audioTemperature,
                                              topK: genConfig.audioTopK, key: nil)
                audioBuf[stage + 1, t + 1] = MLXArray(Int32(previous))
            }

            // A frame is only real once every codebook's delay has elapsed.
            // `audioBuf` is the DELAYED grid, so the frame has to be gathered
            // per codebook at its own offset — a raw column would mix codebooks
            // from up to maxDelay different frames.
            let readyIndex = t - maxDelay
            if Dia2OutputWindow.shouldEmit(
                outputIndex: readyIndex,
                prefixFrames: prefix?.alignedFrames ?? 0
            ) {
                let frame = Dia2Grid.frame(audioBuf, outputIndex: readyIndex, delays: delays)
                let pcm = decoder.decodeFrames(frame.expandedDimensions(axis: 0))
                eval(pcm)
                let samples = pcm.reshaped([-1]).asArray(Float.self)
                var words: [(String, Double)] = []
                while emittedWords < state.transcript.count {
                    let (text, step) = state.transcript[emittedWords]
                    words.append((text, Double(step) / runtime.mimi.frameRate))
                    emittedWords += 1
                }
                emit(Dia2Chunk(samples: samples, words: words))
            }

            // Only start the flush countdown once no more text can arrive.
            if eosCutoff == nil, let end = state.endStep {
                let complete = RunLoopBridge.blockingIsComplete(isComplete)
                if complete { eosCutoff = end + flushTail }
            }
        }
    }
}

enum Dia2OutputWindow {
    static func shouldEmit(outputIndex: Int, prefixFrames: Int) -> Bool {
        outputIndex >= prefixFrames
    }
}

/// The decode loop is synchronous and CPU-bound; this is the one place it needs
/// an answer from the actor. Kept tiny and explicit rather than making the whole
/// loop async, which would add an await per frame.
enum RunLoopBridge {
    /// The result is carried in a reference box rather than a captured `var`:
    /// under strict concurrency a local mutated from inside the Task is a data
    /// race, even though the semaphore makes the handoff safe in practice.
    private final class Box: @unchecked Sendable { var value = false }

    static func blockingIsComplete(_ probe: @escaping @Sendable () async -> Bool) -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        let box = Box()
        Task { box.value = await probe(); semaphore.signal() }
        semaphore.wait()
        return box.value
    }
}


/// Everything the decode loop owns once generation starts. Unchecked because
/// MLX modules and the state machine are reference types the compiler cannot
/// prove are handed over rather than shared; see `Dia2Session.start()`.
struct Dia2LoopHandoff: @unchecked Sendable {
    let runtime: Dia2Runtime
    let genConfig: Dia2GenerationConfig
    let machine: Dia2StateMachine
    let state: Dia2State
    let prefix: Dia2PrefixPlan?
}
