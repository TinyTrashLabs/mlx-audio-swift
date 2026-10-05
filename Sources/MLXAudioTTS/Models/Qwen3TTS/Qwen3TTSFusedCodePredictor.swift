import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// Fused code-predictor frame (iOS speed work, 2026-10-05).
//
// The code predictor runs 5 layers × 15 sequential sub-steps a frame and,
// on a hot iPhone 15 Pro, is ~70% of the frame (62–71 ms of ~92). The work
// is tiny — one token, attention over at most 16 rows — so the frame is
// bound by kernel launches, whose cost grows with throttled clocks. With
// the per-layer fused step (`Qwen3TTSFusedStep`, hybrid) a frame's code
// predictor was still ~1,550 Metal dispatches (counted on the Mac, 0.6B
// 4-bit mobile). This runs the whole frame:
//
//   * a layer's closing residual add is fused with the next layer's input
//     rmsnorm, or the final norm (`cp_add_rms`, MLX's `rms_single_row`
//     reproduced);
//   * the q/k head norm + RoPE glue writes q and appends k/v to a per-frame
//     (1, kvHeads, 16, 128) cache in one kernel (`cp_exact_glue`) — no
//     KVCacheSimple slice updates or SDPA-side casts;
//   * for greedy sub-codes, argmax + the next sub-step's embedding lookup +
//     layer 0's input rmsnorm + the running codec-embedding sum the talker
//     takes next frame are one kernel (`cp_argmax_embed`);
//   * q/k/v, o, gate/up and down stay MLX's quantized matmul (it measured
//     faster on the phone than hand-written matvecs).
//
// Exact mode (default, `Qwen3TTSModel.fusedCodePredictorExact`): ~940
// dispatches a frame, and bit-identical sub-codes — which is the only kind
// of parity there is here: the sub-code logits are bf16 with top-two gaps
// of 0 or one ulp in most frames, so any rounding change flips greedy
// codes in about two frames of three. So exact mode reproduces the
// per-layer path op for op: step 0 is that path's own module pass (in the
// float32 the talker's first hidden state arrives in, leaving the float32
// cache buffers every later frame writes into), attention is MLX's SDPA
// over the frame's cache, and every custom kernel matches the MLX ops it
// replaces bit for bit (`QwenFusedCodePredictorTests`).
//
// Fast mode (`fusedCodePredictorExact = false`): ~640 dispatches. Attention
// is one kernel over the cache (`cp_attn`, MLX's sdpa_vector line for line,
// but a runtime-compiled kernel is not fast-math like MLX's metallib, so
// ~1 output in 10^4 lands an ulp apart) and step 0 is a fused two-row bf16
// step. Different rounding, so different — equally valid, not identical —
// sub-codes.
final class Qwen3TTSFusedCodePredictor {
    let layers: [Qwen3TTSFusedStep.Layer]
    let finalNorm: MLXArray
    let lmHeads: [Qwen3TTSFusedStep.Layer.Projection]
    /// Each sub-code's embedding table, (vocab, embedDim), plain (unquantized) weights.
    let embeddings: [MLXArray]
    let projection: Linear?
    let hidden: Int, intermediate: Int, vocab: Int, embedDim: Int
    let numHeads: Int, numKvHeads: Int
    /// KV rows a frame needs: 2 for step 0, then one per sub-step.
    let capacity: Int
    let eps: Float
    let params: MLXArray
    let positions: [MLXArray]
    private var emptyCaches: [DType: MLXArray] = [:]
    /// Unowned: the predictor owns this object (`fusedFrame()`).
    private unowned let codePredictor: Qwen3TTSCodePredictor

    init?(_ cp: Qwen3TTSCodePredictor) {
        let fused = cp.model.layers.map { $0.fusedLayer() }
        guard !fused.isEmpty, fused.allSatisfy({ $0 != nil }) else { return nil }
        let layers = fused.map { $0! }
        let first = layers[0]
        guard layers.allSatisfy({ $0.numHeads == first.numHeads && $0.numKvHeads == first.numKvHeads && $0.eps == first.eps
                && $0.ropeTheta == first.ropeTheta && $0.scale == first.scale }),
              first.numHeads % first.numKvHeads == 0 else { return nil }
        let heads = cp.lmHead.compactMap(Qwen3TTSFusedStep.Layer.Projection.init)
        guard heads.count == cp.lmHead.count else { return nil }
        // A quantized embedding table would need its own gather; keep those on the old path.
        guard cp.codecEmbedding.allSatisfy({ type(of: $0) == Embedding.self }) else { return nil }
        let hidden = first.inputNorm.dim(0)
        let dtype = first.inputNorm.dtype
        guard cp.model.norm.weight.dtype == dtype, layers.allSatisfy({ $0.inputNorm.dtype == dtype }),
              cp.codecEmbedding.allSatisfy({ $0.weight.dtype == dtype }),
              // rms_single_row's layout (256 threads × 4) and the attention kernel's 1024 threads.
              hidden == 1024, cp.config.numCodeGroups <= 32 else { return nil }
        let embedDim = cp.codecEmbedding[0].weight.dim(1)
        if cp.projection == nil { guard embedDim == hidden else { return nil } }
        guard embedDim % 256 == 0 else { return nil }
        self.codePredictor = cp
        self.layers = layers
        self.finalNorm = cp.model.norm.weight
        self.lmHeads = heads
        self.embeddings = cp.codecEmbedding.map { $0.weight }
        self.projection = cp.projection
        self.hidden = hidden
        self.intermediate = first.gate.scales.dim(0)
        self.vocab = heads[0].scales.dim(0)
        self.embedDim = embedDim
        self.numHeads = first.numHeads
        self.numKvHeads = first.numKvHeads
        self.capacity = cp.config.numCodeGroups
        self.eps = first.eps
        self.params = MLXArray([first.eps, first.ropeTheta, first.scale])
        self.positions = (0 ..< cp.config.numCodeGroups).map { MLXArray([Int32($0)]) }
        guard vocab % 256 == 0 else { return nil }
        eval(params)
        eval(positions)
    }

    private func emptyCache(_ dtype: DType) -> MLXArray {
        if let c = emptyCaches[dtype] { return c }
        let c = MLXArray.zeros([1, numKvHeads, capacity, Qwen3TTSFusedStep.headDim], dtype: dtype)
        eval(c)
        emptyCaches[dtype] = c
        return c
    }

    private func qmm(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray, bits: Int, groupSize: Int) -> MLXArray {
        quantizedMM(x, w, scales: s, biases: b, transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
    }

    /// One layer over `rows` new tokens at cache position `pos`. `x` is the
    /// residual stream, `normed` its input rmsnorm; returns the next
    /// residual and its rmsnorm under `nextNorm`.
    private func layerStep(_ x: MLXArray, _ normed: MLXArray, _ layer: Qwen3TTSFusedStep.Layer, nextNorm: MLXArray,
                           cache: inout (MLXArray, MLXArray), pos: Int, rows: Int) -> (MLXArray, MLXArray) {
        let dtype = x.dtype
        let c = layer.concat
        let (h, d) = (numHeads, numKvHeads)
        let qkv = qmm(normed, c.qkvW, c.qkvS, c.qkvB, bits: layer.bits, groupSize: layer.groupSize)
        let attn = Self.kernel(.attn, dtype: dtype, k: hidden, nq: h, nkv: d, rows: rows, cap: capacity)(
            [qkv, layer.qNorm, layer.kNorm, params, positions[pos], cache.0, cache.1],
            grid: (h * 1024, rows, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, rows, h * 128], cache.0.shape, cache.1.shape],
            outputDTypes: [dtype, dtype, dtype])
        cache = (attn[1], attn[2])
        let o = qmm(attn[0], layer.o.weight, layer.o.scales, layer.o.biases, bits: layer.bits, groupSize: layer.groupSize)
        return mlpAndNorm(x, o, layer, nextNorm: nextNorm, rows: rows)
    }

    /// All layers, then the sub-step's lm head on the last row.
    private func logits(_ x: MLXArray, _ normed: MLXArray, caches: inout [(MLXArray, MLXArray)],
                        pos: Int, rows: Int, step: Int) -> MLXArray {
        var (x, normed) = (x, normed)
        for (i, layer) in layers.enumerated() {
            let next = i + 1 < layers.count ? layers[i + 1].inputNorm : finalNorm
            (x, normed) = layerStep(x, normed, layer, nextNorm: next, cache: &caches[i], pos: pos, rows: rows)
        }
        let last = rows == 1 ? normed : normed[0..., (rows - 1)..., 0...]
        let head = lmHeads[step]
        return qmm(last, head.weight, head.scales, head.biases, bits: head.bits, groupSize: head.groupSize)
    }

    /// Exact or fast mode (see the top of the file); forwards to
    /// `Qwen3TTSModel.fusedCodePredictorExact`.
    static var exact: Bool {
        get { Qwen3TTSModel.fusedCodePredictorExact }
        set { Qwen3TTSModel.fusedCodePredictorExact = newValue }
    }

    /// One layer of an exact sub-step (one row at cache position `pos`).
    private func exactLayerStep(_ x: MLXArray, _ normed: MLXArray, _ layer: Qwen3TTSFusedStep.Layer, nextNorm: MLXArray,
                                cache: inout (MLXArray, MLXArray), pos: Int) -> (MLXArray, MLXArray) {
        let dtype = x.dtype
        let cacheType = cache.0.dtype
        let c = layer.concat
        let (h, d) = (numHeads, numKvHeads)
        let qkv = qmm(normed, c.qkvW, c.qkvS, c.qkvB, bits: layer.bits, groupSize: layer.groupSize)
        let glue = Self.kernel(.exactGlue, dtype: dtype, k: hidden, nq: h, nkv: d, cap: capacity,
                               kinCap: cache.0.dim(2), cacheType: cacheType)(
            [qkv, layer.qNorm, layer.kNorm, params, positions[pos], cache.0, cache.1],
            grid: ((h + 2 * d) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, h, 1, 128], [1, d, capacity, 128], [1, d, capacity, 128]],
            outputDTypes: [cacheType, cacheType, cacheType])
        cache = (glue[1], glue[2])
        let n = pos + 1
        let attended = MLXFast.scaledDotProductAttention(
            queries: glue[0], keys: glue[1][0..., 0..., ..<n, 0...], values: glue[2][0..., 0..., ..<n, 0...],
            scale: layer.scale, mask: nil)
        let o = qmm(attended.reshaped(1, 1, h * 128).asType(dtype), layer.o.weight, layer.o.scales, layer.o.biases,
                    bits: layer.bits, groupSize: layer.groupSize)
        return mlpAndNorm(x, o, layer, nextNorm: nextNorm, rows: 1)
    }

    /// Residual + post norm, MLP, residual + the next norm.
    private func mlpAndNorm(_ x: MLXArray, _ o: MLXArray, _ layer: Qwen3TTSFusedStep.Layer, nextNorm: MLXArray,
                            rows: Int) -> (MLXArray, MLXArray) {
        let dtype = x.dtype
        let c = layer.concat
        let addNorm = Self.kernel(.addNorm, dtype: dtype, k: hidden, rows: rows)(
            [x, o, layer.postNorm, params],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, hidden], [1, rows, hidden]], outputDTypes: [dtype, dtype])
        let gu = qmm(addNorm[1], c.guW, c.guS, c.guB, bits: layer.bits, groupSize: layer.groupSize)
        let act = Self.kernel(.silu, dtype: dtype, k: intermediate, rows: rows)(
            [gu], grid: (intermediate, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, intermediate]], outputDTypes: [dtype])[0]
        let down = qmm(act, layer.down.weight, layer.down.scales, layer.down.biases, bits: layer.bits, groupSize: layer.groupSize)
        let out = Self.kernel(.addRms, dtype: dtype, k: hidden, rows: rows)(
            [addNorm[0], down, nextNorm, params],
            grid: (256, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, hidden], [1, rows, hidden]], outputDTypes: [dtype, dtype])
        return (out[0], out[1])
    }

    struct Frame {
        /// Sub-codes 1…15, each (1, 1).
        let codes: [MLXArray]
        /// `code0Embed` + Σ codecEmbedding[i](codes[i]), summed in the
        /// activation type in that order (exactly the loop's own sum).
        let codecEmbedSum: MLXArray
    }

    /// The 15 sub-codes of one frame. `codeHidden` is the talker's last
    /// hidden state and `code0Embed` the talker embedding of code 0, both
    /// (1, 1, embedDim). `stepZeroCache` is the loop's code-predictor cache
    /// (exact mode runs step 0 through it). `sample` picks a token from
    /// (1, 1, vocab) logits when `greedy` is off.
    func frame(codeHidden: MLXArray, code0Embed: MLXArray, stepZeroCache: [any KVCache], greedy: Bool, pipeline: Bool,
               sample: (MLXArray) -> MLXArray) -> Frame {
        let dtype = code0Embed.dtype
        let exact = Self.exact
        var caches: [(MLXArray, MLXArray)]
        var x = MLXArray(0), normed = MLXArray(0)
        if exact {
            caches = []
        } else {
            let empty = emptyCache(dtype)
            caches = Array(repeating: (empty, empty), count: layers.count)
            x = concatenated([codeHidden.asType(dtype), code0Embed], axis: 1)
            if let projection { x = projection(x) }
            normed = MLXFast.rmsNorm(x, weight: layers[0].inputNorm, eps: eps)
        }
        var acc = code0Embed
        var codes: [MLXArray] = []
        let steps = embeddings.count
        for step in 0 ..< steps {
            let stepLogits: MLXArray
            if exact, step == 0 {
                for cache in stepZeroCache { _ = cache.trim(cache.offset) }
                let (logits, _, _) = codePredictor(
                    concatenated([codeHidden, code0Embed], axis: 1), cache: stepZeroCache, generationStep: 0)
                stepLogits = logits[0..., (-1)..., 0...]
                // The module caches' whole buffers (rows 0, 1 written); the glue kernel reads them in place.
                caches = stepZeroCache.map { let s = $0.innerState(); return (s[0], s[1]) }
            } else if exact {
                var (h, n) = (x, normed)
                for (i, layer) in layers.enumerated() {
                    let next = i + 1 < layers.count ? layers[i + 1].inputNorm : finalNorm
                    (h, n) = exactLayerStep(h, n, layer, nextNorm: next, cache: &caches[i], pos: step + 1)
                }
                let head = lmHeads[step]
                stepLogits = qmm(n, head.weight, head.scales, head.biases, bits: head.bits, groupSize: head.groupSize)
            } else {
                stepLogits = logits(x, normed, caches: &caches, pos: step == 0 ? 0 : step + 1, rows: step == 0 ? 2 : 1, step: step)
            }
            if greedy {
                let norm = projection == nil
                let out = Self.kernel(norm ? .argmaxEmbedNorm : .argmaxEmbed, dtype: dtype, k: embedDim, vocab: vocab,
                                      cacheType: stepLogits.dtype)(
                    [stepLogits, embeddings[step], acc, layers[0].inputNorm, params],
                    grid: (256, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: norm ? [[1, 1], [1, 1, embedDim], [1, 1, embedDim], [1, 1, embedDim]]
                        : [[1, 1], [1, 1, embedDim], [1, 1, embedDim]],
                    outputDTypes: norm ? [.uint32, dtype, dtype, dtype] : [.uint32, dtype, dtype])
                codes.append(out[0])
                acc = out[2]
                if step + 1 < steps {
                    if norm {
                        (x, normed) = (out[1], out[3])
                    } else {
                        x = projection!(out[1])
                        normed = MLXFast.rmsNorm(x, weight: layers[0].inputNorm, eps: eps)
                    }
                }
            } else {
                let token = sample(stepLogits)
                codes.append(token)
                let embed = codePredictor.codecEmbedding[step](token)
                acc = acc + embed
                if step + 1 < steps {
                    x = projection.map { $0(embed) } ?? embed
                    normed = MLXFast.rmsNorm(x, weight: layers[0].inputNorm, eps: eps)
                }
            }
            if pipeline { asyncEval(codes[codes.count - 1]) }
        }
        return Frame(codes: codes, codecEmbedSum: acc)
    }

    // MARK: - Kernels
    //
    // Baked per dtype/shape as macros under their own names, as in
    // `Qwen3TTSFusedStep` (no `template:` arguments: those rebuild the
    // name through a std::regex on every call).

    enum Kind: String {
        case attn = "qwen_cp_attn", addNorm = "qwen_cp_add_norm", addRms = "qwen_cp_add_rms", silu = "qwen_cp_silu"
        case argmaxEmbed = "qwen_cp_argmax_embed", argmaxEmbedNorm = "qwen_cp_argmax_embed_norm"
        case exactGlue = "qwen_cp_exact_glue"
    }

    private struct Key: Hashable {
        let kind: Kind, dtype: DType, k: Int, nq: Int, nkv: Int, rows: Int, cap: Int, vocab: Int, kinCap: Int, cacheType: DType
    }
    private nonisolated(unsafe) static var kernels: [Key: MLXFast.MLXFastKernel] = [:]
    private static let kernelsLock = NSLock()

    /// `cacheType`: the KV cache's type (exact glue) or the logits' (argmax).
    static func kernel(_ kind: Kind, dtype: DType, k: Int, nq: Int = 0, nkv: Int = 0, rows: Int = 1, cap: Int = 0,
                       vocab: Int = 0, kinCap: Int = 0, cacheType: DType? = nil) -> MLXFast.MLXFastKernel {
        let cacheType = cacheType ?? dtype
        let key = Key(kind: kind, dtype: dtype, k: k, nq: nq, nkv: nkv, rows: rows, cap: cap, vocab: vocab,
                      kinCap: kinCap, cacheType: cacheType)
        kernelsLock.lock(); defer { kernelsLock.unlock() }
        if let cached = kernels[key] { return cached }
        let defs = """
        using T = \(Qwen3TTSFusedStep.metalType(dtype));
        #define K \(k)
        #define NQ \(nq)
        #define NKV \(nkv)
        #define ROWS \(rows)
        #define CAP \(cap)
        #define VOCAB \(vocab)
        #define KIN_CAP \(kinCap)
        using TC = \(Qwen3TTSFusedStep.metalType(cacheType));

        """
        let name = "\(kind.rawValue)_\(dtype)_k\(k)_q\(nq)_kv\(nkv)_r\(rows)_c\(cap)_v\(vocab)_kc\(kinCap)_\(cacheType)"
        let built: MLXFast.MLXFastKernel
        switch kind {
        case .attn:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["qkv", "qnW", "knW", "params", "pos", "kin", "vin"],
                outputNames: ["out", "kout", "vout"], source: attnSource, header: defs)
        case .exactGlue:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["qkv", "qnW", "knW", "params", "pos", "kin", "vin"],
                outputNames: ["q", "kout", "vout"], source: exactGlueSource, header: defs)
        case .addNorm:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["a", "b", "w", "params"], outputNames: ["sum", "normed"],
                source: addNormSource, header: defs)
        case .addRms:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["a", "b", "w", "params"], outputNames: ["sum", "normed"],
                source: addRmsSource, header: defs)
        case .silu:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["gu"], outputNames: ["out"], source: siluSource, header: defs)
        case .argmaxEmbed:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["logits", "table", "acc", "w", "params"],
                outputNames: ["token", "emb", "accOut"], source: argmaxEmbedSource(norm: false), header: defs)
        case .argmaxEmbedNorm:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["logits", "table", "acc", "w", "params"],
                outputNames: ["token", "emb", "accOut", "normed"], source: argmaxEmbedSource(norm: true), header: defs)
        }
        kernels[key] = built
        return built
    }

    /// One 1024-thread group per (q head, row). Phase 1 (threads in 128-wide
    /// segments): segment 0 makes this row's q head, segments 1+2j / 2+2j
    /// the k / v head of new row j for this group's kv head — head rmsnorm
    /// and RoPE exactly as `Qwen3TTSFusedStep`'s qkv_post, rounded to T.
    /// Phase 2: the first group of each kv head (row 0) writes the cache
    /// forward (rows < pos copied, the new rows appended). Phase 3: MLX's
    /// `sdpa_vector` reproduced line for line (32 simdgroups, one key each;
    /// row r sees keys 0 … pos + r).
    static let attnSource = """
        threadgroup float hs[1 + 2 * ROWS][128];
        threadgroup float red[1 + 2 * ROWS][4];
        threadgroup float max_scores[32];
        threadgroup float sum_exp_scores[32];
        threadgroup float outputs[32 * 32];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint qh = threadgroup_position_in_grid.x;
        const uint r = threadgroup_position_in_grid.y;
        const uint kvh = qh / (NQ / NKV);
        const int p0 = pos[0];
        const float eps = params[0];
        const float theta = params[1];
        const float scale = params[2];
        constexpr uint W = (NQ + 2 * NKV) * 128;

        const uint seg = tid / 128;
        const uint e = tid % 128;
        const bool active = seg < 1 + 2 * ROWS;
        // kind 0 = q, 1 = k, 2 = v
        const uint kind = seg == 0 ? 0 : ((seg - 1) % 2 == 0 ? 1 : 2);
        const uint row = seg == 0 ? r : (seg - 1) / 2;
        const uint col = kind == 0 ? qh * 128 : (kind == 1 ? (NQ + kvh) * 128 : (NQ + NKV + kvh) * 128);
        float v0 = 0;
        if (active) {
            v0 = float(qkv[row * W + col + e]);
            const float part = simd_sum(v0 * v0);
            if (lane == 0) red[seg][(tid % 128) / 32] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float n0 = 0;
        if (active) {
            if (kind == 2) {
                hs[seg][e] = float(T(v0));
            } else {
                const float inv = metal::rsqrt((red[seg][0] + red[seg][1] + red[seg][2] + red[seg][3]) / 128.0f + eps);
                n0 = v0 * inv * float(kind == 0 ? qnW[e] : knW[e]);
                hs[seg][e] = n0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rot = 0;
        if (active && kind != 2) {
            const uint i = e & 63;
            const float freq = metal::pow(theta, -float(i) / 64.0f);
            const float angle = float(p0 + int(row)) * freq;
            const float c = metal::cos(angle), s = metal::sin(angle);
            const float a = hs[seg][i], b = hs[seg][i + 64];
            rot = float(e < 64 ? T(a * c - b * s) : T(b * c + a * s));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && kind != 2) hs[seg][e] = rot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cache forward: one group per kv head does it.
        const uint cbase = kvh * CAP * 128;
        if (r == 0 && qh % (NQ / NKV) == 0) {
            for (int idx = int(tid); idx < p0 * 128; idx += 1024) {
                kout[cbase + idx] = kin[cbase + idx];
                vout[cbase + idx] = vin[cbase + idx];
            }
            if (tid < ROWS * 128) {
                const uint j = tid / 128;
                kout[cbase + (p0 + j) * 128 + e] = T(hs[1 + 2 * j][e]);
                vout[cbase + (p0 + j) * 128 + e] = T(hs[2 + 2 * j][e]);
            }
        }

        // sdpa_vector (BN = BD = 32, 4 elements per lane).
        const int N = p0 + int(r) + 1;
        float q[4], k[4], o[4];
        for (int i = 0; i < 4; i++) q[i] = static_cast<float>(scale) * hs[0][lane * 4 + i];
        for (int i = 0; i < 4; i++) o[i] = 0;
        float max_score = -metal::numeric_limits<float>::max();
        float sum_exp_score = 0;
        for (int i = int(sg); i < N; i += 32) {
            const bool cached = i < p0;
            for (int j = 0; j < 4; j++) {
                k[j] = cached ? float(kin[cbase + i * 128 + lane * 4 + j]) : hs[1 + 2 * (i - p0)][lane * 4 + j];
            }
            float score = 0;
            for (int j = 0; j < 4; j++) score += q[j] * k[j];
            score = simd_sum(score);
            const float new_max = max(max_score, score);
            const float factor = fast::exp(max_score - new_max);
            const float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < 4; j++) {
                const float vj = cached ? float(vin[cbase + i * 128 + lane * 4 + j]) : hs[2 + 2 * (i - p0)][lane * 4 + j];
                o[j] = o[j] * factor + exp_score * vj;
            }
        }
        if (lane == 0) {
            max_scores[sg] = max_score;
            sum_exp_scores[sg] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[lane];
        const float new_max = simd_max(max_score);
        const float factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[lane] * factor);
        for (int i = 0; i < 4; i++) {
            outputs[lane * 32 + sg] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[sg * 32 + lane] * factor);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            for (int i = 0; i < 4; i++) out[(r * NQ + qh) * 128 + sg * 4 + i] = static_cast<T>(o[i]);
        }
        """

    /// Exact mode's glue after the q/k/v matmul: `Qwen3TTSFusedStep`'s
    /// qkv_post (per-head q/k rmsnorm, RoPE at `pos`, rounded to T) with q
    /// written in the cache type (what SDPA would cast it to) and each kv
    /// head's cache carried forward — rows < pos copied from `kin` (row
    /// stride KIN_CAP), the new row at `pos` (stride CAP) — as
    /// KVCacheSimple's update stores them. One 128-thread group per head.
    static let exactGlueSource = """
        threadgroup float red[4];
        threadgroup float hs[128];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint head = threadgroup_position_in_grid.x;       // 0..<NQ+2·NKV
        const int p0 = pos[0];
        const float eps = params[0];
        const float theta = params[1];
        const float v0 = float(qkv[head * 128 + tid]);
        if (head >= NQ) {
            const bool isK = head < NQ + NKV;
            const uint kvh = isK ? head - NQ : head - NQ - NKV;
            device TC* dst = isK ? kout : vout;
            const device TC* src = isK ? kin : vin;
            for (int r = 0; r < p0; r++) dst[(kvh * CAP + r) * 128 + tid] = src[(kvh * KIN_CAP + r) * 128 + tid];
            if (!isK) { vout[(kvh * CAP + p0) * 128 + tid] = TC(T(v0)); return; }
        }
        const bool isQ = head < NQ;
        const float part = simd_sum(v0 * v0);
        if (lane == 0) red[simd] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = metal::rsqrt((red[0] + red[1] + red[2] + red[3]) / 128.0f + eps);
        const float n0 = v0 * inv * float(isQ ? qnW[tid] : knW[tid]);
        hs[tid] = n0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint i = tid & 63;
        const float freq = metal::pow(theta, -float(i) / 64.0f);
        const float angle = float(p0) * freq;
        const float c = metal::cos(angle), s = metal::sin(angle);
        const float a = hs[i], b = hs[i + 64];
        const T r = tid < 64 ? T(a * c - b * s) : T(b * c + a * s);
        if (isQ) q[head * 128 + tid] = TC(r);
        else kout[((head - NQ) * CAP + p0) * 128 + tid] = TC(r);
        """

    /// `Qwen3TTSFusedStep.addNormSource` (sum = a + b; normed = rmsnorm(sum)·w),
    /// one 256-thread group per row.
    static let addNormSource = """
        threadgroup float red[8];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint base = threadgroup_position_in_grid.y * K;
        float vals[K / 256];
        float ss = 0;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < K / 256; ++j) {
            const int i = tid + 256 * j;
            const float v = float(a[base + i]) + float(b[base + i]);
            vals[j] = v; ss += v * v;
            sum[base + i] = T(v);
        }
        const float part = simd_sum(ss);
        if (lane == 0) red[simd] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 8; ++r) total += red[r];
        const float inv = metal::rsqrt(total / float(K) + params[0]);
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < K / 256; ++j) {
            const int i = tid + 256 * j;
            normed[base + i] = T(vals[j] * inv * float(w[i]));
        }
        """

    /// sum = a + b in T (MLX's binary add); normed = MLX's `rms_single_row`
    /// on sum (256 threads × 4 consecutive elements, K = 1024).
    static let addRmsSource = """
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[32];
        const uint lid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sgid = simdgroup_index_in_threadgroup;
        const uint base = threadgroup_position_in_grid.y * K + lid * 4;
        T s[4];
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            s[i] = a[base + i] + b[base + i];
            sum[base + i] = s[i];
            float xi = s[i];
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (sgid == 0) local_sums[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) local_sums[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            acc = simd_sum(local_sums[lane]);
            if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / uint(K) + params[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 4; i++) {
            normed[base + i] = w[lid * 4 + i] * static_cast<T>(s[i] * local_inv_mean[0]);
        }
        """

    /// `Qwen3TTSFusedStep.siluMulSource` over rows.
    static let siluSource = """
        const uint i = thread_position_in_grid.x;
        const uint r = thread_position_in_grid.y;
        const float g = float(gu[r * 2 * K + i]);
        out[r * K + i] = T(g / (1.0f + metal::exp(-g)) * float(gu[r * 2 * K + i + K]));
        """

    /// Greedy sub-code: argmax over VOCAB logits (lowest index on ties, as
    /// MLX's argmax), the token's row of `table` (the next sub-step's
    /// input), `acc + row` in T (the talker's next codec embedding, summed
    /// as the loop sums it) and, with `norm`, the row's rmsnorm under `w`
    /// (layer 0's input norm) as MLX's `rms_single_row`. One 256-thread group.
    static func argmaxEmbedSource(norm: Bool) -> String {
        let normPart = norm ? """
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc2 = 0;
            for (int i = 0; i < 4; i++) { float xi = vals[i]; acc2 += xi * xi; }
            acc2 = simd_sum(acc2);
            if (sgid == 0) local_sums[lane] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) local_sums[sgid] = acc2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0) {
                acc2 = simd_sum(local_sums[lane]);
                if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc2 / uint(K) + params[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < 4; i++) {
                normed[lid * 4 + i] = w[lid * 4 + i] * static_cast<T>(vals[i] * local_inv_mean[0]);
            }
            """ : ""
        return """
            threadgroup float best_v[8];
            threadgroup uint best_i[8];
            threadgroup uint chosen[1];
            const uint lid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint sgid = simdgroup_index_in_threadgroup;
            float bv = float(logits[lid]);
            uint bi = lid;
            for (uint j = lid + 256; j < VOCAB; j += 256) {
                const float v = float(logits[j]);
                if (v > bv) { bv = v; bi = j; }
            }
            float m = simd_max(bv);
            uint mi = simd_min(bv == m ? bi : 0xffffffffu);
            if (lane == 0) { best_v[sgid] = m; best_i[sgid] = mi; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0) {
                const float v = lane < 8 ? best_v[lane] : -INFINITY;
                const uint idx = lane < 8 ? best_i[lane] : 0xffffffffu;
                const float mm = simd_max(v);
                const uint ii = simd_min(v == mm ? idx : 0xffffffffu);
                if (lane == 0) { chosen[0] = ii; token[0] = ii; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint t = chosen[0];
            T vals[K / 256];
            for (int i = 0; i < K / 256; i++) {
                const uint e = lid * (K / 256) + i;
                vals[i] = table[t * K + e];
                emb[e] = vals[i];
                accOut[e] = acc[e] + vals[i];
            }
            \(normPart)
            """
    }
}

extension Qwen3TTSCodePredictor {
    /// The fused frame for this predictor, or nil when its weights are not
    /// in a layout the kernels take (then the per-layer path runs).
    func fusedFrame() -> Qwen3TTSFusedCodePredictor? {
        if fusedFrameResolved { return fusedFrameCache }
        fusedFrameResolved = true
        fusedFrameCache = Qwen3TTSFusedCodePredictor(self)
        return fusedFrameCache
    }
}
