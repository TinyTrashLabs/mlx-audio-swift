import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// Fused single-token decoder step (iOS speed work, 2026-09-09).
//
// On an iPhone the Qwen3-TTS loop is bound by kernel-launch latency, not
// arithmetic: one decoder layer on one token is ~22 MLX launches, and a
// frame runs 103 of them (28 talker layers + 5 code-predictor layers × 15
// sub-steps). This path does the same layer in four hand-written Metal
// kernels plus the (unchanged) KV cache update and MLXFast SDPA:
//
//   qkv     rmsnorm → q/k/v 4-bit matvec → q/k head rmsnorm → RoPE
//   mv_res  o_proj matvec + residual
//   gateup  rmsnorm → gate/up matvec → silu(gate)·up
//   mv_res  down matvec + residual
//
// The matvec is one simdgroup per output row over x held in threadgroup
// memory, with the affine 4-bit groups (64 wide, scale·q + bias) unpacked
// in-register; lane l owns elements [c·1024 + 32·l, +32) of chunk c, so
// each lane's slice lies in exactly one quantization group. Everything
// accumulates in float32 and is written back in the activation dtype.
// Requires the mobile checkpoint's layout: 4-bit affine, group size 64,
// no projection biases, head_dim 128, widths that are multiples of 1024.
enum Qwen3TTSFusedStep {
    /// One layer's tensors, in the order the kernels take them.
    struct Layer {
        struct Projection {
            let weight: MLXArray, scales: MLXArray, biases: MLXArray
            init?(_ linear: Linear) {
                guard let q = linear as? QuantizedLinear, q.bits == 4, q.groupSize == 64,
                      q.mode == .affine, q.bias == nil, let biases = q.biases else { return nil }
                weight = q.weight; scales = q.scales; self.biases = biases
            }
        }
        let inputNorm: MLXArray, postNorm: MLXArray, qNorm: MLXArray, kNorm: MLXArray
        let q: Projection, k: Projection, v: Projection, o: Projection
        let gate: Projection, up: Projection, down: Projection
        let numHeads: Int, numKvHeads: Int, headDim: Int
        let eps: Float, ropeTheta: Float, scale: Float

        init?(inputNorm: RMSNorm, postNorm: RMSNorm, qNorm: RMSNorm, kNorm: RMSNorm,
              q: Linear, k: Linear, v: Linear, o: Linear, gate: Linear, up: Linear, down: Linear,
              numHeads: Int, numKvHeads: Int, headDim: Int, eps: Float, ropeTheta: Float, scale: Float) {
            guard headDim == Qwen3TTSFusedStep.headDim,
                  let q = Projection(q), let k = Projection(k), let v = Projection(v), let o = Projection(o),
                  let gate = Projection(gate), let up = Projection(up), let down = Projection(down) else { return nil }
            let hidden = inputNorm.weight.dim(0)
            let intermediate = gate.scales.dim(0)
            guard hidden % Qwen3TTSFusedStep.chunk == 0, intermediate % Qwen3TTSFusedStep.chunk == 0,
                  q.scales.dim(0) == numHeads * headDim, k.scales.dim(0) == numKvHeads * headDim,
                  o.scales.dim(0) == hidden, down.scales.dim(0) == hidden,
                  inputNorm.weight.dtype == q.scales.dtype else { return nil }
            self.inputNorm = inputNorm.weight; self.postNorm = postNorm.weight
            self.qNorm = qNorm.weight; self.kNorm = kNorm.weight
            self.q = q; self.k = k; self.v = v; self.o = o
            self.gate = gate; self.up = up; self.down = down
            self.numHeads = numHeads; self.numKvHeads = numKvHeads; self.headDim = headDim
            self.eps = eps; self.ropeTheta = ropeTheta; self.scale = scale
        }
    }

    static let headDim = 128
    static let chunk = 1024
    /// Threads per group. qkv needs a whole head (128 rows) per group for
    /// the head norm, so it runs 32 simdgroups of 4 rows; the others run
    /// four simdgroups (16 rows) per group so enough groups are in flight.
    static let qkvThreads = 512
    static let matvecThreads = 128
    static let gateUpThreads = 128
    static var matvecRowsPerGroup: Int { matvecThreads / 8 }
    static var gateUpRowsPerGroup: Int { gateUpThreads / 8 }

    /// `x` is (1, 1, hidden). Returns the layer output in the same shape.
    static func run(_ x: MLXArray, layer: Layer, cache: (any KVCache)?) -> MLXArray {
        let dtype = x.dtype
        let paramType = layer.inputNorm.dtype
        let hidden = x.dim(2)
        let intermediate = layer.gate.scales.dim(0)
        let (h, d) = (layer.numHeads, layer.numKvHeads)
        let params = MLXArray([layer.eps, layer.ropeTheta])
        let position = MLXArray([Int32(cache?.offset ?? 0)])

        let qkv = kernel(.qkv, dtype: dtype, paramType: paramType, k: hidden, nt: qkvThreads, nq: h, nkv: d)(
            [x, layer.inputNorm,
             layer.q.weight, layer.q.scales, layer.q.biases,
             layer.k.weight, layer.k.scales, layer.k.biases,
             layer.v.weight, layer.v.scales, layer.v.biases,
             layer.qNorm, layer.kNorm, params, position],
            grid: ((h + 2 * d) * qkvThreads, 1, 1), threadGroup: (qkvThreads, 1, 1),
            outputShapes: [[1, h, 1, headDim], [1, d, 1, headDim], [1, d, 1, headDim]],
            outputDTypes: [dtype, dtype, dtype])
        var (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: layer.scale, mask: nil)

        let afterAttention = matvecResidual(attended, layer.o, residual: x, inputWidth: h * headDim, rows: hidden)
        let activated = kernel(.gateUp, dtype: dtype, paramType: paramType, k: hidden, nt: gateUpThreads)(
            [afterAttention, layer.postNorm,
             layer.gate.weight, layer.gate.scales, layer.gate.biases,
             layer.up.weight, layer.up.scales, layer.up.biases, params],
            grid: (intermediate / gateUpRowsPerGroup * gateUpThreads, 1, 1), threadGroup: (gateUpThreads, 1, 1),
            outputShapes: [[1, 1, intermediate]], outputDTypes: [dtype])[0]
        return matvecResidual(activated, layer.down, residual: afterAttention, inputWidth: intermediate, rows: hidden)
    }

    private static func matvecResidual(_ x: MLXArray, _ p: Layer.Projection, residual: MLXArray,
                                       inputWidth: Int, rows: Int) -> MLXArray {
        // `x` may not share the residual's dtype (on the phone MLX's SDPA
        // hands back float32 for bf16 inputs), so the kernel carries both.
        kernel(.matvec, dtype: residual.dtype, paramType: p.scales.dtype, k: inputWidth, nt: matvecThreads, xType: x.dtype)(
            [x, p.weight, p.scales, p.biases, residual],
            grid: (rows / matvecRowsPerGroup * matvecThreads, 1, 1), threadGroup: (matvecThreads, 1, 1),
            outputShapes: [residual.shape], outputDTypes: [residual.dtype])[0]
    }

    // MARK: - Kernel cache
    //
    // No `template:` arguments: MLX's custom-kernel builder rebuilds the
    // kernel name from them on every call, constructing a std::regex to do
    // it, which measured ~43 µs per call on an M-series Mac — four times
    // the launch it was meant to save. Each dtype/width combination is
    // instead baked into the source as `using`/`constexpr` definitions
    // under its own kernel name, and the kernel object cached.

    enum Kind: String { case qkv = "qwen_qkv", matvec = "qwen_mv_res", gateUp = "qwen_gateup" }

    private struct Key: Hashable {
        let kind: Kind, dtype: DType, paramType: DType, k: Int, nt: Int, nq: Int, nkv: Int, xType: DType
    }

    private nonisolated(unsafe) static var kernels: [Key: MLXFast.MLXFastKernel] = [:]
    private static let kernelsLock = NSLock()

    static func metalType(_ dtype: DType) -> String {
        switch dtype {
        case .float32: return "float"
        case .float16: return "half"
        case .bfloat16: return "bfloat16_t"
        default: fatalError("Qwen3TTSFusedStep: unsupported activation dtype \(dtype)")
        }
    }

    /// `dtype` is the output/residual/activation type, `xType` the type of
    /// the vector being multiplied (matvec only; defaults to `dtype`).
    static func kernel(_ kind: Kind, dtype: DType, paramType: DType, k: Int, nt: Int, nq: Int = 0, nkv: Int = 0,
                       xType: DType? = nil) -> MLXFast.MLXFastKernel {
        let xType = xType ?? dtype
        let key = Key(kind: kind, dtype: dtype, paramType: paramType, k: k, nt: nt, nq: nq, nkv: nkv, xType: xType)
        kernelsLock.lock(); defer { kernelsLock.unlock() }
        if let cached = kernels[key] { return cached }
        // Metal allows no program-scope constexpr variables; macros serve
        // as the array sizes and template arguments instead.
        let defs = """
        using T = \(metalType(dtype));
        using TX = \(metalType(xType));
        using S = \(metalType(paramType));
        #define K \(k)
        #define NT \(nt)
        #define NQ \(nq)
        #define NKV \(nkv)

        """
        let name = "\(kind.rawValue)_\(dtype)_x\(xType)_\(paramType)_k\(k)_nt\(nt)_q\(nq)_kv\(nkv)"
        let built: MLXFast.MLXFastKernel
        switch kind {
        case .qkv:
            built = MLXFast.metalKernel(
                name: name,
                inputNames: ["x", "inW", "wq", "sq", "bq", "wk", "sk", "bk", "wv", "sv", "bv", "qnW", "knW", "params", "pos"],
                outputNames: ["q", "k", "v"], source: qkvSource, header: defs + header)
        case .matvec:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["x", "w", "sc", "bi", "resid"], outputNames: ["out"],
                source: matvecSource, header: defs + header)
        case .gateUp:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["x", "postW", "wg", "sg", "bg", "wu", "su", "bu", "params"],
                outputNames: ["out"], source: gateUpSource, header: defs + header)
        }
        kernels[key] = built
        return built
    }

    // MARK: - Metal

    /// Shared device code: the quantized row dot product and reductions.
    ///
    /// x lives in threadgroup memory transposed within each 1024-wide chunk
    /// (element c·1024 + 32·l + i is stored at c·1024 + 32·i + l), so the 32
    /// lanes of a simdgroup read consecutive words at every step instead of
    /// a 32-way bank conflict. Rows go in batches of R with all their weight
    /// loads issued before the arithmetic, which is what keeps enough bytes in
    /// flight for the matvec to run at memory speed.
    static let header = """
    // Sum over the whole NT-thread group; `red` is NT/32 threadgroup floats.
    inline float qwen_tg_sum(float v, threadgroup float* red, uint lane, uint simd) {
        v = simd_sum(v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[simd] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < NT / 32; ++i) total += red[i];
        return total;
    }

    // Element index held at transposed slot `t` (see the Swift comment).
    inline int qwen_elem(int t) {
        const int c = t >> 10, r = t & 1023;
        return (c << 10) + ((r & 31) << 5) + (r >> 5);
    }

    // R rows (row0, row0+1, …) of a [rows, K/8] uint32 4-bit affine matrix
    // (group 64, scale·q + bias) dotted with the transposed xs; the reduced
    // values land in out[0..R) on every lane.
    template <int R>
    inline void qwen_rows_dot(const device uint* w, const device S* scales, const device S* biases,
                              uint row0, threadgroup const float* xs, uint lane, thread float* out) {
        constexpr int wordsPerRow = K / 8;
        constexpr int groupsPerRow = K / 64;
        float acc[R];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < R; ++r) acc[r] = 0;
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < K / 1024; ++c) {
            const int e0 = c * 1024 + 32 * int(lane);
            const int g = e0 / 64;
            uint4 packed[R];
            float s[R], b[R];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < R; ++r) {
                const uint row = row0 + r;
                packed[r] = *((const device uint4*)(w + row * wordsPerRow + e0 / 8));
                s[r] = float(scales[row * groupsPerRow + g]);
                b[r] = float(biases[row * groupsPerRow + g]);
            }
            float sx = 0;
            float sq[R];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < R; ++r) sq[r] = 0;
            threadgroup const float* xp = xs + c * 1024 + lane;
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < 4; ++j) {
                _Pragma("clang loop unroll(full)")
                for (int n = 0; n < 8; ++n) {
                    const float xv = xp[32 * (8 * j + n)];
                    sx += xv;
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < R; ++r) {
                        const uint wv = j == 0 ? packed[r].x : j == 1 ? packed[r].y : j == 2 ? packed[r].z : packed[r].w;
                        sq[r] += xv * float((wv >> (4 * n)) & 0xfu);
                    }
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < R; ++r) acc[r] += s[r] * sq[r] + b[r] * sx;
        }
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < R; ++r) out[r] = simd_sum(acc[r]);
    }

    // xs = transposed(rmsnorm(x) · weight) in float, over the whole group.
    // Returns with xs complete (barrier included).
    inline void qwen_load_normed(const device T* x, const device S* weight, float eps,
                                 threadgroup float* xs, threadgroup float* red,
                                 uint tid, uint lane, uint simd) {
        float ss = 0;
        for (int t = tid; t < K; t += NT) { const float v = float(x[qwen_elem(t)]); xs[t] = v; ss += v * v; }
        const float inv = metal::rsqrt(qwen_tg_sum(ss, red, lane, simd) / float(K) + eps);
        for (int t = tid; t < K; t += NT) xs[t] = xs[t] * inv * float(weight[qwen_elem(t)]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // xs = transposed(x) in float; barrier included.
    template <typename X>
    inline void qwen_load(const device X* x, threadgroup float* xs, uint tid) {
        for (int t = tid; t < K; t += NT) xs[t] = float(x[qwen_elem(t)]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """

    /// rmsnorm → q/k/v projections → q/k head norm → RoPE. One NT-thread
    /// group per head (NQ q heads, then NKV k heads, then NKV v heads); the
    /// NT/32 simdgroups share the head's 128 rows four at a time.
    static let qkvSource = """
        threadgroup float xs[K];
        threadgroup float hs[128];
        threadgroup float red[NT / 32];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        const float eps = params[0];
        const float theta = params[1];

        qwen_load_normed(x, inW, eps, xs, red, tid, lane, simd);

        const device uint* w; const device S* sc; const device S* bi; device T* out; const device S* nw;
        uint head; bool rotate = true;
        if (block < NQ) { head = block; w = wq; sc = sq; bi = bq; out = q; nw = qnW; }
        else if (block < NQ + NKV) { head = block - NQ; w = wk; sc = sk; bi = bk; out = k; nw = knW; }
        else { head = block - NQ - NKV; w = wv; sc = sv; bi = bv; out = v; rotate = false; nw = qnW; }

        // 128 rows per head over NT/32 simdgroups, four rows each pass.
        _Pragma("clang loop unroll(full)")
        for (uint local = simd * 4; local < 128; local += (NT / 32) * 4) {
            float dots[4];
            qwen_rows_dot<4>(w, sc, bi, head * 128 + local, xs, lane, dots);
            if (lane == 0) { hs[local] = dots[0]; hs[local + 1] = dots[1]; hs[local + 2] = dots[2]; hs[local + 3] = dots[3]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!rotate) {
            if (tid < 128) out[head * 128 + tid] = T(hs[tid]);
            return;
        }
        // Per-head rmsnorm (q_norm / k_norm), then rotate-half RoPE at `pos`.
        const float hv = tid < 128 ? hs[tid] : 0.0f;
        const float inv = metal::rsqrt(qwen_tg_sum(hv * hv, red, lane, simd) / 128.0f + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) hs[tid] = hv * inv * float(nw[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const float freq = metal::pow(theta, -float(tid) / 64.0f);
            const float angle = float(pos[0]) * freq;
            const float c = metal::cos(angle), s = metal::sin(angle);
            const float a = hs[tid], b = hs[tid + 64];
            out[head * 128 + tid] = T(a * c - b * s);
            out[head * 128 + tid + 64] = T(b * c + a * s);
        }
        """

    /// out = residual + W·x for a [rows, K] matrix; NT/8 rows per group,
    /// four per simdgroup.
    static let matvecSource = """
        threadgroup float xs[K];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        qwen_load(x, xs, tid);
        const uint row0 = block * (NT / 8) + simd * 4;
        float dots[4];
        qwen_rows_dot<4>(w, sc, bi, row0, xs, lane, dots);
        if (lane == 0) {
            out[row0] = T(float(resid[row0]) + dots[0]);
            out[row0 + 1] = T(float(resid[row0 + 1]) + dots[1]);
            out[row0 + 2] = T(float(resid[row0 + 2]) + dots[2]);
            out[row0 + 3] = T(float(resid[row0 + 3]) + dots[3]);
        }
        """

    /// rmsnorm → silu(gate·x) · (up·x); NT/8 output rows per group, four
    /// per simdgroup (a gate row and its up row each).
    static let gateUpSource = """
        threadgroup float xs[K];
        threadgroup float red[NT / 32];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        qwen_load_normed(x, postW, params[0], xs, red, tid, lane, simd);
        const uint row0 = block * (NT / 8) + simd * 4;
        float g[4], u[4];
        qwen_rows_dot<4>(wg, sg, bg, row0, xs, lane, g);
        qwen_rows_dot<4>(wu, su, bu, row0, xs, lane, u);
        if (lane == 0) {
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 4; ++r) out[row0 + r] = T(g[r] / (1.0f + metal::exp(-g[r])) * u[r]);
        }
        """
}

extension TalkerDecoderLayer {
    /// The fused-step view of this layer, or nil when its weights are not in
    /// the layout the kernels take (then the module path runs).
    func fusedLayer() -> Qwen3TTSFusedStep.Layer? {
        Qwen3TTSFusedStep.Layer(
            inputNorm: inputLayernorm, postNorm: postAttentionLayernorm,
            qNorm: selfAttn.qNorm, kNorm: selfAttn.kNorm,
            q: selfAttn.qProj, k: selfAttn.kProj, v: selfAttn.vProj, o: selfAttn.oProj,
            gate: mlp.gateProj, up: mlp.upProj, down: mlp.downProj,
            numHeads: selfAttn.numHeads, numKvHeads: selfAttn.numKvHeads, headDim: selfAttn.headDim,
            eps: inputLayernorm.eps, ropeTheta: selfAttn.ropeTheta, scale: selfAttn.scale)
    }
}

extension CodePredictorDecoderLayer {
    func fusedLayer() -> Qwen3TTSFusedStep.Layer? {
        Qwen3TTSFusedStep.Layer(
            inputNorm: inputLayernorm, postNorm: postAttentionLayernorm,
            qNorm: selfAttn.qNorm, kNorm: selfAttn.kNorm,
            q: selfAttn.qProj, k: selfAttn.kProj, v: selfAttn.vProj, o: selfAttn.oProj,
            gate: mlp.gateProj, up: mlp.upProj, down: mlp.downProj,
            numHeads: selfAttn.numHeads, numKvHeads: selfAttn.numKvHeads, headDim: selfAttn.headDim,
            eps: inputLayernorm.eps, ropeTheta: selfAttn.ropeTheta, scale: selfAttn.scale)
    }
}
