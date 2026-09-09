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

    /// `.hybrid` (default): MLX's quantized matmul for every projection, with
    /// q/k/v and gate/up concatenated at load so each costs one launch, and
    /// custom kernels only for the glue (head norm + RoPE + split, residual +
    /// norm, silu·up) — ~12 launches per layer, all at the launch floor.
    /// `.customMatvec`: the four hand-written matvec kernels (measured on the
    /// iPhone 15 Pro at 2–4× MLX's matmul cost; kept for the bench).
    enum Mode { case hybrid, customMatvec }
    nonisolated(unsafe) static var mode: Mode = .hybrid

    /// Concatenated projections, built once per layer (keyed by the q weight)
    /// so one matmul launch serves q/k/v and one serves gate/up.
    struct Concat { let qkvW: MLXArray, qkvS: MLXArray, qkvB: MLXArray, guW: MLXArray, guS: MLXArray, guB: MLXArray }
    private nonisolated(unsafe) static var concats: [ObjectIdentifier: Concat] = [:]
    private static let concatsLock = NSLock()

    static func concat(for layer: Layer) -> Concat {
        let key = ObjectIdentifier(layer.q.weight)
        concatsLock.lock(); defer { concatsLock.unlock() }
        if let c = concats[key] { return c }
        let c = Concat(
            qkvW: concatenated([layer.q.weight, layer.k.weight, layer.v.weight], axis: 0),
            qkvS: concatenated([layer.q.scales, layer.k.scales, layer.v.scales], axis: 0),
            qkvB: concatenated([layer.q.biases, layer.k.biases, layer.v.biases], axis: 0),
            guW: concatenated([layer.gate.weight, layer.up.weight], axis: 0),
            guS: concatenated([layer.gate.scales, layer.up.scales], axis: 0),
            guB: concatenated([layer.gate.biases, layer.up.biases], axis: 0))
        eval(c.qkvW, c.qkvS, c.qkvB, c.guW, c.guS, c.guB)
        // Hand the layer's own tensors row-slices of the concatenations (views
        // over the same buffers) so the originals are freed: otherwise the
        // fused path carries +5 MB per layer (+165 MB on the 0.6B), which on
        // the iPhone 15 Pro was the difference between surviving the ~1.3 GB
        // transient of a 2 s decode chunk on long text and being jetsammed
        // (2026-09-09). The module path (prefill) keeps working on the views,
        // and the cache key (the q weight's identity) is unchanged.
        let (nq, nk) = (layer.q.scales.dim(0), layer.k.scales.dim(0))
        let nv = layer.v.scales.dim(0)
        let ni = layer.gate.scales.dim(0)
        func adopt(_ p: Layer.Projection, _ w: MLXArray, _ sc: MLXArray, _ b: MLXArray, _ range: Range<Int>) {
            let (ws, ss, bs) = (w[range], sc[range], b[range])
            eval(ws, ss, bs)
            p.weight._updateInternal(ws); p.scales._updateInternal(ss); p.biases._updateInternal(bs)
        }
        adopt(layer.q, c.qkvW, c.qkvS, c.qkvB, 0 ..< nq)
        adopt(layer.k, c.qkvW, c.qkvS, c.qkvB, nq ..< nq + nk)
        adopt(layer.v, c.qkvW, c.qkvS, c.qkvB, nq + nk ..< nq + nk + nv)
        adopt(layer.gate, c.guW, c.guS, c.guB, 0 ..< ni)
        adopt(layer.up, c.guW, c.guS, c.guB, ni ..< 2 * ni)
        concats[key] = c
        return c
    }

    private static func qmm(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray) -> MLXArray {
        quantizedMM(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: 4, mode: .affine)
    }

    /// The hybrid step (see `Mode`).
    static func runHybrid(_ x: MLXArray, layer: Layer, cache: (any KVCache)?) -> MLXArray {
        let dtype = x.dtype
        let paramType = layer.inputNorm.dtype
        let hidden = x.dim(2)
        let intermediate = layer.gate.scales.dim(0)
        let (h, d) = (layer.numHeads, layer.numKvHeads)
        let c = concat(for: layer)
        let params = MLXArray([layer.eps, layer.ropeTheta])
        let position = MLXArray([Int32(cache?.offset ?? 0)])

        let normed = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: layer.eps)
        let qkv = qmm(normed, c.qkvW, c.qkvS, c.qkvB)                       // (1, 1, (h+2d)·128)
        let split = kernel(.qkvPost, dtype: qkv.dtype, paramType: paramType, k: hidden, nt: 128, nq: h, nkv: d)(
            [qkv, layer.qNorm, layer.kNorm, params, position],
            grid: ((h + 2 * d) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, h, 1, headDim], [1, d, 1, headDim], [1, d, 1, headDim]],
            outputDTypes: [dtype, dtype, dtype])
        var (q, k, v) = (split[0], split[1], split[2])
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: layer.scale, mask: nil)
        let o = qmm(attended.reshaped(1, 1, h * headDim).asType(dtype), layer.o.weight, layer.o.scales, layer.o.biases)
        let addNorm = kernel(.addNorm, dtype: dtype, paramType: paramType, k: hidden, nt: 256, xType: o.dtype)(
            [x, o, layer.postNorm, params],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, hidden], [1, 1, hidden]], outputDTypes: [dtype, dtype])
        let (afterAttention, postNormed) = (addNorm[0], addNorm[1])
        let gu = qmm(postNormed, c.guW, c.guS, c.guB)                        // (1, 1, 2·intermediate)
        let activated = kernel(.siluMul, dtype: gu.dtype, paramType: paramType, k: intermediate, nt: 256)(
            [gu], grid: (intermediate, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, intermediate]], outputDTypes: [gu.dtype])[0]
        let down = qmm(activated, layer.down.weight, layer.down.scales, layer.down.biases)
        return afterAttention + down
    }
    /// Threads per group. qkv needs a whole head (128 rows) per group for
    /// the head norm, so it runs 32 simdgroups of 4 rows; the others run
    /// four simdgroups (16 rows) per group so enough groups are in flight.
    nonisolated(unsafe) static var qkvThreads = 512
    nonisolated(unsafe) static var matvecThreads = 128
    nonisolated(unsafe) static var gateUpThreads = 128
    static var matvecRowsPerGroup: Int { matvecThreads / 8 }
    static var gateUpRowsPerGroup: Int { gateUpThreads / 8 }

    /// `x` is (1, 1, hidden). Returns the layer output in the same shape.
    static func run(_ x: MLXArray, layer: Layer, cache: (any KVCache)?) -> MLXArray {
        if mode == .hybrid { return runHybrid(x, layer: layer, cache: cache) }
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

    static func matvecResidual(_ x: MLXArray, _ p: Layer.Projection, residual: MLXArray,
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

    enum Kind: String {
        case qkv = "qwen_qkv", matvec = "qwen_mv_res", gateUp = "qwen_gateup"
        case qkvPost = "qwen_qkv_post", addNorm = "qwen_add_norm", siluMul = "qwen_silu_mul"
    }

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
        case .qkvPost:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["qkv", "qnW", "knW", "params", "pos"],
                outputNames: ["q", "k", "v"], source: qkvPostSource, header: defs)
        case .addNorm:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["a", "b", "w", "params"],
                outputNames: ["sum", "normed"], source: addNormSource, header: defs)
        case .siluMul:
            built = MLXFast.metalKernel(
                name: name, inputNames: ["gu"], outputNames: ["out"], source: siluMulSource, header: defs)
        }
        kernels[key] = built
        return built
    }

    // MARK: - Metal

    /// Shared device code.
    ///
    /// No threadgroup staging of x: every simdgroup covers the whole K-vector
    /// on its own (lane l owns elements [c·1024 + 32·l, +32) of chunk c), so
    /// each lane reads its slice straight from device memory into registers,
    /// the rmsnorm's sum of squares is a plain simd_sum, and the matvec needs
    /// no barrier and no threadgroup memory. (The first cut staged x in a
    /// K-float threadgroup array; at K = 3072 that is 12 KB per group, which
    /// caps an Apple GPU core at two or three groups and starves the memory
    /// system.) Rows go in batches of R with all weight loads issued first.
    static let header = """
    // The lane's slices of every chunk of x, as float, times `scale` and (if
    // `w` is non-null) the per-element weight: xv[c * 32 + i] holds element
    // c·1024 + 32·lane + i. Loaded once per lane and reused for every row.
    template <typename X>
    inline void qwen_load_x(const device X* x, const device S* w, uint lane, float scale, thread float* xv) {
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < K / 1024; ++c) {
            const int e0 = c * 1024 + 32 * int(lane);
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < 32; ++i) {
                float v = float(x[e0 + i]) * scale;
                if (w != nullptr) v *= float(w[e0 + i]);
                xv[c * 32 + i] = v;
            }
        }
    }

    // 1/rms over the whole K-vector (identical on every lane of the simdgroup).
    template <typename X>
    inline float qwen_inv_rms(const device X* x, uint lane, float eps) {
        float ss = 0;
        for (int c = 0; c < K / 1024; ++c) {
            const int e0 = c * 1024 + 32 * int(lane);
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < 32; ++i) { const float v = float(x[e0 + i]); ss += v * v; }
        }
        return metal::rsqrt(simd_sum(ss) / float(K) + eps);
    }

    // R rows (row0, row0+1, …) of a [rows, K/8] uint32 4-bit affine matrix
    // (group 64, scale·q + bias) dotted with the preloaded xv; the reduced
    // values land in out[0..R) on every lane.
    template <int R>
    inline void qwen_rows_dot(const device uint* wq, const device S* scales, const device S* biases,
                              uint row0, thread const float* xv, uint lane, thread float* out) {
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
                packed[r] = *((const device uint4*)(wq + row * wordsPerRow + e0 / 8));
                s[r] = float(scales[row * groupsPerRow + g]);
                b[r] = float(biases[row * groupsPerRow + g]);
            }
            float sx = 0;
            float sq[R];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < R; ++r) sq[r] = 0;
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < 4; ++j) {
                _Pragma("clang loop unroll(full)")
                for (int n = 0; n < 8; ++n) {
                    const float v = xv[c * 32 + 8 * j + n];
                    sx += v;
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < R; ++r) {
                        const uint wv = j == 0 ? packed[r].x : j == 1 ? packed[r].y : j == 2 ? packed[r].z : packed[r].w;
                        sq[r] += v * float((wv >> (4 * n)) & 0xfu);
                    }
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < R; ++r) acc[r] += s[r] * sq[r] + b[r] * sx;
        }
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < R; ++r) out[r] = simd_sum(acc[r]);
    }
    """

    /// Glue after the fused q/k/v matmul: per-head rmsnorm on q and k, RoPE
    /// at `pos`, and the split into (1, heads, 1, 128) tensors. One 128-thread
    /// group per head, one element per thread.
    static let qkvPostSource = """
        threadgroup float red[4];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint head = threadgroup_position_in_grid.x;       // 0..<NQ+2·NKV
        const float eps = params[0];
        const float theta = params[1];
        const float v0 = float(qkv[head * 128 + tid]);
        if (head >= NQ + NKV) { v[(head - NQ - NKV) * 128 + tid] = T(v0); return; }
        const bool isQ = head < NQ;
        device T* out = isQ ? q + head * 128 : k + (head - NQ) * 128;
        const device S* nw = isQ ? qnW : knW;
        const float part = simd_sum(v0 * v0);
        if (lane == 0) red[simd] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = metal::rsqrt((red[0] + red[1] + red[2] + red[3]) / 128.0f + eps);
        const float n0 = v0 * inv * float(nw[tid]);
        // rotate-half pairs (i, i+64): fetch the partner through threadgroup memory
        threadgroup float hs[128];
        hs[tid] = n0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint i = tid & 63;
        const float freq = metal::pow(theta, -float(i) / 64.0f);
        const float angle = float(pos[0]) * freq;
        const float c = metal::cos(angle), s = metal::sin(angle);
        const float a = hs[i], b = hs[i + 64];
        out[tid] = tid < 64 ? T(a * c - b * s) : T(b * c + a * s);
        """

    /// sum = a + b; normed = rmsnorm(sum)·w. One 256-thread group, K/256
    /// elements per thread (`a` in T, `b` in TX).
    static let addNormSource = """
        threadgroup float red[8];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        float vals[K / 256];
        float ss = 0;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < K / 256; ++j) {
            const int i = tid + 256 * j;
            const float v = float(a[i]) + float(b[i]);
            vals[j] = v; ss += v * v;
            sum[i] = T(v);
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
            normed[i] = T(vals[j] * inv * float(w[i]));
        }
        """

    /// out[i] = silu(gu[i]) · gu[i + K] over the concatenated gate/up output.
    static let siluMulSource = """
        const uint i = thread_position_in_grid.x;
        const float g = float(gu[i]);
        out[i] = T(g / (1.0f + metal::exp(-g)) * float(gu[i + K]));
        """

    /// rmsnorm → q/k/v projections → q/k head norm → RoPE. One NT-thread
    /// group per head (NQ q heads, then NKV k heads, then NKV v heads); the
    /// NT/32 simdgroups share the head's 128 rows four at a time.
    static let qkvSource = """
        threadgroup float hs[128];
        threadgroup float red[NT / 32];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        const float eps = params[0];
        const float theta = params[1];
        float xv[K / 32];
        qwen_load_x<T>(x, inW, lane, qwen_inv_rms<T>(x, lane, eps), xv);

        const device uint* w; const device S* sc; const device S* bi; device T* out; const device S* nw;
        uint head; bool rotate = true;
        if (block < NQ) { head = block; w = wq; sc = sq; bi = bq; out = q; nw = qnW; }
        else if (block < NQ + NKV) { head = block - NQ; w = wk; sc = sk; bi = bk; out = k; nw = knW; }
        else { head = block - NQ - NKV; w = wv; sc = sv; bi = bv; out = v; rotate = false; nw = qnW; }

        // 128 rows per head over NT/32 simdgroups, four rows each pass.
        for (uint local = simd * 4; local < 128; local += (NT / 32) * 4) {
            float dots[4];
            qwen_rows_dot<4>(w, sc, bi, head * 128 + local, xv, lane, dots);
            if (lane == 0) { hs[local] = dots[0]; hs[local + 1] = dots[1]; hs[local + 2] = dots[2]; hs[local + 3] = dots[3]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!rotate) {
            if (tid < 128) out[head * 128 + tid] = T(hs[tid]);
            return;
        }
        // Per-head rmsnorm (q_norm / k_norm), then rotate-half RoPE at `pos`.
        const float hv = tid < 128 ? hs[tid] : 0.0f;
        float ss = simd_sum(hv * hv);
        if (lane == 0) red[simd] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = 0;
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < NT / 32; ++i) ss += red[i];
        const float hinv = metal::rsqrt(ss / 128.0f + eps);
        if (tid < 128) hs[tid] = hv * hinv * float(nw[tid]);
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
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        const uint row0 = block * (NT / 8) + simd * 4;
        float xv[K / 32];
        qwen_load_x<TX>(x, nullptr, lane, 1.0f, xv);
        float dots[4];
        qwen_rows_dot<4>(w, sc, bi, row0, xv, lane, dots);
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
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.x;
        float xv[K / 32];
        qwen_load_x<T>(x, postW, lane, qwen_inv_rms<T>(x, lane, params[0]), xv);
        const uint row0 = block * (NT / 8) + simd * 4;
        float g[4], u[4];
        qwen_rows_dot<4>(wg, sg, bg, row0, xv, lane, g);
        qwen_rows_dot<4>(wu, su, bu, row0, xv, lane, u);
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
