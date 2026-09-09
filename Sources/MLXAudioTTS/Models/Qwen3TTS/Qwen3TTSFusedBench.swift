import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

/// On-device kernel microbenchmark for the fused step (iOS speed work,
/// 2026-09-09). The Mac is launch-bound and cannot show how the kernels
/// behave on an iPhone's GPU, so this runs the pieces there: N independent
/// ops per eval, wall time divided by N. Returns microseconds per op keyed
/// by a short label; the spike runner writes them into its report.
public enum Qwen3TTSFusedBench {
    public static func run(ops: Int = 50, progress: (String) -> Void = { _ in }) -> [String: Double] {
        var results: [String: Double] = [:]
        let dtype = DType.bfloat16
        func timed(_ label: String, repeats: Int = 3, _ body: () -> [MLXArray]) {
            eval(body())                                  // compile + warm
            var best = Double.infinity
            for _ in 0 ..< repeats {
                let t0 = Date()
                eval(body())
                best = min(best, Date().timeIntervalSince(t0))
            }
            let perOp = (best / Double(ops) * 1e6 * 10).rounded() / 10
            // Two rounds over the whole set; keep the minimum so the GPU
            // clock drifting within a run does not favour whatever ran first.
            results[label] = min(results[label] ?? .infinity, perOp)
            progress(String(format: "  %@: %.1f µs/op", label, perOp))
        }
        func quantized(_ out: Int, _ inp: Int) -> QuantizedLinear {
            let plain = Linear(inp, out, bias: false)
            plain.update(parameters: ModuleParameters.unflattened([("weight", plain.weight.asType(dtype))]))
            let q = QuantizedLinear(plain, groupSize: 64, bits: 4)
            eval(q.parameters())
            return q
        }
        let x = MLXRandom.normal([1, 1, 1024]).asType(dtype)
        let h = MLXRandom.normal([1, 1, 3072]).asType(dtype)
        let att = MLXRandom.normal([1, 16, 1, 128]).asType(dtype)
        eval(x, h, att)

        let trivial = MLXFast.metalKernel(name: "qwen_trivial_bench", inputNames: ["a"], outputNames: ["o"],
                                          source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];")
        for _ in 0 ..< 2 {
        timed("trivial custom kernel") {
            (0 ..< ops).map { _ in trivial([x], grid: (1024, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[1, 1, 1024]], outputDTypes: [dtype])[0] }
        }
        timed("MLX rmsNorm") { (0 ..< ops).map { _ in MLXFast.rmsNorm(x, weight: x[0, 0], eps: 1e-6) } }
        timed("MLX add") { (0 ..< ops).map { _ in x + x } }

        let o = quantized(1024, 2048), down = quantized(1024, 3072), gate = quantized(3072, 1024), up = quantized(3072, 1024)
        let qp = quantized(2048, 1024), kp = quantized(1024, 1024), vp = quantized(1024, 1024)
        timed("MLX qmm o_proj 1024x2048") { (0 ..< ops).map { _ in o(att.reshaped(1, 1, 2048)) } }
        timed("MLX qmm down 1024x3072") { (0 ..< ops).map { _ in down(h) } }
        timed("MLX qmm gate 3072x1024") { (0 ..< ops).map { _ in gate(x) } }
        timed("MLX qmm q_proj 2048x1024") { (0 ..< ops).map { _ in qp(x) } }

        guard let oP = Qwen3TTSFusedStep.Layer.Projection(o), let dP = Qwen3TTSFusedStep.Layer.Projection(down),
              let gP = Qwen3TTSFusedStep.Layer.Projection(gate), let uP = Qwen3TTSFusedStep.Layer.Projection(up),
              let qP = Qwen3TTSFusedStep.Layer.Projection(qp), let kP = Qwen3TTSFusedStep.Layer.Projection(kp),
              let vP = Qwen3TTSFusedStep.Layer.Projection(vp) else { break }
        let norm = MLXArray.ones([1024]).asType(dtype), headNorm = MLXArray.ones([128]).asType(dtype)
        let params = MLXArray([Float(1e-6), Float(1_000_000)]), pos = MLXArray([Int32(7)])

        for nt in [64, 128, 256] {
            Qwen3TTSFusedStep.matvecThreads = nt
            timed("fused o_proj+res nt\(nt)") { (0 ..< ops).map { _ in Qwen3TTSFusedStep.matvecResidual(att, oP, residual: x, inputWidth: 2048, rows: 1024) } }
            timed("fused down+res nt\(nt)") { (0 ..< ops).map { _ in Qwen3TTSFusedStep.matvecResidual(h, dP, residual: x, inputWidth: 3072, rows: 1024) } }
        }
        for nt in [64, 128, 256] {
            timed("fused gateup nt\(nt)") {
                (0 ..< ops).map { _ in
                    Qwen3TTSFusedStep.kernel(.gateUp, dtype: dtype, paramType: dtype, k: 1024, nt: nt)(
                        [x, norm, gP.weight, gP.scales, gP.biases, uP.weight, uP.scales, uP.biases, params],
                        grid: (3072 / (nt / 8) * nt, 1, 1), threadGroup: (nt, 1, 1),
                        outputShapes: [[1, 1, 3072]], outputDTypes: [dtype])[0]
                }
            }
        }
        for nt in [256, 512, 1024] {
            timed("fused qkv nt\(nt)") {
                (0 ..< ops).map { _ in
                    Qwen3TTSFusedStep.kernel(.qkv, dtype: dtype, paramType: dtype, k: 1024, nt: nt, nq: 16, nkv: 8)(
                        [x, norm, qP.weight, qP.scales, qP.biases, kP.weight, kP.scales, kP.biases,
                         vP.weight, vP.scales, vP.biases, headNorm, headNorm, params, pos],
                        grid: (32 * nt, 1, 1), threadGroup: (nt, 1, 1),
                        outputShapes: [[1, 16, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]],
                        outputDTypes: [dtype, dtype, dtype])[0]
                }
            }
        }
        }
        Qwen3TTSFusedStep.matvecThreads = 128
        return results
    }
}
