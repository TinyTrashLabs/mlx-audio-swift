// SuperTonic 3 shared MLX modules — Swift port of the validated Python-MLX
// spike (`modules.py`). Internal layout is channels-LAST: x [B, T, C];
// masks [B, T, 1]. Weights come from a flat safetensors dict keyed by dotted
// paths; each module pulls its parameters via a prefix.
//
// All math mirrors the ONNX graphs exactly:
// - ConvNeXt1D: mask -> edge-pad + dwconv -> mask -> LN(eps 1e-6) -> pwconv1
//   -> gelu(erf) -> pwconv2 -> gamma -> +masked-input residual -> mask
// - VITSEncoder: VITS relative-position self-attention encoder (window 4, post-LN)
// - GSTCrossAttention: GST-style multihead, tanh on projected keys, scale 1/16,
//   post-softmax query masking
// - RotaryCrossAttention: half-split (NeoX) rotary on Q/K, positions normalized
//   by masked length, theta = 10 * 10000^(-i/32), scale 1/16
// - TimeEncoder: sinusoidal(t*1000) -> Linear -> Mish -> Linear
//
// IMPORTANT: all pads are edge-replicate (not zero); vocoder blocks are causal
// (left-only). LayerNorm eps is 1e-6 everywhere.

import Foundation
import MLX

enum SupertonicError: Error, CustomStringConvertible {
    case missingWeight(String)
    case missingFile(String)
    case invalidInput(String)

    var description: String {
        switch self {
        case .missingWeight(let k): return "Supertonic: missing weight \(k)"
        case .missingFile(let f): return "Supertonic: missing file \(f)"
        case .invalidInput(let m): return "Supertonic: \(m)"
        }
    }
}

extension [String: MLXArray] {
    func req(_ key: String) throws -> MLXArray {
        guard let v = self[key] else { throw SupertonicError.missingWeight(key) }
        return v
    }
}

let supertonicLNEps: Float = 1e-6

func supertonicGELUErf(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1.0 + erf(x / sqrtf(2.0)))
}

func supertonicMish(_ x: MLXArray) -> MLXArray {
    // softplus via logaddexp for numerical stability
    x * tanh(logAddExp(x, MLXArray(Float(0))))
}

/// Edge-replicate padding along the time axis (axis 1) of [B,T,C].
func supertonicEdgePad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    var parts = [MLXArray]()
    if left > 0 {
        parts.append(broadcast(x[0..., 0 ..< 1, 0...], to: [x.dim(0), left, x.dim(2)]))
    }
    parts.append(x)
    if right > 0 {
        parts.append(broadcast(x[0..., (x.dim(1) - 1)..., 0...], to: [x.dim(0), right, x.dim(2)]))
    }
    return parts.count > 1 ? concatenated(parts, axis: 1) : x
}

func supertonicLayerNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = supertonicLNEps) -> MLXArray {
    let mu = x.mean(axis: -1, keepDims: true)
    let v = variance(x, axis: -1, keepDims: true)
    return (x - mu) * rsqrt(v + eps) * w + b
}

/// One masked ConvNeXt-1D block, channels-last.
final class SupertonicConvNeXt1D {
    let dwW: MLXArray  // [C, K, 1]
    let dwB: MLXArray
    let nW: MLXArray
    let nB: MLXArray
    let pw1W: MLXArray  // [I, 1, C] (conv layout); used as [C, I] matmul
    let pw1B: MLXArray
    let pw2W: MLXArray
    let pw2B: MLXArray
    let gamma: MLXArray  // [C]
    let dilation: Int
    let causal: Bool
    let channels: Int
    let kernel: Int

    init(_ w: [String: MLXArray], _ p: String, dilation: Int = 1, causal: Bool = false) throws {
        let dw = w["\(p).dwconv.weight"] != nil ? "\(p).dwconv" : "\(p).dwconv.net"
        dwW = try w.req("\(dw).weight")
        dwB = try w.req("\(dw).bias")
        nW = try w.req("\(p).norm.norm.weight")
        nB = try w.req("\(p).norm.norm.bias")
        pw1W = try w.req("\(p).pwconv1.weight")
        pw1B = try w.req("\(p).pwconv1.bias")
        pw2W = try w.req("\(p).pwconv2.weight")
        pw2B = try w.req("\(p).pwconv2.bias")
        gamma = try w.req("\(p).gamma").reshaped([-1])
        self.dilation = dilation
        self.causal = causal
        channels = dwW.dim(0)
        kernel = dwW.dim(1)
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = input
        if let mask { x = x * mask }
        let total = dilation * (kernel - 1)
        var h = causal
            ? supertonicEdgePad(x, left: total, right: 0)
            : supertonicEdgePad(x, left: total / 2, right: total / 2)
        h = conv1d(h, dwW, dilation: dilation, groups: channels) + dwB
        if let mask { h = h * mask }
        h = supertonicLayerNorm(h, nW, nB)
        h = matmul(h, pw1W[0..., 0, 0...].transposed()) + pw1B
        h = supertonicGELUErf(h)
        h = matmul(h, pw2W[0..., 0, 0...].transposed()) + pw2B
        h = gamma * h
        x = x + h
        if let mask { x = x * mask }
        return x
    }
}

final class SupertonicConvNeXtStack {
    let blocks: [SupertonicConvNeXt1D]

    init(_ w: [String: MLXArray], _ p: String, _ n: Int, dilations: [Int]? = nil, causal: Bool = false) throws {
        let d = dilations ?? Array(repeating: 1, count: n)
        blocks = try (0 ..< n).map {
            try SupertonicConvNeXt1D(w, "\(p).\($0)", dilation: d[$0], causal: causal)
        }
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = input
        for b in blocks { x = b(x, mask: mask) }
        return x
    }
}

/// VITS `_relative_position_to_absolute_position`. x: [B,H,T,2T-1] -> [B,H,T,T].
func supertonicRelShiftLogits(_ x: MLXArray) -> MLXArray {
    let (b, h, t) = (x.dim(0), x.dim(1), x.dim(2))
    var y = padded(x, widths: [.init(0), .init(0), .init(0), .init((0, 1))])
    y = y.reshaped([b, h, t * 2 * t])
    y = padded(y, widths: [.init(0), .init(0), .init((0, t - 1))])
    y = y.reshaped([b, h, t + 1, 2 * t - 1])
    return y[0..., 0..., 0 ..< t, (t - 1)...]
}

/// VITS `_absolute_position_to_relative_position`. x: [B,H,T,T] -> [B,H,T,2T-1].
func supertonicAbsToRel(_ x: MLXArray) -> MLXArray {
    let (b, h, t) = (x.dim(0), x.dim(1), x.dim(2))
    var y = padded(x, widths: [.init(0), .init(0), .init(0), .init((0, t - 1))])
    y = y.reshaped([b, h, t * t + t * (t - 1)])
    y = padded(y, widths: [.init(0), .init(0), .init((t, 0))])
    y = y.reshaped([b, h, t, 2 * t])
    return y[0..., 0..., 0..., 1...]
}

/// VITS `_get_relative_embeddings`. emb: [1, 2w+1, d] -> [1, 2t-1, d].
func supertonicRelEmbeddings(_ emb: MLXArray, t: Int, window: Int) -> MLXArray {
    let padLen = max(t - (window + 1), 0)
    let start = max((window + 1) - t, 0)
    var e = emb
    if padLen > 0 {
        e = padded(e, widths: [.init(0), .init((padLen, padLen)), .init(0)])
    }
    return e[0..., start ..< (start + 2 * t - 1), 0...]
}

/// VITS MultiHeadAttention with relative position embeddings (self-attn).
final class SupertonicRelPosSelfAttention {
    let qW: MLXArray, qB: MLXArray
    let kW: MLXArray, kB: MLXArray
    let vW: MLXArray, vB: MLXArray
    let oW: MLXArray, oB: MLXArray
    let embRelK: MLXArray  // [1, 2w+1, d]
    let embRelV: MLXArray
    let heads: Int
    let window: Int

    init(_ w: [String: MLXArray], _ p: String, heads: Int, window: Int = 4) throws {
        qW = try w.req("\(p).conv_q.weight")[0..., 0, 0...].transposed()  // [C,C] for x@W
        qB = try w.req("\(p).conv_q.bias")
        kW = try w.req("\(p).conv_k.weight")[0..., 0, 0...].transposed()
        kB = try w.req("\(p).conv_k.bias")
        vW = try w.req("\(p).conv_v.weight")[0..., 0, 0...].transposed()
        vB = try w.req("\(p).conv_v.bias")
        oW = try w.req("\(p).conv_o.weight")[0..., 0, 0...].transposed()
        oB = try w.req("\(p).conv_o.bias")
        embRelK = try w.req("\(p).emb_rel_k")
        embRelV = try w.req("\(p).emb_rel_v")
        self.heads = heads
        self.window = window
    }

    func callAsFunction(_ x: MLXArray, attnMask: MLXArray?) -> MLXArray {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let h = heads
        let d = c / h
        let q = (matmul(x, qW) + qB).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)
        let k = (matmul(x, kW) + kB).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)
        let v = (matmul(x, vW) + vB).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)
        let qs = q / sqrtf(Float(d))
        var scores = matmul(qs, k.transposed(0, 1, 3, 2))
        let relK = supertonicRelEmbeddings(embRelK, t: t, window: window)  // [1,2t-1,d]
        let relLogits = matmul(qs, relK[0].transposed())  // [b,h,t,2t-1]
        scores = scores + supertonicRelShiftLogits(relLogits)
        if let attnMask {
            scores = which(attnMask .== 0, MLXArray(Float(-1e4)), scores)
        }
        let pAttn = softmax(scores, axis: -1)
        var out = matmul(pAttn, v)
        let relV = supertonicRelEmbeddings(embRelV, t: t, window: window)
        out = out + matmul(supertonicAbsToRel(pAttn), relV[0])
        let flat = out.transposed(0, 2, 1, 3).reshaped([b, t, c])
        return matmul(flat, oW) + oB
    }
}

final class SupertonicFFN {
    let w1: MLXArray, b1: MLXArray
    let w2: MLXArray, b2: MLXArray

    init(_ w: [String: MLXArray], _ p: String) throws {
        w1 = try w.req("\(p).conv_1.weight")[0..., 0, 0...].transposed()
        b1 = try w.req("\(p).conv_1.bias")
        w2 = try w.req("\(p).conv_2.weight")[0..., 0, 0...].transposed()
        b2 = try w.req("\(p).conv_2.bias")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = matmul(x * mask, w1) + b1
        h = maximum(h, 0.0)
        return matmul(h * mask, w2) + b2
    }
}

/// attn_encoder: n layers of [rel-pos MHA -> post-LN -> FFN -> post-LN].
final class SupertonicVITSEncoder {
    let attn: [SupertonicRelPosSelfAttention]
    let ffn: [SupertonicFFN]
    let n1W: [MLXArray], n1B: [MLXArray]
    let n2W: [MLXArray], n2B: [MLXArray]

    init(_ w: [String: MLXArray], _ p: String, nLayers: Int, heads: Int, window: Int = 4) throws {
        attn = try (0 ..< nLayers).map {
            try SupertonicRelPosSelfAttention(w, "\(p).attn_layers.\($0)", heads: heads, window: window)
        }
        ffn = try (0 ..< nLayers).map { try SupertonicFFN(w, "\(p).ffn_layers.\($0)") }
        n1W = try (0 ..< nLayers).map { try w.req("\(p).norm_layers_1.\($0).norm.weight") }
        n1B = try (0 ..< nLayers).map { try w.req("\(p).norm_layers_1.\($0).norm.bias") }
        n2W = try (0 ..< nLayers).map { try w.req("\(p).norm_layers_2.\($0).norm.weight") }
        n2B = try (0 ..< nLayers).map { try w.req("\(p).norm_layers_2.\($0).norm.bias") }
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        // attn_mask [B,1,T,T]
        let m4 = mask.expandedDimensions(axis: 1)  // [B,1,T,1]
        let attnMask = m4 * m4.transposed(0, 1, 3, 2)
        var x = input * mask
        for i in 0 ..< attn.count {
            var y = attn[i](x, attnMask: attnMask)
            x = supertonicLayerNorm(x + y, n1W[i], n1B[i])
            y = ffn[i](x, mask: mask)
            x = supertonicLayerNorm(x + y, n2W[i], n2B[i])
        }
        return x * mask
    }
}

/// GST-style multi-head cross-attention: tanh on projected keys, scale 1/16,
/// softmax then query-mask zeroing. Returns out_fc output (unmasked).
final class SupertonicGSTCrossAttention {
    let qW: MLXArray, qB: MLXArray
    let kW: MLXArray, kB: MLXArray
    let vW: MLXArray, vB: MLXArray
    let oW: MLXArray, oB: MLXArray
    let heads: Int
    let scale: Float

    init(_ w: [String: MLXArray], _ p: String, heads: Int = 2, scale: Float = 16.0) throws {
        qW = try w.req("\(p).W_query.linear.weight")
        qB = try w.req("\(p).W_query.linear.bias")
        kW = try w.req("\(p).W_key.linear.weight")
        kB = try w.req("\(p).W_key.linear.bias")
        vW = try w.req("\(p).W_value.linear.weight")
        vB = try w.req("\(p).W_value.linear.bias")
        oW = try w.req("\(p).out_fc.linear.weight")
        oB = try w.req("\(p).out_fc.linear.bias")
        self.heads = heads
        self.scale = scale
    }

    func callAsFunction(_ x: MLXArray, keyIn: MLXArray, valIn: MLXArray, qMask: MLXArray? = nil) -> MLXArray {
        let (b, tq) = (x.dim(0), x.dim(1))
        let s = keyIn.dim(1)
        let h = heads
        let q0 = matmul(x, qW) + qB  // [B,Tq,U]
        let k0 = matmul(keyIn, kW) + kB  // [B,S,U]
        let v0 = matmul(valIn, vW) + vB
        let u = q0.dim(-1)
        let d = u / h
        let q = q0.reshaped([b, tq, h, d]).transposed(0, 2, 1, 3)
        let k = k0.reshaped([b, s, h, d]).transposed(0, 2, 1, 3)
        let v = v0.reshaped([b, s, h, d]).transposed(0, 2, 1, 3)
        let scores = matmul(q, tanh(k).transposed(0, 1, 3, 2)) / scale
        var p = softmax(scores, axis: -1)
        if let qMask {
            p = which(qMask.expandedDimensions(axis: 1) .== 0, MLXArray(Float(0)), p)
        }
        let out = matmul(p, v).transposed(0, 2, 1, 3).reshaped([b, tq, u])
        return matmul(out, oW) + oB
    }
}

/// Half-split rotary. x [B,H,T,64], pos [B,T,1], theta [1,1,32].
func supertonicRotaryRotate(_ x: MLXArray, pos: MLXArray, theta: MLXArray) -> MLXArray {
    let ang = pos * theta  // [B,T,32]
    let sin = MLX.sin(ang).expandedDimensions(axis: 1)  // [B,1,T,32]
    let cos = MLX.cos(ang).expandedDimensions(axis: 1)
    let d2 = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., 0 ..< d2]
    let x2 = x[0..., 0..., 0..., d2...]
    return concatenated([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis: -1)
}

/// VE text cross-attention: 8 heads x 64, rotary on Q/K, scale 1/16,
/// -inf key masking pre-softmax, query-mask zeroing post-softmax.
final class SupertonicRotaryCrossAttention {
    let qW: MLXArray, qB: MLXArray
    let kW: MLXArray, kB: MLXArray
    let vW: MLXArray, vB: MLXArray
    let oW: MLXArray, oB: MLXArray
    let theta: MLXArray  // [1,1,32]
    let heads: Int
    let scale: Float

    init(_ w: [String: MLXArray], _ p: String, heads: Int = 8, scale: Float = 16.0, theta: MLXArray? = nil) throws {
        qW = try w.req("\(p).W_query.linear.weight")
        qB = try w.req("\(p).W_query.linear.bias")
        kW = try w.req("\(p).W_key.linear.weight")
        kB = try w.req("\(p).W_key.linear.bias")
        vW = try w.req("\(p).W_value.linear.weight")
        vB = try w.req("\(p).W_value.linear.bias")
        oW = try w.req("\(p).out_fc.linear.weight")
        oB = try w.req("\(p).out_fc.linear.bias")
        // ONNX dedupes identical initializers: only one block carries theta
        if let t = w["\(p).theta"] {
            self.theta = t
        } else if let theta {
            self.theta = theta
        } else {
            throw SupertonicError.missingWeight("\(p).theta")
        }
        self.heads = heads
        self.scale = scale
    }

    /// x [B,Tq,512] (already masked), kv [B,Tk,256], masks [B,T,1].
    func callAsFunction(_ x: MLXArray, kv: MLXArray, qMask: MLXArray, kvMask: MLXArray) -> MLXArray {
        let (b, tq) = (x.dim(0), x.dim(1))
        let tk = kv.dim(1)
        let h = heads
        let q0 = matmul(x, qW) + qB
        let d = q0.dim(-1) / h
        var q = q0.reshaped([b, tq, h, d]).transposed(0, 2, 1, 3)
        var k = (matmul(kv, kW) + kB).reshaped([b, tk, h, d]).transposed(0, 2, 1, 3)
        let v = (matmul(kv, vW) + vB).reshaped([b, tk, h, d]).transposed(0, 2, 1, 3)
        // normalized positions: arange(T) / masked_length
        let lenQ = qMask.sum(axes: [1, 2]).reshaped([b, 1, 1])
        let lenK = kvMask.sum(axes: [1, 2]).reshaped([b, 1, 1])
        let posQ = MLXArray((0 ..< tq).map { Float($0) }).reshaped([1, tq, 1]) / lenQ
        let posK = MLXArray((0 ..< tk).map { Float($0) }).reshaped([1, tk, 1]) / lenK
        q = supertonicRotaryRotate(q, pos: posQ, theta: theta)
        k = supertonicRotaryRotate(k, pos: posK, theta: theta)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / scale
        let kmask = kvMask.expandedDimensions(axis: 1).transposed(0, 1, 3, 2)  // [B,1,1,Tk]
        scores = which(kmask .== 0, MLXArray(-Float.infinity), scores)
        var p = softmax(scores, axis: -1)
        p = which(qMask.expandedDimensions(axis: 1) .== 0, MLXArray(Float(0)), p)
        let out = matmul(p, v).transposed(0, 2, 1, 3).reshaped([b, tq, h * d])
        return matmul(out, oW) + oB
    }
}

/// t [B] -> sinusoidal(t*1000) -> Linear(64,256) -> Mish -> Linear(256,64).
final class SupertonicTimeEncoder {
    let invFreq: MLXArray  // [1,32]
    let w0: MLXArray, b0: MLXArray  // [64,256] (in,out)
    let w2: MLXArray, b2: MLXArray  // [256,64]

    init(_ w: [String: MLXArray], _ p: String) throws {
        invFreq = try w.req("\(p).sinusoidal.inv_freq").reshaped([1, -1])
        w0 = try w.req("\(p).mlp.0.linear.weight")
        b0 = try w.req("\(p).mlp.0.linear.bias")
        w2 = try w.req("\(p).mlp.2.linear.weight")
        b2 = try w.req("\(p).mlp.2.linear.bias")
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let ang = (t.reshaped([-1, 1]) * 1000.0) * invFreq
        let emb = concatenated([sin(ang), cos(ang)], axis: -1)  // [B,64]
        var h = matmul(emb, w0) + b0
        h = supertonicMish(h)
        return matmul(h, w2) + b2  // [B,64]
    }
}
