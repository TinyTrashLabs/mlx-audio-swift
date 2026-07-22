// SuperTonic 3 WaveNeXt vocoder (AE decoder) — Swift port of the spike's
// `vocoder.py`.
//
// Pipeline (decoded from vocoder.onnx):
//   latent [B,144,L] / 0.25 -> reshape [B,24,6,L] -> transpose -> [B,24,L*6]
//   -> * latent_std + latent_mean
//   -> edge-pad(6,0) -> stem conv k7 24->512   (fully CAUSAL, edge-replicate)
//   -> 10x ConvNeXt(512, k7, inter 2048, dil [1,2,4,1,2,4,1,1,1,1]) (unmasked, causal)
//   -> BatchNorm (running stats, eps 1e-5)
//   -> edge-pad(2,0) -> head conv k3 512->2048 -> PReLU(scalar) -> conv1x1 2048->512 (no bias)
//   -> reshape [B, T*512]

import Foundation
import MLX

final class SupertonicVocoder {
    let scale: MLXArray
    let latentStd: MLXArray  // [1,1,24]
    let latentMean: MLXArray
    let stemW: MLXArray  // [512,7,24]
    let stemB: MLXArray
    let convnext: SupertonicConvNeXtStack
    let bnScale: MLXArray
    let bnShift: MLXArray
    let head1W: MLXArray  // [2048,3,512]
    let head1B: MLXArray
    let preluA: MLXArray
    let head2W: MLXArray  // [2048,512]

    init(_ w: [String: MLXArray]) throws {
        scale = try w.req("ttl.normalizer.scale").reshaped([])
        latentStd = try w.req("ae.latent_std").reshaped([1, 1, -1])
        latentMean = try w.req("ae.latent_mean").reshaped([1, 1, -1])
        let p = "ae.decoder"
        stemW = try w.req("\(p).embed.net.weight")
        stemB = try w.req("\(p).embed.net.bias")
        convnext = try SupertonicConvNeXtStack(
            w, "\(p).convnext", 10, dilations: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1], causal: true)
        let bnEps: Float = 1e-5
        let variance = try w.req("\(p).final_norm.norm.running_var")
        let mean = try w.req("\(p).final_norm.norm.running_mean")
        let g = try w.req("\(p).final_norm.norm.weight")
        let b = try w.req("\(p).final_norm.norm.bias")
        bnScale = g * rsqrt(variance + bnEps)
        bnShift = b - mean * bnScale
        head1W = try w.req("\(p).head.layer1.net.weight")
        head1B = try w.req("\(p).head.layer1.net.bias")
        preluA = try w.req("\(p).head.act.weight").reshaped([])
        head2W = try w.req("\(p).head.layer2.weight")[0..., 0, 0...].transposed()
    }

    static func fromSafetensors(_ url: URL) throws -> SupertonicVocoder {
        try SupertonicVocoder(try loadArrays(url: url))
    }

    /// latent [B,144,L] -> wav [B, L*6*512].
    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let (b, l) = (latent.dim(0), latent.dim(2))
        var x = latent / scale
        x = x.reshaped([b, 24, 6, l]).transposed(0, 1, 3, 2).reshaped([b, 24, l * 6])
        x = x.transposed(0, 2, 1)  // [B,T=6L,24]
        x = x * latentStd + latentMean
        var h = supertonicEdgePad(x, left: 6, right: 0)  // causal edge pad
        h = conv1d(h, stemW) + stemB  // [B,T,512]
        h = convnext(h)
        h = h * bnScale + bnShift
        h = supertonicEdgePad(h, left: 2, right: 0)  // causal edge pad
        h = conv1d(h, head1W) + head1B  // [B,T,2048]
        h = which(h .>= 0, h, preluA * h)
        h = matmul(h, head2W)  // [B,T,512]
        return h.reshaped([b, -1])
    }
}
