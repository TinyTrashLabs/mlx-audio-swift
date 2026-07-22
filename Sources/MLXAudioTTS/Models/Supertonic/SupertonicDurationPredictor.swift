// SuperTonic 3 duration predictor — Swift port of the spike's `dp.py`.
//
// Pipeline (decoded from duration_predictor.onnx):
//   embed(text_ids)*mask -> prepend learned sentence_token (mask 1)
//   -> 6x ConvNeXt(64, k5, inter 256, dil 1) -> 2-layer VITS rel-pos encoder (2 heads)
//   -> skip add (attn_out + convnext_out) -> slice sentence-token position
//   -> proj_out conv1x1 (no bias) * mask -> concat(flatten(style_dp))
//   -> Gemm 192->128 -> PReLU -> Gemm 128->1 -> Exp -> squeeze

import Foundation
import MLX

final class SupertonicDurationPredictor {
    let embed: MLXArray  // [8322,64]
    let sentenceToken: MLXArray  // [1,64,1]
    let convnext: SupertonicConvNeXtStack
    let attnEncoder: SupertonicVITSEncoder
    let projW: MLXArray  // [64,64]
    let predW0: MLXArray, predB0: MLXArray  // [in,out]
    let predW1: MLXArray, predB1: MLXArray
    let preluA: MLXArray

    init(_ w: [String: MLXArray]) throws {
        let p = "sentence_encoder"
        embed = try w.req("\(p).text_embedder.char_embedder.weight")
        sentenceToken = try w.req("\(p).sentence_token")
        convnext = try SupertonicConvNeXtStack(w, "\(p).convnext.convnext", 6)
        attnEncoder = try SupertonicVITSEncoder(w, "\(p).attn_encoder", nLayers: 2, heads: 2)
        projW = try w.req("\(p).proj_out.net.weight")[0..., 0, 0...].transposed()
        predW0 = try w.req("predictor.layers.0.weight")
        predB0 = try w.req("predictor.layers.0.bias")
        predW1 = try w.req("predictor.layers.1.weight")
        predB1 = try w.req("predictor.layers.1.bias")
        preluA = try w.req("predictor.activation.weight").reshaped([])
    }

    static func fromSafetensors(_ url: URL) throws -> SupertonicDurationPredictor {
        try SupertonicDurationPredictor(try loadArrays(url: url))
    }

    /// text_ids [B,T] int, style_dp [B,8,16], text_mask [B,1,T] -> duration [B].
    func callAsFunction(_ textIds: MLXArray, styleDp: MLXArray, textMask: MLXArray) -> MLXArray {
        let b = textIds.dim(0)
        let mask = textMask.transposed(0, 2, 1)  // [B,T,1]
        let e = take(embed, textIds, axis: 0) * mask  // [B,T,64]
        let tok = broadcast(sentenceToken.transposed(0, 2, 1), to: [b, 1, e.dim(-1)])
        var x = concatenated([tok, e], axis: 1)  // [B,T+1,64]
        let m = concatenated([MLXArray.ones([b, 1, 1], dtype: mask.dtype), mask], axis: 1)
        x = convnext(x, mask: m)
        let y = attnEncoder(x, mask: m)
        x = y + x  // skip
        let s = matmul(x[0..., 0 ..< 1, 0...], projW) * m[0..., 0 ..< 1, 0...]  // [B,1,64]
        let sent = s.reshaped([b, -1])
        let feat = concatenated([sent, styleDp.reshaped([b, -1])], axis: 1)  // [B,192]
        var h = matmul(feat, predW0) + predB0
        h = which(h .>= 0, h, preluA * h)
        h = matmul(h, predW1) + predB1
        return exp(h)[0..., 0]
    }
}
