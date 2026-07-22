// SuperTonic 3 text encoder — Swift port of the spike's `text_encoder.py`.
//
// Pipeline (decoded from text_encoder.onnx):
//   embed(text_ids)*mask -> 6x ConvNeXt(256, k5, inter 1024, dil [1,1,2,2,4,4])
//   -> 4-layer VITS rel-pos encoder (4 heads) -> skip add -> *mask
//   -> speech_prompted_text_encoder:
//        x1 = x0 + GST_attn1(x0; K=style_key, V=style_ttl)*tmask
//        out = x0 + GST_attn2(x1; K=style_key, V=style_ttl)*tmask   (residual from x0!)
//        out = LayerNorm(out) * mask
// Keys of both GST attentions come from the FIXED learned style_key prototype
// [1,50,256]; only the values come from style_ttl.

import Foundation
import MLX

final class SupertonicTextEncoder {
    let embed: MLXArray  // [8322,256]
    let convnext: SupertonicConvNeXtStack
    let attnEncoder: SupertonicVITSEncoder
    let styleKey: MLXArray  // [1,50,256]
    let attn1: SupertonicGSTCrossAttention
    let attn2: SupertonicGSTCrossAttention
    let nW: MLXArray
    let nB: MLXArray

    init(_ w: [String: MLXArray]) throws {
        let p = "text_encoder"
        embed = try w.req("\(p).text_embedder.char_embedder.weight")
        convnext = try SupertonicConvNeXtStack(w, "\(p).convnext.convnext", 6, dilations: [1, 1, 2, 2, 4, 4])
        attnEncoder = try SupertonicVITSEncoder(w, "\(p).attn_encoder", nLayers: 4, heads: 4)
        styleKey = try w.req("style_encoder.style_token_layer.style_key")
        let sp = "speech_prompted_text_encoder"
        attn1 = try SupertonicGSTCrossAttention(w, "\(sp).attention1", heads: 2)
        attn2 = try SupertonicGSTCrossAttention(w, "\(sp).attention2", heads: 2)
        nW = try w.req("\(sp).norm.norm.weight")
        nB = try w.req("\(sp).norm.norm.bias")
    }

    static func fromSafetensors(_ url: URL) throws -> SupertonicTextEncoder {
        try SupertonicTextEncoder(try loadArrays(url: url))
    }

    /// text_ids [B,T] int, style_ttl [B,50,256], text_mask [B,1,T] -> [B,256,T].
    func callAsFunction(_ textIds: MLXArray, styleTtl: MLXArray, textMask: MLXArray) -> MLXArray {
        let b = textIds.dim(0)
        let mask = textMask.transposed(0, 2, 1)  // [B,T,1]
        let e = take(embed, textIds, axis: 0) * mask
        let xc = convnext(e, mask: mask)
        let y = attnEncoder(xc, mask: mask)
        let x0 = (y + xc) * mask
        let key = broadcast(styleKey, to: [b, styleKey.dim(1), styleKey.dim(2)])
        let a1 = attn1(x0, keyIn: key, valIn: styleTtl, qMask: mask) * mask
        let x1 = x0 + a1
        let a2 = attn2(x1, keyIn: key, valIn: styleTtl, qMask: mask) * mask
        var out = x0 + a2
        out = supertonicLayerNorm(out, nW, nB) * mask
        return out.transposed(0, 2, 1)  // [B,256,T]
    }
}
