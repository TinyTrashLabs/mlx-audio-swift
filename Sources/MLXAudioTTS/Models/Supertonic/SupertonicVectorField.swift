// SuperTonic 3 flow-matching vector field — Swift port of the spike's
// `vector_estimator.py`, with CFG + Euler factored out.
//
// Backbone (decoded from vector_estimator.onnx), channels-last internally:
//   proj_in conv1x1 144->512 (no bias)
//   4 groups of 6 main_blocks:
//     [6g+0] ConvNeXt x4 (512, k5, inter 2048, dil 1,2,4,8)
//     [6g+1] x = (x + Linear(t_emb 64->512)) * mask
//     [6g+2] ConvNeXt x1
//     [6g+3] rotary text cross-attn (8h x 64, theta = 10*10000^(-i/32), scale 1/16)
//            -> +residual -> post-LN -> *mask
//     [6g+4] ConvNeXt x1
//     [6g+5] GST style cross-attn -> +residual -> post-LN -> *mask
//   last_convnext x4 (dil 1) -> proj_out conv1x1 512->144 (no bias) -> *mask
//
// eulerCFG reproduces the ONNX graph: batch-doubled cond/uncond with
// uncond_masker special tokens, v = cfg*cond - (cfg-1)*uncond,
// x' = (x + v/total)*mask. velocity() returns the raw conditional velocity.

import Foundation
import MLX

final class SupertonicVectorField {
    struct Group {
        let conv4: SupertonicConvNeXtStack
        let timeW: MLXArray, timeB: MLXArray  // [64,512]
        let conv1a: SupertonicConvNeXtStack
        let attnT: SupertonicRotaryCrossAttention
        let normTW: MLXArray, normTB: MLXArray
        let conv1b: SupertonicConvNeXtStack
        let attnS: SupertonicGSTCrossAttention
        let normSW: MLXArray, normSB: MLXArray
    }

    let projIn: MLXArray  // [144,512]
    let projOut: MLXArray  // [512,144]
    let timeEncoder: SupertonicTimeEncoder
    let styleKey: MLXArray  // [1,50,256]
    let uncondText: MLXArray  // [1,256,1]
    let uncondStyleKey: MLXArray  // [1,50,256]
    let uncondStyleVal: MLXArray
    let groups: [Group]
    let last: SupertonicConvNeXtStack

    init(_ w: [String: MLXArray]) throws {
        let p = "vector_field"
        projIn = try w.req("\(p).proj_in.net.weight")[0..., 0, 0...].transposed()
        projOut = try w.req("\(p).proj_out.net.weight")[0..., 0, 0...].transposed()
        timeEncoder = try SupertonicTimeEncoder(w, "\(p).time_encoder")
        styleKey = try w.req("\(p).style_key")
        uncondText = try w.req("uncond_masker.text_special_token")
        uncondStyleKey = try w.req("uncond_masker.style_key_special_token")
        uncondStyleVal = try w.req("uncond_masker.style_value_special_token")
        // ONNX dedupes identical initializers: only main_blocks.3.attn carries theta
        let sharedTheta = try w.req("\(p).main_blocks.3.attn.theta")
        var gs = [Group]()
        for g in 0 ..< 4 {
            let base = 6 * g
            gs.append(Group(
                conv4: try SupertonicConvNeXtStack(w, "\(p).main_blocks.\(base).convnext", 4, dilations: [1, 2, 4, 8]),
                timeW: try w.req("\(p).main_blocks.\(base + 1).linear.linear.weight"),
                timeB: try w.req("\(p).main_blocks.\(base + 1).linear.linear.bias"),
                conv1a: try SupertonicConvNeXtStack(w, "\(p).main_blocks.\(base + 2).convnext", 1),
                attnT: try SupertonicRotaryCrossAttention(w, "\(p).main_blocks.\(base + 3).attn", theta: sharedTheta),
                normTW: try w.req("\(p).main_blocks.\(base + 3).norm.norm.weight"),
                normTB: try w.req("\(p).main_blocks.\(base + 3).norm.norm.bias"),
                conv1b: try SupertonicConvNeXtStack(w, "\(p).main_blocks.\(base + 4).convnext", 1),
                attnS: try SupertonicGSTCrossAttention(w, "\(p).main_blocks.\(base + 5).attention", heads: 2),
                normSW: try w.req("\(p).main_blocks.\(base + 5).norm.norm.weight"),
                normSB: try w.req("\(p).main_blocks.\(base + 5).norm.norm.bias")
            ))
        }
        groups = gs
        last = try SupertonicConvNeXtStack(w, "\(p).last_convnext.convnext", 4)
    }

    static func fromSafetensors(_ url: URL) throws -> SupertonicVectorField {
        try SupertonicVectorField(try loadArrays(url: url))
    }

    /// All channels-last: noisy [B,L,144], text_mem [B,Tk,256],
    /// style mems [B,50,256], masks [B,*,1], t [B] -> v [B,L,144].
    private func backbone(
        _ noisy: MLXArray, textMem: MLXArray, styleKeyMem: MLXArray, styleValMem: MLXArray,
        latentMask: MLXArray, textMask: MLXArray, t: MLXArray
    ) -> MLXArray {
        let tEmb = timeEncoder(t)  // [B,64]
        var x = matmul(noisy, projIn)
        for grp in groups {
            x = grp.conv4(x, mask: latentMask)
            x = (x + (matmul(tEmb, grp.timeW) + grp.timeB).expandedDimensions(axis: 1)) * latentMask
            x = grp.conv1a(x, mask: latentMask)
            var xm = x * latentMask
            let aT = grp.attnT(xm, kv: textMem, qMask: latentMask, kvMask: textMask) * latentMask
            x = supertonicLayerNorm(xm + aT, grp.normTW, grp.normTB) * latentMask
            x = grp.conv1b(x, mask: latentMask)
            xm = x * latentMask
            let aS = grp.attnS(xm, keyIn: styleKeyMem, valIn: styleValMem, qMask: latentMask) * latentMask
            x = supertonicLayerNorm(xm + aS, grp.normSW, grp.normSB) * latentMask
        }
        x = last(x, mask: latentMask)
        return matmul(x, projOut) * latentMask
    }

    /// Raw conditional velocity. ONNX-layout inputs:
    /// noisy [B,144,L], text_emb [B,256,T], style_ttl [B,50,256],
    /// latent_mask [B,1,L], text_mask [B,1,T], t [B] -> v_cond [B,144,L].
    func velocity(
        _ noisy: MLXArray, textEmb: MLXArray, styleTtl: MLXArray,
        latentMask: MLXArray, textMask: MLXArray, t: MLXArray
    ) -> MLXArray {
        let b = noisy.dim(0)
        let lm = latentMask.transposed(0, 2, 1)
        let tm = textMask.transposed(0, 2, 1)
        let key = broadcast(styleKey, to: [b, styleKey.dim(1), styleKey.dim(2)])
        let v = backbone(
            noisy.transposed(0, 2, 1), textMem: textEmb.transposed(0, 2, 1),
            styleKeyMem: key, styleValMem: styleTtl,
            latentMask: lm, textMask: tm, t: t
        )
        return v.transposed(0, 2, 1)
    }

    /// CFG-combined velocity via the batch-doubled cond/uncond trick.
    func velocityCFG(
        _ noisy: MLXArray, textEmb: MLXArray, styleTtl: MLXArray,
        latentMask: MLXArray, textMask: MLXArray, t: MLXArray, cfgScale: Float = 4.0
    ) -> MLXArray {
        let b = noisy.dim(0)
        let tk = textEmb.dim(2)
        // doubled batch: [cond, uncond]
        let noisy2 = concatenated([noisy, noisy], axis: 0).transposed(0, 2, 1)
        let lm = concatenated([latentMask, latentMask], axis: 0).transposed(0, 2, 1)
        let tm = concatenated([textMask, textMask], axis: 0).transposed(0, 2, 1)
        let uText = broadcast(uncondText, to: [b, 256, tk])
        let textMem = concatenated([textEmb, uText], axis: 0).transposed(0, 2, 1)
        let key = broadcast(styleKey, to: [b, styleKey.dim(1), styleKey.dim(2)])
        let uKey = broadcast(uncondStyleKey, to: [b, 50, 256])
        let uVal = broadcast(uncondStyleVal, to: [b, 50, 256])
        let keyMem = concatenated([key, uKey], axis: 0)
        let valMem = concatenated([styleTtl, uVal], axis: 0)
        let t2 = concatenated([t, t], axis: 0)
        let v = backbone(
            noisy2, textMem: textMem, styleKeyMem: keyMem, styleValMem: valMem,
            latentMask: lm, textMask: tm, t: t2
        ).transposed(0, 2, 1)  // [2B,144,L]
        let vCond = v[0 ..< b]
        let vUncond = v[b...]
        return cfgScale * vCond - (cfgScale - 1.0) * vUncond
    }

    /// Reproduces vector_estimator.onnx exactly.
    /// current_step/total_step: [B] float arrays -> denoised [B,144,L].
    func eulerCFG(
        _ noisy: MLXArray, textEmb: MLXArray, styleTtl: MLXArray,
        latentMask: MLXArray, textMask: MLXArray,
        currentStep: MLXArray, totalStep: MLXArray, cfgScale: Float = 4.0
    ) -> MLXArray {
        let t = currentStep / totalStep
        let v = velocityCFG(
            noisy, textEmb: textEmb, styleTtl: styleTtl,
            latentMask: latentMask, textMask: textMask, t: t, cfgScale: cfgScale
        )
        let dt = (1.0 / totalStep).reshaped([-1, 1, 1])
        return (noisy + dt * v) * latentMask
    }
}
