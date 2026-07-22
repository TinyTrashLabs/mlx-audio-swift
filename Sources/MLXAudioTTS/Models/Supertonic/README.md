# SuperTonic 3 (native MLX)

Native-MLX port of [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3),
a 99M-parameter flow-matching text-to-speech model (44.1 kHz, 10 preset voices
M1–M5 / F1–F5, 32 languages). No ONNX Runtime — the four ONNX sub-graphs were
reverse-engineered into MLX, validated per-stage at ≤1e-4 vs ONNX Runtime in a
Python spike, then ported here with the same parity gates against golden
fixtures (`Tests/Fixtures/supertonic/`).

## Pipeline

```
text ──frontend──▶ ids ──duration_predictor──▶ seconds
                    │
                    └──text_encoder(style_ttl)──▶ text_emb [1,256,T]
noise [1,144,L] ──(8 × Euler, CFG 4.0)── vector_field ──▶ latent [1,144,L]
latent ──WaveNeXt vocoder──▶ wav [L·3072] @ 44100 Hz
```

- **Frontend** (`SupertonicTextFrontend`): NFKD (compatibility) decomposition,
  emoji/dash/quote normalization, trailing period, then wraps the sentence in
  `<lang>…</lang>` tags before per-codepoint lookup in `unicode_indexer.json`
  (unmapped → row 8321). The language tags are load-bearing — omitting them
  produces slurred speech.
- **Duration predictor / text encoder**: ConvNeXt-1D stacks + VITS
  relative-position attention; GST cross-attention where keys come from a fixed
  learned `style_key` prototype and only the values come from the voice style.
- **Vector field** (`SupertonicVectorField`): 4 groups of ConvNeXt + time
  modulation (Mish MLP) + rotary text cross-attention (NeoX half-split,
  positions normalized by masked length, θ = 10·10000^(-i/32), scale 1/16) +
  GST style cross-attention. CFG (scale 4.0) and the Euler step are baked into
  the ONNX export but factored out here as `velocity()` / `eulerCFG()`.
- **Vocoder** (`SupertonicVocoder`): WaveNeXt, fully causal edge-replicate
  padding, no ISTFT — a direct 512-sample-per-frame projection head.

All LayerNorms use eps 1e-6; all convolution pads are edge-replicate.

Precision note: fp32 parity requires disabling TF32 tensor-op GEMMs
(`MLX_ENABLE_TF32=0`, set automatically on first model load — the MLX core
latches the flag at the first fp32 GEMM in the process).

## Model directory layout

```
config.json                      # {"model_type": "supertonic", ...}
duration_predictor.safetensors   # converted from ONNX (see NOTICE)
text_encoder.safetensors
vector_estimator.safetensors
vocoder.safetensors
unicode_indexer.json
voice_styles/{M1..M5,F1..F5}.json
```

## Usage

```swift
let model = try await TTS.loadModel(modelRepo: "TinyTrashLabs/supertonic-3-mlx")
let audio = try await model.generate(
    text: "The quick brown fox jumps over the lazy dog.",
    voice: "M1", refAudio: nil, refText: nil, language: "en",
    generationParameters: model.defaultGenerationParameters)
// audio: MLXArray of Float samples @ 44100 Hz
```

Preset voices only (`voice` = M1…M5, F1…F5); `refAudio`/`refText` are ignored.

## License

The SuperTonic 3 weights are licensed under the **BigScience Open RAIL-M
License** with use-based restrictions — see `LICENSE-SuperTonic` and `NOTICE`
in this directory. The Swift code is an independent reimplementation under
this repository's license.
