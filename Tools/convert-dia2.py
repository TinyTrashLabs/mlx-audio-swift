#!/usr/bin/env -S uv run --with mlx --with safetensors --with huggingface_hub --script
"""Convert Nari Labs' Dia2 checkpoints to MLX bf16 / 8-bit / 4-bit and publish.

Config, tokenizer and added_tokens are copied byte-for-byte so the Swift port
reads Nari's own field names.

  Tools/convert-dia2.py --source nari-labs/Dia2-2B --dest-prefix tinytrashlabs/dia2-2b-mlx
  Tools/convert-dia2.py --source nari-labs/Dia2-2B --dest-prefix tinytrashlabs/dia2-2b-mlx --push
"""
import argparse, json, shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import HfApi, snapshot_download

COPY = ["config.json", "tokenizer.json", "tokenizer_config.json",
        "added_tokens.json", "special_tokens_map.json", "vocab.json",
        "merges.txt", "dia2_assets.json"]
GROUP_SIZE = 64

CARD = """---
license: apache-2.0
base_model: {source}
tags: [mlx, tts, dialogue, dia2]
language: [en]
---

# {name}

MLX conversion of [{source}](https://huggingface.co/{source}) ({precision}).

Dia2 is a streaming two-speaker dialogue TTS model by Nari Labs. English only,
two speakers, up to two minutes per generation. Audio is decoded through the
Kyutai Mimi codec at 24 kHz.

Converted with `Tools/convert-dia2.py` from
[TinyTrashLabs/mlx-audio-swift](https://github.com/TinyTrashLabs/mlx-audio-swift)
and consumed by [Gloam Voice Studio](https://github.com/TinyTrashLabs/gloam-voice-studio).

Licensed Apache 2.0, as is the source model. Third-party assets (the Mimi
codec) retain their own licenses.
"""


def convert(src: Path, out: Path, precision: str, source: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    weights = {}
    for shard in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}

    if precision != "bf16":
        bits = 8 if precision == "8bit" else 4
        quantized = {}
        for key, value in weights.items():
            # Quantise 2-D projection/embedding matrices only; norms, biases and
            # anything narrower than the group size stay bf16.
            if value.ndim == 2 and value.shape[-1] % GROUP_SIZE == 0:
                w, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=bits)
                quantized[key] = w
                quantized[f"{key}.scales"] = scales
                quantized[f"{key}.biases"] = biases
            else:
                quantized[key] = value
        weights = quantized
        config = json.loads((out / "config.json").read_text())
        config["quantization"] = {"group_size": GROUP_SIZE, "bits": bits}
        (out / "config.json").write_text(json.dumps(config, indent=2))

    mx.save_safetensors(str(out / "model.safetensors"), weights)
    name = out.name
    (out / "README.md").write_text(CARD.format(source=source, name=name, precision=precision))
    print(f"{name}: {len(weights)} tensors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest-prefix", required=True)
    ap.add_argument("--work", default="/tmp/dia2-convert")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    src = Path(snapshot_download(args.source))
    work = Path(args.work)
    api = HfApi()

    for precision in ("bf16", "8bit", "4bit"):
        out = work / f"{args.dest_prefix.split('/')[-1]}-{precision}"
        convert(src, out, precision, args.source)
        if args.push:
            repo = f"{args.dest_prefix}-{precision}"
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=repo)
            print(f"pushed {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
