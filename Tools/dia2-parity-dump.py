#!/usr/bin/env -S uv run --with torch --with numpy --with transformers --with safetensors==0.5.3 --with huggingface-hub --with sphn --with soundfile --script
"""Teacher-forced logit dump from the PyTorch Dia2 reference, for Swift parity.

Deterministic by construction: fixed tokens, no sampling, CPU, float32. The
Swift port replays the same tokens and must match within tolerance.

Usage:
  Tools/dia2-parity-dump.py --repo nari-labs/Dia2-1B \
      --dia2-src /path/to/nari-labs/dia2 \
      --out Tests/Fixtures/dia2/parity-1b.safetensors
"""
import argparse, json, sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

STEPS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nari-labs/Dia2-1B")
    ap.add_argument("--dia2-src", required=True,
                    help="clone of github.com/nari-labs/dia2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.dia2_src)
    from dia2.config import load_config
    from dia2.core.model import Dia2Model
    from dia2.core.precision import resolve_precision
    from dia2.runtime.context import load_file_into_model

    local = Path(snapshot_download(args.repo, allow_patterns=["config.json", "*.safetensors"]))
    config = load_config(local / "config.json")
    precision = resolve_precision("float32", torch.device("cpu"))
    model = Dia2Model(config, precision, device=torch.device("cpu"))
    load_file_into_model(model, str(local / "model.safetensors"), device="cpu")
    model.eval()

    data = config.data
    channels = data.channels
    depth = model.depformer.num_depth
    gen = torch.Generator().manual_seed(20260901)

    # One batch row: parity checks the network, not the CFG plumbing.
    tokens = torch.empty(1, channels, STEPS, dtype=torch.long)
    tokens[0, 0] = torch.randint(0, data.text_vocab_size, (STEPS,), generator=gen)
    tokens[0, 1] = data.text_pad_token_id
    for ch in range(2, channels):
        tokens[0, ch] = torch.randint(0, data.audio_vocab_size - 2, (STEPS,), generator=gen)

    out = {"tokens": tokens, "positions": torch.arange(STEPS, dtype=torch.long)}
    state = model.init_state(1, torch.device("cpu"), STEPS + 1)

    with torch.inference_mode():
        for t in range(STEPS):
            pos = torch.full((1, 1), t, dtype=torch.long)
            step = tokens[:, :, t : t + 1]
            hidden, action, cb0 = model.step_text(step, pos, state)
            out[f"hidden_{t}"] = hidden.clone()
            out[f"action_{t}"] = action.clone()
            out[f"cb0_{t}"] = cb0.clone()

            state.depformer.reset()
            prev = tokens[:, 2, t]
            for stage in range(depth):
                logits = model.step_audio_stage(
                    stage, prev, hidden, state,
                    main_text=tokens[:, 0, t] if stage == 0 else None,
                    second_text=tokens[:, 1, t] if stage == 0 else None,
                )
                out[f"dep_{t}_{stage}"] = logits.clone()
                prev = tokens[:, 3 + stage, t] if 3 + stage < channels else prev

    def as_bytes(s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)

    out["meta_repo"] = as_bytes(args.repo)
    out["meta_revision"] = as_bytes(json.dumps({"steps": STEPS, "depth": depth}))

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(dest))
    print(f"wrote {dest} ({len(out)} tensors, depth={depth})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
