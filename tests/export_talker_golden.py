#!/usr/bin/env python3
"""Export deterministic Qwen3-TTS Talker golden tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "docs" / "qwen3-tts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Talker hidden/logits and KV-cache golden tensors"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REFERENCE_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "testdata" / "talker-golden",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_reference() -> tuple[Any, Any]:
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration,
        )
    except ImportError as error:
        raise RuntimeError(
            f"failed to import reference implementation from {REFERENCE_ROOT}: {error}"
        ) from error
    return Qwen3TTSConfig, Qwen3TTSTalkerForConditionalGeneration


def save_f32(path: Path, tensor: Any) -> None:
    data = tensor.detach().to(device="cpu", dtype=__import__("torch").float32)
    data = data.contiguous().numpy().astype("<f4", copy=False)
    data.tofile(path)


def save_i32(path: Path, value: int) -> None:
    import numpy as np

    np.asarray([value], dtype="<i4").tofile(path)


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError(f"model checkpoint not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    Qwen3TTSConfig, TalkerModel = import_reference()
    root_config = Qwen3TTSConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    talker_config = root_config.talker_config
    talker_config._attn_implementation = "eager"

    print("loading Talker reference model...")
    talker = TalkerModel(talker_config)
    state = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if name.startswith("talker."):
                state[name.removeprefix("talker.")] = tensors.get_tensor(name)
    missing, unexpected = talker.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Talker checkpoint mismatch: "
            f"missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )
    talker.eval().to(dtype=torch.float32)

    frames = 2
    hidden_size = int(talker_config.hidden_size)
    vocab_size = int(talker_config.vocab_size)
    values = torch.arange(frames * hidden_size, dtype=torch.float32)
    inputs = (((values % 97) - 48.0) * 0.001).reshape(1, frames, hidden_size)
    step_values = torch.arange(hidden_size, dtype=torch.float32)
    step_input = (((step_values % 31) - 15.0) * 0.002).reshape(1, 1, hidden_size)

    print("running Talker prefill...")
    with torch.inference_mode():
        prefill = talker.model(
            inputs_embeds=inputs,
            use_cache=False,
            return_dict=True,
        )
        prefill_hidden = prefill.last_hidden_state
        prefill_logits = talker.codec_head(prefill_hidden)
        c0 = int(torch.argmax(prefill_logits[0, -1]).item())

        cached_prefill = talker.model(
            inputs_embeds=inputs,
            use_cache=True,
            return_dict=True,
        )
        decode = talker.model(
            inputs_embeds=step_input,
            past_key_values=cached_prefill.past_key_values,
            cache_position=torch.tensor([frames], dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
        decode_hidden = decode.last_hidden_state
        decode_logits = talker.codec_head(decode_hidden)

    tensors = {
        "talker_input.bin": (inputs, [1, frames, hidden_size]),
        "talker_hidden.bin": (prefill_hidden, [1, frames, hidden_size]),
        "talker_logits.bin": (prefill_logits, [1, frames, vocab_size]),
        "talker_decode_step_input.bin": (step_input, [1, 1, hidden_size]),
        "talker_decode_step_hidden.bin": (decode_hidden, [1, 1, hidden_size]),
        "talker_decode_step_logits.bin": (decode_logits, [1, 1, vocab_size]),
    }
    files: dict[str, dict[str, Any]] = {}
    for filename, (tensor, shape) in tensors.items():
        save_f32(output_dir / filename, tensor)
        files[filename] = {"shape": shape, "dtype": "float32"}
    save_i32(output_dir / "talker_c0.bin", c0)
    files["talker_c0.bin"] = {"shape": [1], "dtype": "int32"}

    manifest = {
        "format": "qwen3-tts-talker-golden-v1",
        "endianness": "little",
        "batch_size": 1,
        "frames": frames,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "prefill_positions": list(range(frames)),
        "decode_positions": [frames],
        "c0": c0,
        "model_config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": int(talker_config.num_hidden_layers),
            "num_attention_heads": int(talker_config.num_attention_heads),
            "num_key_value_heads": int(talker_config.num_key_value_heads),
            "head_dim": int(talker_config.head_dim),
            "rope_theta": float(talker_config.rope_theta),
            "mrope_section": list(talker_config.rope_scaling["mrope_section"]),
            "interleaved": bool(talker_config.rope_scaling["interleaved"]),
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "files": files,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(f"Wrote Talker golden fixtures to {output_dir}")
    print(f"  input: [{frames}, {hidden_size}]")
    print(f"  logits: [{frames}, {vocab_size}]")
    print(f"  c0: {c0}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Talker golden export failed: {error}", file=sys.stderr)
        raise
