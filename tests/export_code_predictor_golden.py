#!/usr/bin/env python3
"""Export deterministic Qwen3-TTS CodePredictor golden data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "vendor" / "qwen3-tts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export greedy CodePredictor frame and per-head logits"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REFERENCE_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "testdata" / "code-predictor-golden",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_reference() -> tuple[Any, Any]:
    sys.path.insert(0, str(REFERENCE_ROOT))
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    return Qwen3TTSConfig, Qwen3TTSTalkerForConditionalGeneration


def save_f32(path: Path, tensor: Any) -> None:
    import torch

    data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    data.numpy().astype("<f4", copy=False).tofile(path)


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
    code_config = talker_config.code_predictor_config
    code_config._attn_implementation = "eager"

    print("loading CodePredictor reference model...")
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

    hidden_size = int(talker_config.hidden_size)
    codebook_size = int(code_config.vocab_size)
    num_groups = int(talker_config.num_code_groups)
    values = torch.arange(hidden_size, dtype=torch.float32)
    talker_hidden = (((values % 97) - 48.0) * 0.001).reshape(
        1, 1, hidden_size
    )
    c0 = 291

    predictor = talker.code_predictor
    if not isinstance(predictor.small_to_mtp_projection, torch.nn.Identity):
        raise RuntimeError("this golden exporter expects Identity MTP projection")

    codes = [c0]
    logits_rows: list[Any] = []
    hidden_rows: list[Any] = []
    with torch.inference_mode():
        c0_input = torch.tensor([[c0]], dtype=torch.long)
        c0_embedding = talker.get_input_embeddings()(c0_input)
        initial = torch.cat((talker_hidden, c0_embedding), dim=1)
        prefill = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(initial),
            use_cache=True,
            return_dict=True,
        )
        past = prefill.past_key_values
        last_hidden = prefill.last_hidden_state[:, -1]
        hidden_rows.append(last_hidden)
        step_logits = predictor.lm_head[0](last_hidden)
        logits_rows.append(step_logits)
        codes.append(int(torch.argmax(step_logits, dim=-1).item()))

        for group in range(1, num_groups - 1):
            code_input = torch.tensor([[codes[-1]]], dtype=torch.long)
            embedding = predictor.get_input_embeddings()[group - 1](code_input)
            outputs = predictor.model(
                inputs_embeds=predictor.small_to_mtp_projection(embedding),
                past_key_values=past,
                cache_position=torch.tensor([group + 1], dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1]
            hidden_rows.append(last_hidden)
            step_logits = predictor.lm_head[group](last_hidden)
            logits_rows.append(step_logits)
            codes.append(int(torch.argmax(step_logits, dim=-1).item()))

    if len(codes) != num_groups:
        raise RuntimeError(
            f"expected {num_groups} codes, generated {len(codes)}"
        )

    logits = torch.stack(logits_rows, dim=1)
    hidden = torch.stack(hidden_rows, dim=1)
    frame = np.asarray(codes, dtype="<i4")
    with torch.inference_mode():
        frame_embedding = talker.get_input_embeddings()(
            torch.tensor([[codes[0]]], dtype=torch.long)
        )
        for group in range(1, num_groups):
            frame_embedding = frame_embedding + predictor.get_input_embeddings()[
                group - 1
            ](torch.tensor([[codes[group]]], dtype=torch.long))
        frame_embedding = frame_embedding[:, 0]
    save_f32(output_dir / "code_predictor_talker_hidden.bin", talker_hidden)
    save_f32(output_dir / "code_predictor_logits.bin", logits)
    save_f32(output_dir / "code_predictor_step_hidden.bin", hidden)
    save_f32(output_dir / "code_predictor_frame_embedding.bin", frame_embedding)
    frame.tofile(output_dir / "code_predictor_frame.bin")

    manifest = {
        "format": "qwen3-tts-code-predictor-golden-v1",
        "endianness": "little",
        "talker_hidden_shape": [1, 1, hidden_size],
        "logits_shape": [1, num_groups - 1, codebook_size],
        "step_hidden_shape": [1, num_groups - 1, hidden_size],
        "frame_embedding_shape": [1, hidden_size],
        "frame_shape": [num_groups],
        "c0": c0,
        "codes": codes,
        "model_config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": int(code_config.num_hidden_layers),
            "num_attention_heads": int(code_config.num_attention_heads),
            "num_key_value_heads": int(code_config.num_key_value_heads),
            "head_dim": int(code_config.head_dim),
            "rope_theta": float(code_config.rope_theta),
            "num_code_groups": num_groups,
            "codebook_size": codebook_size,
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(f"Wrote CodePredictor golden to {output_dir}")
    print(f"  codes: {codes}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CodePredictor golden export failed: {error}", file=sys.stderr)
        raise
