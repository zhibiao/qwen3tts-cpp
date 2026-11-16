#!/usr/bin/env python3
"""Export multi-frame Talker + CodePredictor greedy golden codes."""

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
        description="Export multi-frame Qwen3TTS speech-code golden data"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REFERENCE_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "testdata" / "qwen3tts-stream-golden",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_f32(path: Path, tensor: Any) -> None:
    import torch

    data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    data.numpy().astype("<f4", copy=False).tofile(path)


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(REFERENCE_ROOT))
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError(f"model checkpoint not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    root_config = Qwen3TTSConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    talker_config = root_config.talker_config
    talker_config._attn_implementation = "eager"
    talker_config.code_predictor_config._attn_implementation = "eager"
    talker = Qwen3TTSTalkerForConditionalGeneration(talker_config)
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
    codebook_size = int(talker_config.code_predictor_config.vocab_size)
    num_groups = int(talker_config.num_code_groups)
    eos_id = int(talker_config.codec_eos_token_id)
    prefill_frames = 2
    trailing_frames = 2
    max_new_tokens = 3

    prefill_values = torch.arange(prefill_frames * hidden_size, dtype=torch.float32)
    prefill = (((prefill_values % 89) - 44.0) * 0.001).reshape(
        1, prefill_frames, hidden_size
    )
    trailing_values = torch.arange(trailing_frames * hidden_size, dtype=torch.float32)
    trailing = (((trailing_values % 53) - 26.0) * 0.0005).reshape(
        1, trailing_frames, hidden_size
    )
    pad_values = torch.arange(hidden_size, dtype=torch.float32)
    pad = (((pad_values % 41) - 20.0) * 0.0003).reshape(1, hidden_size)

    predictor = talker.code_predictor

    def predict_frame(talker_hidden: Any, c0: int) -> list[int]:
        codes = [c0]
        c0_input = torch.tensor([[c0]], dtype=torch.long)
        c0_embedding = talker.get_input_embeddings()(c0_input)
        initial = torch.cat((talker_hidden, c0_embedding), dim=1)
        outputs = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(initial),
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1]
        step_logits = predictor.lm_head[0](last_hidden)
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
            step_logits = predictor.lm_head[group](last_hidden)
            codes.append(int(torch.argmax(step_logits, dim=-1).item()))
        return codes

    def frame_embedding(codes: list[int]) -> Any:
        result = talker.get_input_embeddings()(
            torch.tensor([[codes[0]]], dtype=torch.long)
        )
        for group in range(1, num_groups):
            result = result + predictor.get_input_embeddings()[group - 1](
                torch.tensor([[codes[group]]], dtype=torch.long)
            )
        return result[:, 0]

    generated: list[list[int]] = []
    with torch.inference_mode():
        outputs = talker.model(
            inputs_embeds=prefill,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        past_hidden = outputs.last_hidden_state[:, -1:]
        c0_logits = talker.codec_head(past_hidden)

        for frame_index in range(max_new_tokens):
            c0 = int(torch.argmax(c0_logits[0, -1]).item())
            if c0 == eos_id:
                break
            frame = predict_frame(past_hidden, c0)
            generated.append(frame)
            if frame_index + 1 >= max_new_tokens:
                break

            embedding = frame_embedding(frame)
            condition = (
                trailing[:, frame_index]
                if frame_index < trailing_frames
                else pad
            )
            next_input = (embedding + condition).unsqueeze(1)
            outputs = talker.model(
                inputs_embeds=next_input,
                past_key_values=past,
                cache_position=torch.tensor(
                    [prefill_frames + frame_index],
                    dtype=torch.long,
                ),
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            past_hidden = outputs.last_hidden_state
            c0_logits = talker.codec_head(past_hidden)

    code_matrix = np.asarray(generated, dtype="<i4")
    frame_path = output_dir / "qwen3tts_codes.bin"
    code_matrix.tofile(frame_path)
    save_f32(output_dir / "qwen3tts_prefill.bin", prefill)
    save_f32(output_dir / "qwen3tts_trailing.bin", trailing)
    save_f32(output_dir / "qwen3tts_pad.bin", pad)

    manifest = {
        "format": "qwen3-tts-stream-golden-v1",
        "endianness": "little",
        "prefill_shape": [1, prefill_frames, hidden_size],
        "trailing_shape": [1, trailing_frames, hidden_size],
        "pad_shape": [1, hidden_size],
        "codes_shape": [len(generated), num_groups],
        "max_new_tokens": max_new_tokens,
        "codec_eos_token_id": eos_id,
        "generated_frames": len(generated),
        "codes": generated,
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

    print(f"Wrote multi-frame golden to {output_dir}")
    print(f"  generated frames: {len(generated)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Qwen3TTS stream golden export failed: {error}", file=sys.stderr)
        raise
