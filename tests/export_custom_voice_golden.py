#!/usr/bin/env python3
"""Export a CustomVoice prompt and greedy speech-code golden."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "vendor" / "qwen3-tts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CustomVoice golden")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REFERENCE_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "testdata" / "custom-voice-golden",
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
    from transformers import AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    text = "你好，世界。"
    prompted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompted_text, return_tensors="pt")["input_ids"]
    role_ids = input_ids[:, :3]
    text_ids = input_ids[:, 3:-5]

    hidden_size = int(talker_config.hidden_size)
    num_groups = int(talker_config.num_code_groups)
    max_new_tokens = 50
    speaker = "serena"
    language = "chinese"
    speaker_id = int(talker_config.spk_id[speaker])
    language_id = int(talker_config.codec_language_id[language])

    predictor = talker.code_predictor

    def project_text(ids: Any) -> Any:
        return talker.text_projection(talker.get_text_embeddings()(ids))

    def codec_embedding(ids: list[int]) -> Any:
        return talker.get_input_embeddings()(
            torch.tensor([ids], dtype=torch.long)
        )

    def predict_frame(talker_hidden: Any, c0: int) -> list[int]:
        codes = [c0]
        c0_embedding = codec_embedding([c0])
        outputs = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(
                torch.cat((talker_hidden, c0_embedding), dim=1)
            ),
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1]
        step_logits = predictor.lm_head[0](last_hidden)
        codes.append(int(torch.argmax(step_logits, dim=-1).item()))
        for group in range(1, num_groups - 1):
            embedding = predictor.get_input_embeddings()[group - 1](
                torch.tensor([[codes[-1]]], dtype=torch.long)
            )
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
        result = codec_embedding(codes[:1])
        for group in range(1, num_groups):
            result = result + predictor.get_input_embeddings()[group - 1](
                torch.tensor([[codes[group]]], dtype=torch.long)
            )
        return result[:, 0]

    with torch.inference_mode():
        tts_bos, tts_eos, tts_pad = project_text(
            torch.tensor(
                [[
                    root_config.tts_bos_token_id,
                    root_config.tts_eos_token_id,
                    root_config.tts_pad_token_id,
                ]],
                dtype=torch.long,
            )
        ).chunk(3, dim=1)
        control_ids = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
            speaker_id,
            talker_config.codec_pad_id,
        ]
        control_embedding = codec_embedding(control_ids)
        control_text = torch.cat(
            (tts_pad.expand(-1, len(control_ids) - 1, -1), tts_bos),
            dim=1,
        )
        codec_pad = codec_embedding([talker_config.codec_pad_id])
        codec_bos = codec_embedding([talker_config.codec_bos_id])
        prompt_parts = (
            project_text(role_ids),
            control_text + control_embedding,
            project_text(text_ids) + codec_pad,
            tts_eos + codec_pad,
            tts_pad + codec_bos,
        )
        print("prompt part shapes:", [tuple(part.shape) for part in prompt_parts])
        prefill = torch.cat(prompt_parts, dim=1)
        prefill = prefill.contiguous()
        print(
            "prefill shape:",
            tuple(prefill.shape),
            "dtype:",
            prefill.dtype,
            "contiguous:",
            prefill.is_contiguous(),
        )

        outputs = talker.model(
            inputs_embeds=prefill,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        past_hidden = outputs.last_hidden_state[:, -1:]
        c0_logits = talker.codec_head(past_hidden)

        generated = []
        for frame_index in range(max_new_tokens):
            c0 = int(torch.argmax(c0_logits[0, -1]).item())
            if c0 == int(talker_config.codec_eos_token_id):
                break
            frame = predict_frame(past_hidden, c0)
            generated.append(frame)
            if frame_index + 1 >= max_new_tokens:
                break
            next_input = frame_embedding(frame) + tts_pad[:, 0]
            outputs = talker.model(
                inputs_embeds=next_input.unsqueeze(1),
                past_key_values=past,
                cache_position=torch.tensor(
                    [prefill.shape[1] + frame_index],
                    dtype=torch.long,
                ),
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            past_hidden = outputs.last_hidden_state
            c0_logits = talker.codec_head(past_hidden)

    save_f32(output_dir / "custom_voice_prefill.bin", prefill)
    save_f32(output_dir / "custom_voice_tts_pad.bin", tts_pad)
    codes = np.asarray(generated, dtype="<i4")
    codes.tofile(output_dir / "custom_voice_codes.bin")
    np.asarray(input_ids[0].numpy(), dtype="<i4").tofile(
        output_dir / "custom_voice_text_ids.bin"
    )

    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        model_dir / "speech_tokenizer",
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    wavs, sample_rate = speech_tokenizer.decode([
        {"audio_codes": torch.tensor(generated, dtype=torch.long)}
    ])
    audio = np.asarray(wavs[0], dtype="<f4")
    audio.tofile(output_dir / "custom_voice_audio.bin")

    manifest = {
        "format": "qwen3-tts-custom-voice-golden-v1",
        "endianness": "little",
        "text": text,
        "speaker": speaker,
        "language": "Chinese",
        "text_ids_shape": [len(input_ids[0])],
        "prefill_shape": [1, len(prefill[0]), hidden_size],
        "tts_pad_shape": [1, hidden_size],
        "codes_shape": [len(generated), num_groups],
        "audio_shape": [len(audio)],
        "sample_rate": int(sample_rate),
        "max_new_tokens": max_new_tokens,
        "codes": generated,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(f"Wrote CustomVoice golden to {output_dir}")
    print(f"  prompt tokens: {len(input_ids[0])}")
    print(f"  generated frames: {len(generated)}")
    print(f"  audio samples: {len(audio)} at {sample_rate} Hz")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CustomVoice golden export failed: {error}", file=sys.stderr)
        raise
