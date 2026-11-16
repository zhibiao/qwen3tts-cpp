#!/usr/bin/env python3
"""Run the vendored Python CustomVoice reference and write a WAV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "vendor" / "qwen3-tts"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REFERENCE_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument("--text", default="你好，世界。")
    parser.add_argument("--speaker", default="serena")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "testdata" / "reference-customvoice.wav",
    )
    parser.add_argument(
        "--codes-output",
        type=Path,
        help="Optional little-endian int32 [frames, groups] code matrix",
    )
    parser.add_argument(
        "--talker-hidden-output",
        type=Path,
        help="Optional little-endian float32 [frames, hidden] Talker states",
    )
    parser.add_argument(
        "--prefill-input-output",
        type=Path,
        help="Optional little-endian float32 CustomVoice prefill embeddings",
    )
    parser.add_argument(
        "--prefill-hidden-output",
        type=Path,
        help="Optional little-endian float32 Talker prefill hidden states",
    )
    parser.add_argument(
        "--predictor-logits-output",
        type=Path,
        help="Optional little-endian float32 CodePredictor logits",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REFERENCE_ROOT))
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    )
    captured_hidden: list[np.ndarray] = []
    hook = None
    captured_prefill: list[np.ndarray] = []
    prefill_hook = None
    captured_predictor_logits: list[np.ndarray] = []
    predictor_logits_hook = None
    if args.talker_hidden_output is not None:
        def capture_predictor_input(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> None:
            inputs = kwargs.get("inputs_embeds")
            if (
                isinstance(inputs, torch.Tensor)
                and inputs.ndim == 3
                and inputs.shape[1] == 2
            ):
                captured_hidden.append(
                    inputs[0, 0].detach().to("cpu", dtype=torch.float32).numpy()
                )

        hook = model.model.talker.code_predictor.register_forward_pre_hook(
            capture_predictor_input,
            with_kwargs=True,
        )
    if args.predictor_logits_output is not None:
        def capture_predictor_logits(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: object,
        ) -> None:
            logits = getattr(output, "logits", None)
            if isinstance(logits, torch.Tensor):
                captured_predictor_logits.append(
                    logits[0, -1].detach().to(
                        "cpu",
                        dtype=torch.float32,
                    ).numpy()
                )

        predictor_logits_hook = (
            model.model.talker.code_predictor.register_forward_hook(
                capture_predictor_logits
            )
        )
    if (
        args.prefill_input_output is not None
        or args.prefill_hidden_output is not None
    ):
        def capture_prefill(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            inputs = kwargs.get("inputs_embeds")
            if (
                isinstance(inputs, torch.Tensor)
                and inputs.ndim == 3
                and inputs.shape[1] > 1
                and not captured_prefill
            ):
                captured_prefill.append(
                    inputs[0].detach().to("cpu", dtype=torch.float32).numpy()
                )
                if args.prefill_hidden_output is not None:
                    hidden_states = getattr(output, "last_hidden_state", None)
                    if not isinstance(hidden_states, torch.Tensor):
                        raise RuntimeError("Talker prefill output has no hidden states")
                    args.prefill_hidden_output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    hidden_states[0].detach().to(
                        "cpu",
                        dtype=torch.float32,
                    ).numpy().astype("<f4", copy=False).tofile(
                        args.prefill_hidden_output
                    )

        prefill_hook = model.model.talker.model.register_forward_hook(
            capture_prefill,
            with_kwargs=True,
        )
    class CodeCollector:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def put(self, codes: torch.Tensor) -> None:
            values = codes.detach().to(device="cpu", dtype=torch.int64)
            self.frames.extend(values.numpy().astype("<i4", copy=False))

        def end(self) -> None:
            pass

    collector = CodeCollector() if args.codes_output is not None else None
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        subtalker_dosample=not args.greedy,
        code_streamer=collector,
        _decode_final_audio=args.codes_output is None,
    )
    if hook is not None:
        hook.remove()
        args.talker_hidden_output.parent.mkdir(parents=True, exist_ok=True)
        hidden = np.asarray(captured_hidden, dtype="<f4")
        hidden.tofile(args.talker_hidden_output)
        print(f"Wrote {args.talker_hidden_output} {hidden.shape}")
    if prefill_hook is not None:
        prefill_hook.remove()
        if args.prefill_input_output is not None:
            args.prefill_input_output.parent.mkdir(parents=True, exist_ok=True)
            prefill = np.asarray(captured_prefill[0], dtype="<f4")
            prefill.tofile(args.prefill_input_output)
            print(f"Wrote {args.prefill_input_output} {prefill.shape}")
    if predictor_logits_hook is not None:
        predictor_logits_hook.remove()
        args.predictor_logits_output.parent.mkdir(parents=True, exist_ok=True)
        predictor_logits = np.asarray(
            captured_predictor_logits,
            dtype="<f4",
        )
        predictor_logits.tofile(args.predictor_logits_output)
        print(
            f"Wrote {args.predictor_logits_output} "
            f"{predictor_logits.shape}"
        )
    if args.codes_output is not None:
        args.codes_output.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(collector.frames, dtype="<i4")
        matrix.tofile(args.codes_output)
        print(
            f"Wrote {args.codes_output} "
            f"({matrix.shape[0]} frames, {matrix.shape[1]} groups)"
        )
        print(f"Codes: {matrix.tolist()}")
    else:
        sf.write(args.output, wavs[0], sample_rate)
        print(f"Wrote {args.output} ({len(wavs[0])} samples at {sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
