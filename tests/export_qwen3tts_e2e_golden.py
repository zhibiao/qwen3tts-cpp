#!/usr/bin/env python3
"""Export text-to-codes/audio golden data for CustomVoice end-to-end tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "docs" / "qwen3-tts"
DEFAULT_OUTPUT = PROJECT_ROOT / "testdata" / "qwen3tts-e2e-golden"

CASES_0_6 = [
    ("short_zh", "你好。"),
    ("punctuation_zh", "你好，世界！今天天气怎么样？"),
    ("english", "Hello, world! How are you?"),
    ("mixed", "你好，OpenAI。今天讨论 Qwen3-TTS（12Hz）与 C++/ggml，OK?"),
    (
        "long_mixed",
        "今天的测试包含较长的中文句子，也包含 English words、数字 12345、"
        "标点符号：逗号，句号。问号？感叹号！以及括号（测试）和斜杠 A/B。"
        "我们继续增加上下文，以覆盖不同长度的输入。",
    ),
]

CASES_1_7 = [
    ("short_mixed", "你好，OpenAI。Hello, Qwen3-TTS!"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models",
        choices=("all", "0.6b", "1.7b"),
        default="all",
    )
    return parser.parse_args()


def install_optional_stubs() -> None:
    package_names = {"torchaudio", "torchaudio.compliance"}
    for module_name in (
        "sox",
        "onnxruntime",
        "torchaudio",
        "torchaudio.compliance",
        "torchaudio.compliance.kaldi",
    ):
        if importlib.util.find_spec(module_name) is not None:
            continue
        module = types.ModuleType(module_name)
        is_package = module_name in package_names
        module.__spec__ = ModuleSpec(
            module_name,
            loader=None,
            is_package=is_package,
        )
        if is_package:
            module.__path__ = []
        sys.modules[module_name] = module
    torchaudio = sys.modules.get("torchaudio")
    compliance = sys.modules.get("torchaudio.compliance")
    if torchaudio is not None and compliance is not None:
        torchaudio.compliance = compliance
        compliance.kaldi = sys.modules.get("torchaudio.compliance.kaldi")
    if "einops" not in sys.modules and importlib.util.find_spec("einops") is None:
        einops = types.ModuleType("einops")
        einops.__spec__ = ModuleSpec("einops", loader=None)
        einops.rearrange = lambda value, *_args, **_kwargs: value
        einops.repeat = lambda value, *_args, **_kwargs: value
        sys.modules["einops"] = einops


class CodeCollector:
    def __init__(self) -> None:
        self.codes: list[int] = []

    def put(self, codes: object) -> None:
        self.codes.extend(int(value) for value in codes[0].tolist())

    def end(self) -> None:
        pass


def export_model(
    model_dir: Path,
    cases: list[tuple[str, str]],
    output_dir: Path,
    max_new_tokens: int,
    device: str,
) -> list[dict[str, object]]:
    install_optional_stubs()
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel

    model_dtype = torch.bfloat16 if "1.7B" in model_dir.name else torch.float32
    load_kwargs: dict[str, object] = {
        "local_files_only": True,
        "dtype": model_dtype,
        "attn_implementation": "eager",
    }
    if device != "cpu":
        load_kwargs["device_map"] = device
    model = Qwen3TTSModel.from_pretrained(model_dir, **load_kwargs)
    metadata = []
    for name, text in cases:
        collector = CodeCollector()
        model.generate_custom_voice(
            text=text,
            speaker="serena",
            language="Chinese",
            max_new_tokens=max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
            code_streamer=collector,
            _decode_final_audio=False,
        )
        frame_count = len(collector.codes) // 16
        if frame_count == 0:
            raise RuntimeError(f"reference generated no frames for {name}")
        codes = np.asarray(
            collector.codes,
            dtype="<i4",
        ).reshape(frame_count, 16)
        wavs, sample_rate = model.model.speech_tokenizer.decode(
            [{"audio_codes": torch.tensor(codes, dtype=torch.long)}]
        )
        audio = np.asarray(wavs[0], dtype="<f4")
        case_dir = output_dir / model_dir.name
        case_dir.mkdir(parents=True, exist_ok=True)
        codes.tofile(case_dir / f"{name}.codes.bin")
        audio.tofile(case_dir / f"{name}.audio.bin")
        metadata.append(
            {
                "name": name,
                "text": text,
                "speaker": "serena",
                "language": "Chinese",
                "frames": frame_count,
                "groups": 16,
                "sample_rate": int(sample_rate),
                "audio_samples": int(audio.size),
                "codes": codes.tolist(),
                "codes_file": f"{model_dir.name}/{name}.codes.bin",
                "audio_file": f"{model_dir.name}/{name}.audio.bin",
            }
        )
        print(f"Exported {model_dir.name}/{name}: {frame_count} frames")
    return metadata


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REFERENCE_ROOT))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = REFERENCE_ROOT / "models"

    manifest_path = output_dir / "manifest.json"
    cases: list[dict[str, object]] = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            cases = list(json.load(file).get("cases", []))
    if args.models in ("all", "0.6b"):
        cases = [
            case for case in cases
            if "0.6B" not in str(case.get("codes_file", ""))
        ]
        cases += export_model(
            models / "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            CASES_0_6,
            output_dir,
            args.max_new_tokens,
            args.device,
        )
    if args.models in ("all", "1.7b"):
        cases = [
            case for case in cases
            if "1.7B" not in str(case.get("codes_file", ""))
        ]
        cases += export_model(
            models / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            CASES_1_7,
            output_dir,
            args.max_new_tokens,
            args.device,
        )
    manifest = {
        "format": "qwen3-tts-end-to-end-golden-v1",
        "endianness": "little",
        "max_new_tokens": args.max_new_tokens,
        "greedy": True,
        "cases": cases,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote end-to-end golden manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
