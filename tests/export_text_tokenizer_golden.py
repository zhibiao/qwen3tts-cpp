#!/usr/bin/env python3
"""Export text tokenizer cases from the Python reference tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "docs"
    / "qwen3-tts"
    / "models"
    / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "testdata" / "text-tokenizer-golden" / "cases.json"

RAW_CASES = [
    ("short_zh", "你好。"),
    ("punctuation_zh", "你好，世界！今天天气怎么样？"),
    ("english", "Hello, world! How are you?"),
    ("mixed", "你好，OpenAI。今天讨论 Qwen3-TTS（12Hz）与 C++/ggml，OK?"),
    (
        "long_mixed",
        "今天的测试包含较长的中文句子，也包含 English words、数字 12345、"
        "标点符号：逗号，句号。问号？感叹号！以及括号（测试）和斜杠 A/B。"
        "我们继续增加上下文，以覆盖不同长度的输入，并检查 tokenizer、prompt、"
        "Talker prefill 和 KV cache 是否能正常处理。",
    ),
    (
        "punctuation_mixed",
        "Wait... really?! 是的——真的吗？(100%) [OK]; \"quoted\"、‘单引号’。",
    ),
    ("whitespace", "第一行\n第二行\tTab 和   多空格。"),
    ("emoji", "Hello 😊，今天 OK 👌。"),
]

WRAPPED_CASES = [
    (
        "assistant_mixed",
        "assistant",
        "你好，OpenAI。今天讨论 Qwen3-TTS（12Hz）与 C++/ggml，OK?",
    ),
    ("reference_english", "reference", "Hello, world! How are you?"),
    ("instruction_mixed", "instruction", "Speak slowly, 用温和的语气，OK?"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(),
        local_files_only=True,
        fix_mistral_regex=True,
    )

    cases = []
    for name, text in RAW_CASES:
        cases.append(
            {
                "name": name,
                "kind": "raw",
                "text": text,
                "ids": tokenizer.encode(text, add_special_tokens=False),
            }
        )

    prefixes = {
        "assistant": "<|im_start|>assistant\n",
        "reference": "<|im_start|>assistant\n",
        "instruction": "<|im_start|>user\n",
    }
    suffixes = {
        "assistant": "<|im_end|>\n<|im_start|>assistant\n",
        "reference": "<|im_end|>\n",
        "instruction": "<|im_end|>\n",
    }
    for name, kind, text in WRAPPED_CASES:
        prompted = prefixes[kind] + text + suffixes[kind]
        cases.append(
            {
                "name": name,
                "kind": kind,
                "text": text,
                "ids": tokenizer.encode(prompted, add_special_tokens=False),
            }
        )

    payload = {
        "format": "qwen3-tts-text-tokenizer-golden-v1",
        "model_dir": str(args.model_dir.resolve()),
        "cases": cases,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {len(cases)} tokenizer cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
