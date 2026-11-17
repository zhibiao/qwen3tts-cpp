#!/usr/bin/env python3
"""Convert Qwen3-TTS Talker GGUF matrix weights to a compact tensor format.

SpeechTokenizer GGUF files are intentionally unsupported. They must remain in
their original F32 form so SpeechDecoder precision is not affected by
conversion.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class QuantizationError(RuntimeError):
    pass


FILE_TYPES = {
    "q8_0": 7,
    "bf16": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3-TTS Talker GGUF file"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input Talker GGUF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GGUF; defaults to <input>-<type>.gguf",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(FILE_TYPES),
        help="Output tensor format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def load_gguf() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from gguf import (
            GGMLQuantizationType,
            GGUFReader,
            GGUFValueType,
            GGUFWriter,
        )
        from gguf import quants
    except ModuleNotFoundError as error:
        raise QuantizationError(
            f"missing Python dependency: {error.name}\n"
            "Install requirements.txt before converting."
        ) from error
    return (
        np,
        (GGUFReader, GGUFValueType, GGUFWriter),
        quants,
        GGMLQuantizationType,
    )


def metadata_value(field: Any) -> Any:
    value = field.contents()
    if isinstance(value, list):
        return [
            item.item() if hasattr(item, "item") else item
            for item in value
        ]
    return value.item() if hasattr(value, "item") else value


def copy_metadata(
    reader: Any,
    writer: Any,
    gguf_value_type: Any,
    preset: str,
) -> str:
    arch = reader.fields["general.architecture"].contents()
    if not isinstance(arch, str) or not arch:
        raise QuantizationError("missing general.architecture")

    synthetic_keys = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
    override_keys = {
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
        "qwen3_tts.quantization.preset",
        "qwen3_tts.quantization.policy",
    }
    for key, field in reader.fields.items():
        if key in synthetic_keys or key in override_keys:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf_value_type.ARRAY else None
        writer.add_key_value(
            key,
            metadata_value(field),
            value_type,
            sub_type,
        )

    writer.add_key_value(
        "general.file_type",
        FILE_TYPES[preset],
        gguf_value_type.UINT32,
    )
    writer.add_key_value(
        "general.quantization_version",
        2,
        gguf_value_type.UINT32,
    )
    writer.add_key_value(
        "qwen3_tts.quantization.preset",
        preset,
        gguf_value_type.STRING,
    )
    writer.add_key_value(
        "qwen3_tts.quantization.policy",
        f"talker-{preset}-matrix-weights-v1",
        gguf_value_type.STRING,
    )
    return arch


def tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def dequantize_tensor(np: Any, quants: Any, tensor: Any) -> Any:
    shape = tuple(reversed(tensor_shape(tensor)))
    if tensor.tensor_type.name == "BF16":
        words = np.asarray(tensor.data, dtype=np.uint8).view(np.uint16)
        bits = words.astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)
    data = quants.dequantize(tensor.data, tensor.tensor_type)
    return np.asarray(data, dtype=np.float32).reshape(shape)


def is_eligible_weight(np: Any, tensor: Any, block_size: int) -> bool:
    shape = tensor_shape(tensor)
    if tensor.name.find("norm") != -1:
        return False
    if not tensor.name.endswith(".weight"):
        return False
    if len(shape) < 2:
        return False
    if shape[0] % block_size != 0:
        return False
    return all(int(value) > 0 for value in shape)


def float32_to_bf16_bits(np: Any, data: Any) -> Any:
    values = np.ascontiguousarray(data, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    rounded = bits + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def convert_tensor(
    np: Any,
    quants: Any,
    tensor: Any,
    qtypes: Any,
    preset: str,
) -> Any | None:
    if preset == "q8_0":
        qtype = qtypes.Q8_0
        block_size, _ = quants.GGML_QUANT_SIZES[qtype]
    elif preset == "bf16":
        qtype = qtypes.BF16
        block_size = 1
    else:
        raise QuantizationError(f"unsupported output type: {preset}")

    if not is_eligible_weight(np, tensor, block_size):
        return None
    data = dequantize_tensor(np, quants, tensor)
    if preset == "q8_0":
        converted = quants.quantize(data, qtype)
    else:
        converted = float32_to_bf16_bits(np, data).view(np.uint8)
    return converted, qtype


def write_gguf(writer: Any, temporary: Path, output: Path) -> None:
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, output)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        temporary.unlink(missing_ok=True)
        raise


def quantize(args: argparse.Namespace) -> None:
    np, gguf_api, quants, qtypes = load_gguf()
    reader_type, gguf_value_type, writer_type = gguf_api

    source = args.input.resolve()
    if not source.is_file():
        raise QuantizationError(f"input file does not exist: {source}")
    output = (
        args.output.resolve()
        if args.output is not None
        else source.with_name(f"{source.stem}-{args.type}.gguf")
    )
    if output == source:
        raise QuantizationError("output must differ from input")
    if output.exists() and not args.overwrite:
        raise QuantizationError(f"output already exists: {output}")

    reader = reader_type(source)
    role_field = reader.fields.get("qwen3_tts.file_role")
    role = None if role_field is None else role_field.contents()
    if role != "talker":
        raise QuantizationError(
            "this quantizer only supports Talker GGUF; SpeechTokenizer must "
            f"remain unquantized F32 (input role is {role!r})"
        )

    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    arch = reader.fields["general.architecture"].contents()
    writer = writer_type(str(temporary), arch, use_temp_file=True)
    copy_metadata(reader, writer, gguf_value_type, args.type)

    counts: Counter[str] = Counter()
    for index, tensor in enumerate(reader.tensors, 1):
        try:
            result = convert_tensor(
                np,
                quants,
                tensor,
                qtypes,
                args.type,
            )
        except Exception as error:
            raise QuantizationError(
                f"failed to convert tensor {tensor.name}: {error}"
            ) from error
        if result is None:
            writer.add_tensor(
                tensor.name,
                tensor.data,
                raw_shape=(
                    tensor.data.shape
                    if tensor.data.dtype == np.uint8
                    else tensor_shape(tensor)
                ),
                raw_dtype=tensor.tensor_type,
            )
            counts[tensor.tensor_type.name] += 1
        else:
            quantized, qtype = result
            writer.add_tensor(
                tensor.name,
                quantized,
                raw_shape=quantized.shape,
                raw_dtype=qtype,
            )
            counts[qtype.name] += 1
        if index % 50 == 0 or index == len(reader.tensors):
            print(
                f"Processed {index}/{len(reader.tensors)} tensors",
                file=sys.stderr,
            )

    write_gguf(writer, temporary, output)
    print(f"Wrote {output}")
    print(f"Size: {output.stat().st_size} bytes")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


def main() -> int:
    args = parse_args()
    try:
        quantize(args)
    except QuantizationError as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Conversion failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
