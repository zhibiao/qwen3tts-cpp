#!/usr/bin/env python3
"""Convert Qwen3-TTS 12Hz CustomVoice HF weights to GGUF.

The C++ runtime only consumes the resulting GGUF file. Python dependencies are
used exclusively by this conversion tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS 12Hz CustomVoice safetensors to GGUF"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Qwen3-TTS-12Hz-0.6B-CustomVoice model directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for talker and speech-tokenizer GGUF files",
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Exclude SpeechTokenizerEncoder tensors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from safetensors import safe_open
        from gguf import GGUFWriter
    except ModuleNotFoundError as error:
        raise ConversionError(
            f"missing Python dependency: {error.name}\n"
            "Install PyTorch, safetensors and llama.cpp gguf-py before conversion."
        ) from error
    return np, torch, safe_open, GGUFWriter


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"failed to read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConversionError(f"JSON root must be an object: {path}")
    return value


def build_tokenizer(
    model_dir: Path,
    tokenizer_config: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[int]]:
    vocab = load_json(model_dir / "vocab.json")
    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
    if not isinstance(added_tokens_decoder, dict):
        raise ConversionError("added_tokens_decoder must be an object")

    max_id = -1
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ConversionError("vocab.json must map strings to integer IDs")
        max_id = max(max_id, token_id)
    for token_id_text in added_tokens_decoder:
        try:
            max_id = max(max_id, int(token_id_text))
        except ValueError as error:
            raise ConversionError(f"invalid added token ID: {token_id_text}") from error
    if max_id < 0:
        raise ConversionError("tokenizer vocabulary is empty")

    tokens: list[str | None] = [None] * (max_id + 1)
    for token, token_id in vocab.items():
        if token_id < 0 or token_id >= len(tokens):
            raise ConversionError(f"vocab token ID out of range: {token}")
        if tokens[token_id] is not None:
            raise ConversionError(f"duplicate vocab token ID: {token_id}")
        tokens[token_id] = token

    added_names: list[str] = []
    added_ids: list[int] = []
    for token_id_text, description in added_tokens_decoder.items():
        token_id = int(token_id_text)
        if not isinstance(description, dict):
            raise ConversionError(f"invalid added token descriptor: {token_id_text}")
        content = description.get("content")
        if not isinstance(content, str):
            raise ConversionError(f"added token has no string content: {token_id_text}")
        if token_id < 0 or token_id >= len(tokens):
            raise ConversionError(f"added token ID out of range: {token_id_text}")
        if tokens[token_id] is not None:
            raise ConversionError(f"added token conflicts with vocab ID: {token_id_text}")
        tokens[token_id] = content
        added_names.append(content)
        added_ids.append(token_id)

    for token_id, token in enumerate(tokens):
        if token is None:
            raise ConversionError(f"non-contiguous tokenizer IDs; missing ID {token_id}")

    try:
        with (model_dir / "merges.txt").open("r", encoding="utf-8") as file:
            merges = [line.rstrip("\r\n") for line in file if line.rstrip("\r\n")]
    except OSError as error:
        raise ConversionError(f"failed to read merges.txt: {error}") from error
    if not merges:
        raise ConversionError("merges.txt is empty")

    return (
        [token for token in tokens if token is not None],
        merges,
        added_names,
        added_ids,
    )


def named_id_arrays(mapping: dict[str, Any]) -> tuple[list[str], list[int]]:
    if not isinstance(mapping, dict):
        return [], []
    entries: list[tuple[str, int]] = []
    for name, token_id in mapping.items():
        if not isinstance(name, str) or not isinstance(token_id, int):
            raise ConversionError("named token mapping must contain string/int pairs")
        entries.append((name, token_id))
    entries.sort(key=lambda item: item[0])
    return [name for name, _ in entries], [token_id for _, token_id in entries]


def speaker_arrays(
    talker: dict[str, Any],
) -> tuple[list[str], list[int], list[str]]:
    names, ids = named_id_arrays(talker.get("spk_id", {}))
    dialect_mapping = talker.get("spk_is_dialect", {})
    if not isinstance(dialect_mapping, dict):
        raise ConversionError("spk_is_dialect must be an object")
    dialects: list[str] = []
    for name in names:
        dialect = dialect_mapping.get(name, False)
        if dialect is False or dialect is None:
            dialects.append("")
        elif isinstance(dialect, str):
            dialects.append(dialect)
        else:
            raise ConversionError(
                f"spk_is_dialect value must be false or a string: {name}"
            )
    return names, ids, dialects


def map_tensor_name(name: str) -> str:
    if name.startswith("talker.code_predictor.model."):
        return "talker.code_predictor." + name[len("talker.code_predictor.model.") :]
    if name.startswith("talker.model."):
        return "talker." + name[len("talker.model.") :]
    if name.startswith("decoder."):
        rest = "sdec." + name[len("decoder.") :]
        substitutions = (
            (".pre_transformer.", ".pt."),
            (".layers.", ".l."),
            (".post_attention_layernorm.", ".post_norm."),
            (".self_attn_layer_scale.", ".sa_scale."),
            (".self_attn.", ".sa."),
            (".quantizer.", ".q."),
            (".rvq_first.", ".q0."),
            (".rvq_rest.", ".q1."),
            (".vq.", ".rvq."),
            ("._codebook.", ".cb."),
            (".cluster_usage", ".usage"),
            (".embedding_sum", ".sum"),
        )
        for source, target in substitutions:
            rest = rest.replace(source, target)
        return rest
    if name.startswith("encoder."):
        rest = "senc." + name[len("encoder.") :]
        substitutions = (
            (".encoder_transformer.", ".et."),
            (".layers.", ".l."),
            (".post_attention_layernorm.", ".post_norm."),
            (".quantizer.", ".q."),
            (".semantic_residual_vector_quantizer.", ".sem."),
            (".acoustic_residual_vector_quantizer.", ".ac."),
            (".codebook.", ".cb."),
            (".cluster_usage", ".usage"),
            (".embed_sum", ".sum"),
            (".initialized", ".init"),
        )
        for source, target in substitutions:
            rest = rest.replace(source, target)
        return rest
    if name.startswith(("talker.code_predictor.", "talker.codec_head.", "talker.text_projection.")):
        return name
    raise ConversionError(f"unmapped safetensors tensor: {name}")


def add_talker_metadata(
    writer: Any,
    model_name: str,
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    generation_config: dict[str, Any],
    tokens: list[str],
    merges: list[str],
    added_names: list[str],
    added_ids: list[int],
    file_type: int,
) -> None:
    talker = model_config["talker_config"]
    code_predictor = talker["code_predictor_config"]
    talker_hidden = int(talker["hidden_size"])
    predictor_hidden = int(code_predictor["hidden_size"])
    has_input_projection = talker_hidden != predictor_hidden

    writer.add_name(model_name)
    writer.add_file_type(file_type)
    writer.add_quantization_version(2)

    writer.add_uint32("qwen3_tts.schema_version", 3)
    writer.add_string("qwen3_tts.file_role", "talker")
    writer.add_string("qwen3_tts.model_type", str(model_config["tts_model_type"]))
    writer.add_string("qwen3_tts.model_size", str(model_config["tts_model_size"]))
    writer.add_string("qwen3_tts.tokenizer_type", str(model_config["tokenizer_type"]))

    prompt_values = {
        "qwen3_tts.prompt.assistant_prefix": "<|im_start|>assistant\n",
        "qwen3_tts.prompt.assistant_suffix": "<|im_end|>\n<|im_start|>assistant\n",
        "qwen3_tts.prompt.reference_prefix": "<|im_start|>assistant\n",
        "qwen3_tts.prompt.reference_suffix": "<|im_end|>\n",
        "qwen3_tts.prompt.instruction_prefix": "<|im_start|>user\n",
        "qwen3_tts.prompt.instruction_suffix": "<|im_end|>\n",
    }
    for key, value in prompt_values.items():
        writer.add_string(key, value)

    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    pad_token = tokenizer_config.get("pad_token")
    eos_token = tokenizer_config.get("eos_token")
    writer.add_string("tokenizer.ggml.model", "qwen2")
    if not isinstance(pad_token, str) or pad_token not in token_to_id:
        raise ConversionError("tokenizer pad_token is missing from vocabulary")
    if not isinstance(eos_token, str) or eos_token not in token_to_id:
        raise ConversionError("tokenizer eos_token is missing from vocabulary")
    writer.add_int32("tokenizer.ggml.padding_token_id", token_to_id[pad_token])
    writer.add_int32("tokenizer.ggml.eos_token_id", token_to_id[eos_token])
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_array("tokenizer.ggml.added_token_names", added_names)
    writer.add_array("tokenizer.ggml.added_token_ids", added_ids)

    writer.add_uint32("qwen3_tts.text.vocab_size", int(talker["text_vocab_size"]))
    writer.add_int32(
        "qwen3_tts.text.im_start_token_id",
        int(model_config["im_start_token_id"]),
    )
    writer.add_int32(
        "qwen3_tts.text.im_end_token_id",
        int(model_config["im_end_token_id"]),
    )
    writer.add_int32(
        "qwen3_tts.text.tts_pad_token_id",
        int(model_config["tts_pad_token_id"]),
    )
    writer.add_int32(
        "qwen3_tts.text.tts_bos_token_id",
        int(model_config["tts_bos_token_id"]),
    )
    writer.add_int32(
        "qwen3_tts.text.tts_eos_token_id",
        int(model_config["tts_eos_token_id"]),
    )

    talker_values = (
        ("qwen3_tts.talker.codec_vocab_size", "vocab_size"),
        ("qwen3_tts.talker.text_vocab_size", "text_vocab_size"),
        ("qwen3_tts.talker.text_embedding_length", "text_hidden_size"),
        ("qwen3_tts.talker.embedding_length", "hidden_size"),
        ("qwen3_tts.talker.feed_forward_length", "intermediate_size"),
        ("qwen3_tts.talker.block_count", "num_hidden_layers"),
        ("qwen3_tts.talker.attention_head_count", "num_attention_heads"),
        ("qwen3_tts.talker.attention_head_count_kv", "num_key_value_heads"),
        ("qwen3_tts.talker.attention_key_length", "head_dim"),
        ("qwen3_tts.talker.context_length", "max_position_embeddings"),
        ("qwen3_tts.talker.position_id_per_second", "position_id_per_seconds"),
        ("qwen3_tts.talker.num_code_groups", "num_code_groups"),
    )
    for key, source_key in talker_values:
        writer.add_uint32(key, int(talker[source_key]))
    writer.add_uint32("qwen3_tts.talker.num_ctx", 2048)
    writer.add_string(
        "qwen3_tts.talker.hidden_activation",
        str(talker["hidden_act"]),
    )
    writer.add_float32(
        "qwen3_tts.talker.rms_norm_epsilon",
        float(talker["rms_norm_eps"]),
    )
    writer.add_bool(
        "qwen3_tts.talker.attention_bias",
        bool(talker.get("attention_bias", False)),
    )
    writer.add_bool(
        "qwen3_tts.talker.use_sliding_window",
        bool(talker.get("use_sliding_window", False)),
    )
    writer.add_uint32(
        "qwen3_tts.talker.sliding_window",
        int(talker.get("sliding_window") or 0),
    )
    writer.add_float32(
        "qwen3_tts.talker.rope_theta",
        float(talker["rope_theta"]),
    )
    talker_rope = talker.get("rope_scaling") or {}
    if not isinstance(talker_rope, dict):
        raise ConversionError("talker rope_scaling must be an object")
    writer.add_string(
        "qwen3_tts.talker.rope_type",
        str(talker_rope.get("rope_type", talker_rope.get("type", "default"))),
    )
    writer.add_array(
        "qwen3_tts.talker.rope_sections",
        [int(value) for value in talker_rope["mrope_section"]],
    )
    writer.add_bool(
        "qwen3_tts.talker.rope_interleaved",
        bool(talker_rope.get("interleaved", False)),
    )
    writer.add_bool("qwen3_tts.talker.rope_plain_neox", True)
    writer.add_int32("qwen3_tts.talker.codec_pad_token_id", int(talker["codec_pad_id"]))
    writer.add_int32("qwen3_tts.talker.codec_bos_token_id", int(talker["codec_bos_id"]))
    writer.add_int32(
        "qwen3_tts.talker.codec_eos_token_id",
        int(talker["codec_eos_token_id"]),
    )
    writer.add_int32("qwen3_tts.talker.codec_think_id", int(talker["codec_think_id"]))
    writer.add_int32(
        "qwen3_tts.talker.codec_nothink_id",
        int(talker["codec_nothink_id"]),
    )
    writer.add_int32(
        "qwen3_tts.talker.codec_think_bos_id",
        int(talker["codec_think_bos_id"]),
    )
    writer.add_int32(
        "qwen3_tts.talker.codec_think_eos_id",
        int(talker["codec_think_eos_id"]),
    )

    predictor_values = (
        ("qwen3_tts.code_predictor.vocab_size", "vocab_size"),
        ("qwen3_tts.code_predictor.hidden_size", "hidden_size"),
        ("qwen3_tts.code_predictor.intermediate_size", "intermediate_size"),
        ("qwen3_tts.code_predictor.block_count", "num_hidden_layers"),
        ("qwen3_tts.code_predictor.attention_head_count", "num_attention_heads"),
        ("qwen3_tts.code_predictor.attention_head_count_kv", "num_key_value_heads"),
        ("qwen3_tts.code_predictor.attention_key_length", "head_dim"),
        ("qwen3_tts.code_predictor.context_length", "max_position_embeddings"),
        ("qwen3_tts.code_predictor.num_code_groups", "num_code_groups"),
    )
    for key, source_key in predictor_values:
        writer.add_uint32(key, int(code_predictor[source_key]))
    writer.add_uint32(
        "qwen3_tts.code_predictor.input_embedding_size",
        talker_hidden,
    )
    writer.add_uint32(
        "qwen3_tts.code_predictor.talker_codec_vocab_size",
        int(talker["vocab_size"]),
    )
    writer.add_string(
        "qwen3_tts.code_predictor.hidden_activation",
        str(code_predictor["hidden_act"]),
    )
    writer.add_float32(
        "qwen3_tts.code_predictor.rms_norm_epsilon",
        float(code_predictor["rms_norm_eps"]),
    )
    writer.add_bool(
        "qwen3_tts.code_predictor.attention_bias",
        bool(code_predictor.get("attention_bias", False)),
    )
    writer.add_bool(
        "qwen3_tts.code_predictor.use_sliding_window",
        bool(code_predictor.get("use_sliding_window", False)),
    )
    writer.add_uint32(
        "qwen3_tts.code_predictor.sliding_window",
        int(code_predictor.get("sliding_window") or 0),
    )
    writer.add_array(
        "qwen3_tts.code_predictor.layer_types",
        [str(value) for value in code_predictor["layer_types"]],
    )
    writer.add_float32(
        "qwen3_tts.code_predictor.rope_theta",
        float(code_predictor["rope_theta"]),
    )
    predictor_rope = code_predictor.get("rope_scaling") or {}
    if not isinstance(predictor_rope, dict):
        raise ConversionError("code predictor rope_scaling must be an object")
    writer.add_string(
        "qwen3_tts.code_predictor.rope_type",
        str(
            predictor_rope.get(
                "rope_type",
                predictor_rope.get("type", "default"),
            )
        ),
    )
    writer.add_bool(
        "qwen3_tts.code_predictor.has_input_projection",
        has_input_projection,
    )
    writer.add_bool(
        "qwen3_tts.code_predictor.input_projection_bias",
        has_input_projection,
    )

    generation_values = (
        ("qwen3_tts.generation.max_new_tokens", "max_new_tokens", "uint32"),
        ("qwen3_tts.generation.do_sample", "do_sample", "bool"),
        ("qwen3_tts.generation.temperature", "temperature", "float32"),
        ("qwen3_tts.generation.top_k", "top_k", "uint32"),
        ("qwen3_tts.generation.top_p", "top_p", "float32"),
        (
            "qwen3_tts.generation.repetition_penalty",
            "repetition_penalty",
            "float32",
        ),
        (
            "qwen3_tts.generation.subtalker_do_sample",
            "subtalker_dosample",
            "bool",
        ),
        (
            "qwen3_tts.generation.subtalker_temperature",
            "subtalker_temperature",
            "float32",
        ),
        ("qwen3_tts.generation.subtalker_top_k", "subtalker_top_k", "uint32"),
        ("qwen3_tts.generation.subtalker_top_p", "subtalker_top_p", "float32"),
    )
    for key, source_key, value_type in generation_values:
        value = generation_config[source_key]
        if value_type == "uint32":
            writer.add_uint32(key, int(value))
        elif value_type == "float32":
            writer.add_float32(key, float(value))
        else:
            writer.add_bool(key, bool(value))
    writer.add_float32(
        "qwen3_tts.generation.subtalker_repetition_penalty",
        float(generation_config.get("subtalker_repetition_penalty", 1.0)),
    )

    speaker_names, speaker_ids, speaker_dialects = speaker_arrays(talker)
    language_names, language_ids = named_id_arrays(talker.get("codec_language_id", {}))
    writer.add_array("qwen3_tts.speaker.names", speaker_names)
    writer.add_array("qwen3_tts.speaker.ids", speaker_ids)
    writer.add_array("qwen3_tts.speaker.dialects", speaker_dialects)
    writer.add_array("qwen3_tts.language.names", language_names)
    writer.add_array("qwen3_tts.language.ids", language_ids)


def add_speech_metadata(
    writer: Any,
    model_name: str,
    model_config: dict[str, Any],
    speech_config: dict[str, Any],
    file_type: int,
) -> None:
    decoder = speech_config["decoder_config"]
    decoder_layers = int(decoder["num_hidden_layers"])

    writer.add_name(model_name)
    writer.add_file_type(file_type)
    writer.add_quantization_version(2)
    writer.add_uint32("qwen3_tts.schema_version", 3)
    writer.add_string("qwen3_tts.file_role", "speech_tokenizer")
    writer.add_string(
        "qwen3_tts.tokenizer_type",
        str(model_config["tokenizer_type"]),
    )
    writer.add_string(
        "qwen3_tts.speech.model_type",
        str(speech_config["model_type"]),
    )

    speech_values = (
        ("qwen3_tts.speech.input_sample_rate", "input_sample_rate"),
        ("qwen3_tts.speech.output_sample_rate", "output_sample_rate"),
        ("qwen3_tts.speech.encode_downsample_rate", "encode_downsample_rate"),
        ("qwen3_tts.speech.decode_upsample_rate", "decode_upsample_rate"),
        (
            "qwen3_tts.speech.encoder_valid_num_quantizers",
            "encoder_valid_num_quantizers",
        ),
    )
    for key, source_key in speech_values:
        writer.add_uint32(key, int(speech_config[source_key]))

    decoder_values = (
        ("qwen3_tts.speech.decoder.latent_dim", "latent_dim"),
        ("qwen3_tts.speech.decoder.codebook_dim", "codebook_dim"),
        ("qwen3_tts.speech.decoder.codebook_size", "codebook_size"),
        ("qwen3_tts.speech.decoder.semantic_codebook_size", "semantic_codebook_size"),
        ("qwen3_tts.speech.decoder.decoder_dim", "decoder_dim"),
        (
            "qwen3_tts.speech.decoder.vector_quantization_hidden_dimension",
            "vector_quantization_hidden_dimension",
        ),
        ("qwen3_tts.speech.decoder.hidden_size", "hidden_size"),
        ("qwen3_tts.speech.decoder.intermediate_size", "intermediate_size"),
        ("qwen3_tts.speech.decoder.block_count", "num_hidden_layers"),
        ("qwen3_tts.speech.decoder.attention_head_count", "num_attention_heads"),
        (
            "qwen3_tts.speech.decoder.attention_head_count_kv",
            "num_key_value_heads",
        ),
        ("qwen3_tts.speech.decoder.attention_key_length", "head_dim"),
        (
            "qwen3_tts.speech.decoder.context_length",
            "max_position_embeddings",
        ),
        ("qwen3_tts.speech.decoder.sliding_window", "sliding_window"),
        ("qwen3_tts.speech.decoder.num_quantizers", "num_quantizers"),
        (
            "qwen3_tts.speech.decoder.num_semantic_quantizers",
            "num_semantic_quantizers",
        ),
    )
    for key, source_key in decoder_values:
        writer.add_uint32(key, int(decoder[source_key]))
    writer.add_string(
        "qwen3_tts.speech.decoder.hidden_activation",
        str(decoder["hidden_act"]),
    )
    writer.add_float32(
        "qwen3_tts.speech.decoder.rms_norm_epsilon",
        float(decoder["rms_norm_eps"]),
    )
    writer.add_float32(
        "qwen3_tts.speech.decoder.layer_scale_initial_scale",
        float(decoder["layer_scale_initial_scale"]),
    )
    writer.add_bool(
        "qwen3_tts.speech.decoder.attention_bias",
        bool(decoder.get("attention_bias", False)),
    )
    writer.add_bool(
        "qwen3_tts.speech.decoder.use_sliding_window",
        bool(decoder.get("sliding_window")),
    )
    writer.add_array(
        "qwen3_tts.speech.decoder.layer_types",
        ["sliding_attention"] * decoder_layers,
    )
    writer.add_float32(
        "qwen3_tts.speech.decoder.rope_theta",
        float(decoder["rope_theta"]),
    )
    decoder_rope = decoder.get("rope_scaling") or {}
    if not isinstance(decoder_rope, dict):
        raise ConversionError("speech decoder rope_scaling must be an object")
    writer.add_string(
        "qwen3_tts.speech.decoder.rope_type",
        str(decoder_rope.get("rope_type", decoder_rope.get("type", "default"))),
    )
    writer.add_array(
        "qwen3_tts.speech.decoder.upsample_rates",
        [int(value) for value in decoder["upsample_rates"]],
    )
    writer.add_array(
        "qwen3_tts.speech.decoder.upsampling_ratios",
        [int(value) for value in decoder["upsampling_ratios"]],
    )


def add_safetensors(
    writer: Any,
    torch: Any,
    safe_open: Any,
    path: Path,
    include_encoder: bool,
    seen_names: set[str],
) -> tuple[int, set[str]]:
    count = 0
    dtypes: set[str] = set()
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        for source_name in tensors.keys():
            if not include_encoder and source_name.startswith("encoder."):
                continue
            destination_name = map_tensor_name(source_name)
            if len(destination_name) >= 64:
                raise ConversionError(
                    f"GGUF tensor name exceeds 63 characters: {destination_name}"
                )
            if destination_name in seen_names:
                raise ConversionError(f"duplicate GGUF tensor name: {destination_name}")
            tensor = tensors.get_tensor(source_name).detach().cpu().contiguous()
            from gguf import GGMLQuantizationType

            dtypes.add(normalize_dtype_name(tensor.dtype))
            if tensor.dtype == torch.bfloat16:
                array = tensor.view(torch.int16).numpy()
                raw_dtype = GGMLQuantizationType.BF16
            elif tensor.dtype in (torch.float16, torch.float32):
                array = tensor.numpy()
                raw_dtype = None
            else:
                raise ConversionError(
                    f"unsupported safetensors dtype {tensor.dtype} for {source_name}"
                )
            writer.add_tensor(
                destination_name,
                array,
                raw_dtype=raw_dtype,
            )
            seen_names.add(destination_name)
            count += 1
    return count, dtypes


def scan_tensor_dtypes(
    safe_open: Any,
    path: Path,
    include_encoder: bool,
) -> set[str]:
    values: set[str] = set()
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if not include_encoder and name.startswith("encoder."):
                continue
            values.add(normalize_dtype_name(tensors.get_slice(name).get_dtype()))
    return values


def normalize_dtype_name(dtype: Any) -> str:
    value = str(dtype).lower().removeprefix("torch.")
    aliases = {
        "bfloat16": "bf16",
        "float16": "f16",
        "half": "f16",
        "float32": "f32",
        "float": "f32",
    }
    return aliases.get(value, value)


def dtype_suffix(dtypes: set[str]) -> str:
    order = {"bf16": 0, "f16": 1, "f32": 2}
    unknown = dtypes.difference(order)
    if unknown:
        raise ConversionError(
            "unsupported tensor dtypes: " + ", ".join(sorted(unknown))
        )
    return "-".join(sorted(dtypes, key=lambda value: order[value]))


def file_type_for_dtypes(dtypes: set[str]) -> int:
    if dtypes == {"f32"}:
        return 0
    if dtypes == {"f16"}:
        return 1
    if dtypes == {"bf16"}:
        return 24
    return 0


def model_size_name(value: str) -> str:
    return {
        "0b6": "0.6b",
        "1b7": "1.7b",
    }.get(value, value)


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


def convert(args: argparse.Namespace) -> None:
    np, torch, safe_open, gguf_writer = load_dependencies()
    del np  # Imported to make the gguf-py numpy dependency failure explicit.

    model_dir = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not model_dir.is_dir():
        raise ConversionError(f"input directory does not exist: {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "model.safetensors"
    speech_file = model_dir / "speech_tokenizer" / "model.safetensors"
    if not model_file.is_file() or not speech_file.is_file():
        raise ConversionError("model.safetensors or speech_tokenizer/model.safetensors is missing")

    model_config = load_json(model_dir / "config.json")
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")
    speech_config = load_json(model_dir / "speech_tokenizer" / "config.json")
    generation_config = load_json(model_dir / "generation_config.json")
    tokens, merges, added_names, added_ids = build_tokenizer(model_dir, tokenizer_config)

    include_encoder = not args.decoder_only
    talker_dtypes = scan_tensor_dtypes(safe_open, model_file, include_encoder=True)
    speech_dtypes = scan_tensor_dtypes(
        safe_open,
        speech_file,
        include_encoder=include_encoder,
    )
    talker_suffix = dtype_suffix(talker_dtypes)
    speech_suffix = dtype_suffix(speech_dtypes)
    size = model_size_name(str(model_config["tts_model_size"]))
    mode = str(model_config["tts_model_type"]).replace("_", "")
    talker_output = output_dir / (
        f"qwen3-tts-talker-{size}-{mode}-{talker_suffix}.gguf"
    )
    speech_output = output_dir / (
        f"qwen3-tts-speech-tokenizer-12hz-{speech_suffix}.gguf"
    )
    if not args.overwrite:
        for output in (talker_output, speech_output):
            if output.exists():
                raise ConversionError(f"output already exists: {output}")

    talker_temporary = talker_output.with_name(talker_output.name + ".tmp")
    speech_temporary = speech_output.with_name(speech_output.name + ".tmp")
    talker_temporary.unlink(missing_ok=True)
    speech_temporary.unlink(missing_ok=True)

    talker_writer = gguf_writer(str(talker_temporary), "qwen3-tts")
    add_talker_metadata(
        talker_writer,
        talker_output.stem,
        model_config,
        tokenizer_config,
        generation_config,
        tokens,
        merges,
        added_names,
        added_ids,
        file_type_for_dtypes(talker_dtypes),
    )
    talker_count, written_talker_dtypes = add_safetensors(
        talker_writer,
        torch,
        safe_open,
        model_file,
        include_encoder=True,
        seen_names=set(),
    )
    write_gguf(talker_writer, talker_temporary, talker_output)

    speech_writer = gguf_writer(
        str(speech_temporary),
        "qwen3-tts-tokenizer",
    )
    add_speech_metadata(
        speech_writer,
        speech_output.stem,
        model_config,
        speech_config,
        file_type_for_dtypes(speech_dtypes),
    )
    speech_count, written_speech_dtypes = add_safetensors(
        speech_writer,
        torch,
        safe_open,
        speech_file,
        include_encoder=include_encoder,
        seen_names=set(),
    )
    write_gguf(speech_writer, speech_temporary, speech_output)

    if written_talker_dtypes != talker_dtypes:
        raise ConversionError("talker tensor dtypes changed during conversion")
    if written_speech_dtypes != speech_dtypes:
        raise ConversionError("speech tensor dtypes changed during conversion")

    print(f"Wrote {talker_output}")
    print(f"Wrote {speech_output}")
    print(f"  Talker tensors: {talker_count}")
    print(f"  Talker dtypes: {', '.join(sorted(written_talker_dtypes))}")
    print(f"  Speech tensors: {speech_count}")
    print(f"  Speech dtypes: {', '.join(sorted(written_speech_dtypes))}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Merges: {len(merges)}")
    print(f"  Added tokens: {len(added_names)}")
    print(f"  Include encoder: {include_encoder}")


def main() -> int:
    args = parse_args()
    try:
        convert(args)
    except ConversionError as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Conversion failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
