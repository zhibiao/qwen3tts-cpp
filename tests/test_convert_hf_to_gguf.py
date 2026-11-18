from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "convert_hf_to_gguf.py"
spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", MODULE_PATH)
converter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(converter)


class ConvertHfToGgufTests(unittest.TestCase):
    def test_tensor_name_mapping(self) -> None:
        self.assertEqual(
            converter.map_tensor_name("talker.model.layers.0.self_attn.q_proj.weight"),
            "talker.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            converter.map_tensor_name(
                "talker.code_predictor.model.layers.0.mlp.up_proj.weight"
            ),
            "talker.code_predictor.layers.0.mlp.up_proj.weight",
        )
        self.assertEqual(
            converter.map_tensor_name("decoder.layers.0.self_attn.q_proj.weight"),
            "sdec.l.0.sa.q_proj.weight",
        )
        self.assertEqual(
            converter.map_tensor_name(
                "encoder.encoder_transformer.layers.0.self_attn.q_proj.weight"
            ),
            "senc.et.l.0.self_attn.q_proj.weight",
        )
        with self.assertRaises(converter.ConversionError):
            converter.map_tensor_name("unknown.weight")

    def test_named_id_and_speaker_metadata(self) -> None:
        names, ids = converter.named_id_arrays({"serena": 2, "aiden": 1})
        self.assertEqual(names, ["aiden", "serena"])
        self.assertEqual(ids, [1, 2])
        names, ids, dialects = converter.speaker_arrays(
            {
                "spk_id": {"serena": 2, "dylan": 1},
                "spk_is_dialect": {"dylan": "beijing_dialect"},
            }
        )
        self.assertEqual(names, ["dylan", "serena"])
        self.assertEqual(ids, [1, 2])
        self.assertEqual(dialects, ["beijing_dialect", ""])

    def test_dtype_and_size_helpers(self) -> None:
        self.assertEqual(converter.dtype_suffix({"bf16", "f32"}), "bf16-f32")
        self.assertEqual(converter.file_type_for_dtypes({"f16"}), 1)
        self.assertEqual(converter.file_type_for_dtypes({"bf16"}), 24)
        self.assertEqual(converter.model_size_name("0b6"), "0.6b")
        self.assertEqual(converter.model_size_name("1b7"), "1.7b")
        self.assertEqual(converter.normalize_dtype_name("torch.bfloat16"), "bf16")

    def test_build_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "vocab.json").write_text(
                json.dumps({"a": 0, "b": 1}),
                encoding="utf-8",
            )
            (model_dir / "merges.txt").write_text(
                "a b\nb c\n",
                encoding="utf-8",
            )
            (
                tokens,
                merges,
                added_names,
                added_ids,
                nfc_normalization,
            ) = converter.build_tokenizer(
                model_dir,
                {
                    "added_tokens_decoder": {
                        "2": {"content": "<|special|>"}
                    }
                },
            )
            self.assertEqual(tokens, ["a", "b", "<|special|>"])
            self.assertEqual(merges, ["a b", "b c"])
            self.assertEqual(added_names, ["<|special|>"])
            self.assertEqual(added_ids, [2])
            self.assertFalse(nfc_normalization)

            (model_dir / "tokenizer.json").write_text(
                json.dumps({"normalizer": {"type": "NFC"}}),
                encoding="utf-8",
            )
            (
                tokens,
                merges,
                added_names,
                added_ids,
                nfc_normalization,
            ) = converter.build_tokenizer(model_dir, {})
            self.assertTrue(nfc_normalization)

            (model_dir / "tokenizer.json").write_text(
                json.dumps({"normalizer": {"type": "NFKC"}}),
                encoding="utf-8",
            )
            with self.assertRaises(converter.ConversionError):
                converter.build_tokenizer(model_dir, {})


if __name__ == "__main__":
    unittest.main()
