from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, quants


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "quantize_gguf.py"


class QuantizeGgufTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "input.gguf"
        writer = GGUFWriter(str(self.input), "qwen3-tts")
        writer.add_name("quantize-test")
        writer.add_file_type(0)
        writer.add_quantization_version(2)
        writer.add_string("qwen3_tts.file_role", "talker")
        self.matrix = np.arange(64, dtype=np.float32).reshape(2, 32) / 64.0
        writer.add_tensor("talker.matrix.weight", self.matrix)
        writer.add_tensor("talker.norm.weight", np.ones(32, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def convert(self, preset: str, *, check: bool = True) -> subprocess.CompletedProcess:
        output = self.directory / f"output-{preset}.gguf"
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(self.input),
                "--output",
                str(output),
                "--type",
                preset,
                "--overwrite",
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    def assert_preset(
        self,
        preset: str,
        file_type: int,
        tensor_type: GGMLQuantizationType,
        tolerance: float,
    ) -> None:
        self.convert(preset)
        output = self.directory / f"output-{preset}.gguf"
        reader = GGUFReader(output)
        self.assertEqual(int(reader.fields["general.file_type"].contents()), file_type)
        self.assertEqual(
            reader.fields["qwen3_tts.quantization.preset"].contents(),
            preset,
        )
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        matrix = tensors["talker.matrix.weight"]
        self.assertEqual(matrix.tensor_type, tensor_type)
        self.assertEqual(
            tensors["talker.norm.weight"].tensor_type,
            GGMLQuantizationType.F32,
        )
        actual = quants.dequantize(matrix.data, matrix.tensor_type).reshape(-1)
        np.testing.assert_allclose(
            actual,
            self.matrix.reshape(-1),
            atol=tolerance,
            rtol=0,
        )

    def test_q8_0(self) -> None:
        self.assert_preset("q8_0", 7, GGMLQuantizationType.Q8_0, 2.0e-2)

    def test_bf16(self) -> None:
        self.assert_preset("bf16", 32, GGMLQuantizationType.BF16, 1.0e-3)

    def test_f16_rejected(self) -> None:
        result = self.convert("f16", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
