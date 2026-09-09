import tempfile
import unittest
from pathlib import Path

import torch

from framo_amp.generation.progen import GenerationSettings, clean_generated_sequence, generate_batch


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def decode(self, _row, skip_special_tokens=True):  # noqa: ANN001
        return "ACD-EF ghi"


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        return torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 2]])


class GenerationTests(unittest.TestCase):
    def test_generation_passes_attention_mask(self) -> None:
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp:
            settings = GenerationSettings(
                model_dir=temp,
                out_fasta=Path(temp) / "out.fa",
                num_return_sequences=1,
                min_seq_length=8,
                max_seq_length=8,
                batch_size=1,
            )
            result = generate_batch(model, FakeTokenizer(), settings, [42])
        self.assertEqual(result, ["ACDEFGHI"])
        self.assertTrue(torch.equal(model.kwargs["attention_mask"], torch.ones((1, 1), dtype=torch.long)))

    def test_clean_generated_sequence(self) -> None:
        self.assertEqual(clean_generated_sequence("acD-e f*"), "ACDEF")


if __name__ == "__main__":
    unittest.main()
