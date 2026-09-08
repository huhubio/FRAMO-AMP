import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from amp_ensemble.inference import EnsemblePredictor, MemberSpec, write_predictions
from amp_ensemble.model import build_mlp


def make_member(directory: Path, name: str, bias: float) -> MemberSpec:
    features = np.zeros((3, 2), dtype=np.float32)
    feature_path = directory / f"{name}.npy"
    np.save(feature_path, features)

    metadata_path = directory / f"{name}.tsv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_row", "seq_id", "sequence"], delimiter="\t")
        writer.writeheader()
        for index in range(3):
            writer.writerow({"feature_row": index, "seq_id": f"seq_{index}", "sequence": "ACD"})

    model = build_mlp(2, [], 0.0)
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(bias)
    checkpoint_path = directory / f"{name}.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    config_path = directory / f"{name}.json"
    config_path.write_text(
        json.dumps({"best_params": {"hidden": [], "dropout": 0.0}, "final_cv_threshold": 0.5}),
        encoding="utf-8",
    )
    scaler_path = directory / f"{name}.npz"
    np.savez(scaler_path, mean=np.zeros(2), scale=np.ones(2))
    return MemberSpec(name, feature_path, metadata_path, checkpoint_path, config_path, scaler_path)


class InferenceTests(unittest.TestCase):
    def test_majority_vote_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            members = [
                make_member(directory, "positive_a", 2.0),
                make_member(directory, "positive_b", 1.0),
                make_member(directory, "negative", -2.0),
            ]
            predictor = EnsemblePredictor(members, device="cpu", batch_size=2)
            metadata, outputs, score, final = predictor.predict("majority")
            self.assertEqual(final.tolist(), [1, 1, 1])
            self.assertTrue(np.allclose(score, 2 / 3))

            output_path = directory / "predictions.tsv"
            write_predictions(output_path, metadata, members, outputs, score, final)
            with output_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["ensemble_label"], "AMP")

    def test_rejects_misaligned_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            members = [make_member(directory, "a", 1.0), make_member(directory, "b", -1.0)]
            text = members[1].metadata_path.read_text(encoding="utf-8").replace("seq_1", "different")
            members[1].metadata_path.write_text(text, encoding="utf-8")
            predictor = EnsemblePredictor(members, device="cpu")
            with self.assertRaisesRegex(ValueError, "sequence IDs/order differ"):
                predictor.predict()


if __name__ == "__main__":
    unittest.main()
