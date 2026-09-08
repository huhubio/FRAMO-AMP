"""Validated inference and voting for the three MLP members."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .model import build_mlp


@dataclass(frozen=True)
class MemberSpec:
    name: str
    feature_path: Path
    metadata_path: Path
    checkpoint_path: Path
    config_path: Path
    scaler_path: Path


@dataclass(frozen=True)
class MemberOutput:
    score: np.ndarray
    prediction: np.ndarray
    threshold: float


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    required = {"feature_row", "seq_id"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"metadata {path} is missing columns: {sorted(missing)}")
    for index, row in enumerate(rows):
        if int(row["feature_row"]) != index:
            raise ValueError(f"feature_row must be contiguous from zero in {path}")
    return rows


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError("checkpoint must be a state_dict or contain a 'state_dict' key")


class EnsemblePredictor:
    """Run aligned embeddings through three MLPs and combine their predictions."""

    def __init__(self, members: list[MemberSpec], device: str = "auto", batch_size: int = 4096):
        if len(members) < 2:
            raise ValueError("an ensemble requires at least two members")
        if len({member.name for member in members}) != len(members):
            raise ValueError("member names must be unique")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.members = members
        self.device = torch.device(device)
        self.batch_size = batch_size

    def _predict_member(self, spec: MemberSpec) -> tuple[list[dict[str, str]], MemberOutput]:
        config = json.loads(spec.config_path.read_text(encoding="utf-8"))
        features = np.load(spec.feature_path, mmap_mode="r")
        metadata = read_metadata(spec.metadata_path)
        if features.ndim != 2:
            raise ValueError(f"features must be a 2D matrix: {spec.feature_path}")
        if features.shape[0] != len(metadata):
            raise ValueError(f"feature/metadata row mismatch for {spec.name}")

        scaler = np.load(spec.scaler_path)
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
            raise ValueError(f"scaler dimension mismatch for {spec.name}")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"invalid scale values for {spec.name}")

        params = config["best_params"]
        model = build_mlp(features.shape[1], [int(x) for x in params["hidden"]], float(params["dropout"]))
        payload = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(_state_dict(payload), strict=True)
        model.to(self.device).eval()

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, features.shape[0], self.batch_size):
                raw = np.asarray(features[start : start + self.batch_size], dtype=np.float32)
                standardized = (raw - mean) / scale
                logits = model(torch.from_numpy(standardized).to(self.device)).squeeze(1)
                chunks.append(torch.sigmoid(logits).cpu().numpy())
        scores = np.concatenate(chunks).astype(np.float64, copy=False)
        threshold = float(config["final_cv_threshold"])
        predictions = (scores >= threshold).astype(np.int8)
        return metadata, MemberOutput(scores, predictions, threshold)

    def predict(
        self, strategy: Literal["majority", "soft"] = "majority", soft_threshold: float = 0.5
    ) -> tuple[list[dict[str, str]], dict[str, MemberOutput], np.ndarray, np.ndarray]:
        if strategy not in {"majority", "soft"}:
            raise ValueError("strategy must be 'majority' or 'soft'")
        if not 0.0 <= soft_threshold <= 1.0:
            raise ValueError("soft_threshold must be in [0, 1]")

        reference: list[dict[str, str]] | None = None
        reference_ids: list[str] | None = None
        outputs: dict[str, MemberOutput] = {}
        for spec in self.members:
            metadata, output = self._predict_member(spec)
            ids = [row["seq_id"] for row in metadata]
            if reference is None:
                reference, reference_ids = metadata, ids
            elif ids != reference_ids:
                raise ValueError(f"sequence IDs/order differ for member {spec.name}")
            outputs[spec.name] = output

        predictions = np.vstack([outputs[m.name].prediction for m in self.members])
        scores = np.vstack([outputs[m.name].score for m in self.members])
        vote_fraction = predictions.mean(axis=0)
        mean_score = scores.mean(axis=0)
        if strategy == "majority":
            final = (predictions.sum(axis=0) >= (len(self.members) // 2 + 1)).astype(np.int8)
            ensemble_score = vote_fraction
        else:
            final = (mean_score >= soft_threshold).astype(np.int8)
            ensemble_score = mean_score
        assert reference is not None
        return reference, outputs, ensemble_score, final


def write_predictions(
    output_path: Path,
    metadata: list[dict[str, str]],
    members: list[MemberSpec],
    outputs: dict[str, MemberOutput],
    ensemble_score: np.ndarray,
    final: np.ndarray,
) -> None:
    fields = ["feature_row", "seq_id", "sequence"]
    for member in members:
        fields.extend((f"{member.name}_score", f"{member.name}_threshold", f"{member.name}_prediction"))
    fields.extend(("ensemble_score", "ensemble_prediction", "ensemble_label"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(metadata):
            item: dict[str, Any] = {
                "feature_row": index,
                "seq_id": row["seq_id"],
                "sequence": row.get("sequence", ""),
            }
            for member in members:
                member_output = outputs[member.name]
                item[f"{member.name}_score"] = f"{member_output.score[index]:.8f}"
                item[f"{member.name}_threshold"] = f"{member_output.threshold:.8f}"
                item[f"{member.name}_prediction"] = int(member_output.prediction[index])
            item["ensemble_score"] = f"{ensemble_score[index]:.8f}"
            item["ensemble_prediction"] = int(final[index])
            item["ensemble_label"] = "AMP" if final[index] else "non-AMP"
            writer.writerow(item)
