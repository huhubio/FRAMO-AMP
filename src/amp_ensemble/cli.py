"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import EnsemblePredictor, MemberSpec, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ESM2/SaProt-ProstT5/ProtBERT MLP ensemble")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON file describing the three members")
    parser.add_argument("--output", type=Path, default=Path("predictions.tsv"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--strategy", choices=("majority", "soft"), default="majority")
    parser.add_argument("--soft-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_manifest(path: Path) -> list[MemberSpec]:
    content = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    return [
        MemberSpec(
            name=item["name"],
            feature_path=resolve(item["features"]),
            metadata_path=resolve(item["metadata"]),
            checkpoint_path=resolve(item["checkpoint"]),
            config_path=resolve(item["config"]),
            scaler_path=resolve(item["scaler"]),
        )
        for item in content["members"]
    ]


def main() -> None:
    args = parse_args()
    members = load_manifest(args.manifest)
    predictor = EnsemblePredictor(members, device=args.device, batch_size=args.batch_size)
    metadata, outputs, ensemble_score, final = predictor.predict(args.strategy, args.soft_threshold)
    write_predictions(args.output, metadata, members, outputs, ensemble_score, final)
    print(f"device={predictor.device} samples={len(final)} AMP={int(final.sum())} non-AMP={int((1-final).sum())}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
