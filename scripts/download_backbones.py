#!/usr/bin/env python3
"""Download the exact frozen backbone snapshots and Foldseek ProstT5 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "backbones.json"
HF_FILES = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_size: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")
    print(f"verified {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "models" / "backbones")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--skip-prostt5", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("esm2", "protbert", "saprot"):
        item = config[name]
        target = args.output_dir / item["directory"]
        print(f"downloading {item['repo_id']}@{item['revision']} -> {target}")
        snapshot_download(
            repo_id=item["repo_id"],
            revision=item["revision"],
            local_dir=target,
            allow_patterns=HF_FILES,
        )
        if not args.no_verify:
            verify(target / item["weight_file"], item["weight_size_bytes"], item["weight_sha256"])

    if args.skip_prostt5:
        return
    foldseek = shutil.which(args.foldseek_bin)
    if foldseek is None:
        raise FileNotFoundError("Foldseek is required for ProstT5; install Foldseek or use --skip-prostt5")
    item = config["prostt5_foldseek"]
    target = args.output_dir / item["directory"]
    weight = target / item["weight_file"]
    if not weight.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output_dir / "foldseek_download_tmp"
        subprocess.run([foldseek, "databases", "ProstT5", str(target), str(temp)], check=True)
    if not args.no_verify:
        verify(weight, item["weight_size_bytes"], item["weight_sha256"])


if __name__ == "__main__":
    main()
