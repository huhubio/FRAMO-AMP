#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT.fasta [OUTPUT_DIR]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTA="$(realpath "$1")"
OUT_DIR="$(realpath -m "${2:-framo_amp_output}")"
MODEL_ROOT="${FRAMO_BACKBONE_DIR:-${REPO_ROOT}/models/backbones}"
GPU_IDS="${FRAMO_GPUS:-0}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-foldseek}"
PROSTT5_DIR="${MODEL_ROOT}/ProstT5_foldseek"
STEM="$(basename "$FASTA")"
STEM="${STEM%.*}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$OUT_DIR/features/esm2" "$OUT_DIR/features/protbert" \
  "$OUT_DIR/features/saprot_prostt5" "$OUT_DIR/structure_tokens" "$OUT_DIR/foldseek_tmp"

if [[ ! -f "${MODEL_ROOT}/esm2_t33_650M_UR50D/pytorch_model.bin" ||
      ! -f "${MODEL_ROOT}/prot_bert/pytorch_model.bin" ||
      ! -f "${MODEL_ROOT}/SaProt_650M_AF2/pytorch_model.bin" ||
      ! -f "${PROSTT5_DIR}/prostt5-f16.gguf" ]]; then
  python "$REPO_ROOT/scripts/download_backbones.py" --output-dir "$MODEL_ROOT" --foldseek-bin "$FOLDSEEK_BIN"
fi

python "$REPO_ROOT/scripts/extract_esm2_embeddings.py" "$FASTA" \
  --model-dir "${MODEL_ROOT}/esm2_t33_650M_UR50D" \
  --model-name esm2_t33_650M_UR50D --output-dir "$OUT_DIR/features/esm2" \
  --gpus "$GPU_IDS" --batch-size "${ESM2_BATCH_SIZE:-32}" --pooling mean \
  --model-dtype "${MODEL_DTYPE:-float16}" --output-dtype float32

python "$REPO_ROOT/scripts/extract_protein_lm_embeddings.py" "$FASTA" \
  --model-dir "${MODEL_ROOT}/prot_bert" --model-name prot_bert --model-type protbert \
  --output-dir "$OUT_DIR/features/protbert" --gpus "$GPU_IDS" \
  --batch-size "${PROTBERT_BATCH_SIZE:-16}" --pooling mean \
  --model-dtype "${MODEL_DTYPE:-float16}" --output-dtype float32

python "$REPO_ROOT/scripts/build_saprot_structure_tokens_with_prostt5.py" \
  --fasta "$FASTA" --prostt5_model "$PROSTT5_DIR" \
  --out_tsv "$OUT_DIR/structure_tokens/${STEM}.prostt5_3di.tsv" \
  --audit_tsv "$OUT_DIR/structure_tokens/${STEM}.prostt5_3di.audit.tsv" \
  --tmp_dir "$OUT_DIR/foldseek_tmp" --foldseek_bin "$FOLDSEEK_BIN" \
  --threads "${FOLDSEEK_THREADS:-16}" ${PROSTT5_GPU:+--gpu}

python "$REPO_ROOT/scripts/extract_protein_lm_embeddings.py" "$FASTA" \
  --model-dir "${MODEL_ROOT}/SaProt_650M_AF2" --model-name SaProt_650M_AF2 --model-type saprot \
  --structure-tsv "$OUT_DIR/structure_tokens/${STEM}.prostt5_3di.tsv" \
  --output-dir "$OUT_DIR/features/saprot_prostt5" --gpus "$GPU_IDS" \
  --batch-size "${SAPROT_BATCH_SIZE:-8}" --pooling mean \
  --model-dtype "${MODEL_DTYPE:-float16}" --output-dtype float32

python - "$REPO_ROOT" "$OUT_DIR" "$STEM" <<'PY'
import json
import sys
from pathlib import Path

repo, out, stem = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
members = [
    ("esm2_mlp", "esm2", "esm2_t33_650M_UR50D", "esm2"),
    ("saprot_prostt5_mlp", "saprot_prostt5", "SaProt_650M_AF2", "saprot_prostt5"),
    ("protbert_mlp", "protbert", "prot_bert", "protbert"),
]
manifest = {"members": []}
for name, feature_dir, feature_model, model_prefix in members:
    base = out / "features" / feature_dir / f"{stem}.{feature_model}.mean"
    manifest["members"].append({
        "name": name,
        "features": str(Path(str(base) + ".npy")),
        "metadata": str(Path(str(base) + ".metadata.tsv")),
        "checkpoint": str(repo / "models" / f"{model_prefix}_mlp.pt"),
        "config": str(repo / "models" / f"{model_prefix}_config.json"),
        "scaler": str(repo / "models" / f"{model_prefix}_scaler.npz"),
    })
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

python -m amp_ensemble.cli --manifest "$OUT_DIR/manifest.json" \
  --device "${ENSEMBLE_DEVICE:-auto}" --output "$OUT_DIR/predictions.tsv"

echo "FRAMO-AMP predictions: $OUT_DIR/predictions.tsv"
