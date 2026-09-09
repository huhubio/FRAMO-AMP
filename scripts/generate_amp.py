"""Generate AMP candidate sequences with the bundled ProGen-based generator.

Example::

    python scripts/generate_amp.py \
        --num-return-sequences 100 \
        --out results/generated_amp.fasta

    # Generate with a LoRA adapter produced by finetune_generation_lora.py.
    python scripts/generate_amp.py \
        --adapter-dir lora_runs/my_dataset/final \
        --num-return-sequences 100 \
        --out results/generated_amp_lora.fasta
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from framo_amp.utils import load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AMP candidates with the bundled generator.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Generator model directory (default: config generation.model_dir).")
    parser.add_argument("--adapter-dir", type=Path, default=None,
                        help="Optional LoRA adapter directory.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output FASTA path.")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Optional JSON output path (default: same stem as --out).")
    parser.add_argument("--num-return-sequences", type=int, default=None)
    parser.add_argument("--min-seq-length", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-attempts-factor", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help='"cuda", "cpu", or "cuda:0". Under torchrun defaults to each local rank.')
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow HuggingFace downloads instead of local-files-only loading.")
    parser.add_argument("--no-merge-adapter", action="store_true",
                        help="Keep PEFT adapter unmerged during generation.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _cfg_value(cli_value, cfg: dict, key: str, default):  # noqa: ANN001, ANN202
    return cli_value if cli_value is not None else cfg.get(key, default)


def main() -> None:
    args = _parse_args()

    from framo_amp.generation import GenerationSettings, run_generation  # noqa: WPS433
    from framo_amp.generation.distributed import get_rank, init_distributed_if_needed  # noqa: WPS433

    init_distributed_if_needed()
    rank = get_rank()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank={rank}] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config, default_path=REPO_ROOT / "configs" / "default.yaml")
    gen_cfg = cfg.get("generation", {})

    model_dir = args.model_dir or Path(gen_cfg.get("model_dir", REPO_ROOT / "AMP_generation"))
    if not Path(model_dir).is_absolute():
        model_dir = REPO_ROOT / model_dir

    settings = GenerationSettings(
        model_dir=model_dir,
        adapter_dir=args.adapter_dir,
        out_fasta=args.out,
        out_json=args.out_json,
        num_return_sequences=_cfg_value(args.num_return_sequences, gen_cfg, "num_return_sequences", 1000),
        min_seq_length=_cfg_value(args.min_seq_length, gen_cfg, "min_seq_length", 8),
        max_seq_length=_cfg_value(args.max_seq_length, gen_cfg, "max_seq_length", 25),
        batch_size=_cfg_value(args.batch_size, gen_cfg, "batch_size", 32),
        seed=_cfg_value(args.seed, gen_cfg, "seed", 42),
        temperature_min=_cfg_value(args.temperature_min, gen_cfg, "temperature_min", 0.8),
        temperature_max=_cfg_value(args.temperature_max, gen_cfg, "temperature_max", 1.0),
        top_k=_cfg_value(args.top_k, gen_cfg, "top_k", 30),
        top_p=_cfg_value(args.top_p, gen_cfg, "top_p", 0.90),
        repetition_penalty=_cfg_value(args.repetition_penalty, gen_cfg, "repetition_penalty", 1.1),
        max_attempts_factor=_cfg_value(args.max_attempts_factor, gen_cfg, "max_attempts_factor", 5),
        device=args.device or gen_cfg.get("device"),
        local_files_only=not args.allow_download and bool(gen_cfg.get("local_files_only", True)),
        merge_adapter=not args.no_merge_adapter,
    )
    run_generation(settings)


if __name__ == "__main__":
    main()
