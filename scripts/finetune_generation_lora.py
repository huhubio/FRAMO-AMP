"""LoRA fine-tune the bundled AMP generator on a user FASTA dataset.

Example::

    torchrun --nproc_per_node=4 scripts/finetune_generation_lora.py \
        --fasta data/my_amp_set.fasta \
        --dataset-name my_amp_set \
        --save-dir lora_runs/my_amp_set \
        --num-train-epochs 5 \
        --per-device-train-batch-size 8
"""
from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from framo_amp.utils import load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the AMP generator.")
    parser.add_argument("--base-model-dir", type=Path, default=None,
                        help="Base generator directory (default: config generation.model_dir).")
    parser.add_argument("--fasta", required=True, type=Path,
                        help="User FASTA/FAA training file.")
    parser.add_argument("--dataset-name", type=str, default="peptide")
    parser.add_argument("--save-dir", required=True, type=Path,
                        help="Directory for LoRA adapters and logs.")
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--no-mask-pad-labels", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", type=str, default=None)
    parser.add_argument("--lora-bias", choices=["none", "lora_only", "all"], default=None)
    parser.add_argument("--lora-modules-to-save", type=str, default=None)
    parser.add_argument("--merge-and-save", action="store_true")

    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=None)

    parser.add_argument("--generate-after-train", action="store_true",
                        help="Generate candidates from the final adapter after training.")
    parser.add_argument("--generation-out", type=Path, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=None)
    parser.add_argument("--min-seq-length", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    return parser.parse_args()


def _cfg_value(cli_value, cfg: dict, key: str, default):  # noqa: ANN001, ANN202
    return cli_value if cli_value is not None else cfg.get(key, default)


def _build_training_args(args: argparse.Namespace, cfg: dict, save_dir: Path):
    from transformers import TrainingArguments  # noqa: WPS433

    eval_strategy = "no" if args.no_eval else "epoch"
    kwargs = {
        "output_dir": str(save_dir),
        "do_train": True,
        "do_eval": not args.no_eval,
        "num_train_epochs": _cfg_value(args.num_train_epochs, cfg, "num_train_epochs", 5),
        "per_device_train_batch_size": _cfg_value(
            args.per_device_train_batch_size, cfg, "per_device_train_batch_size", 8
        ),
        "per_device_eval_batch_size": _cfg_value(
            args.per_device_eval_batch_size, cfg, "per_device_eval_batch_size", 8
        ),
        "gradient_accumulation_steps": _cfg_value(
            args.gradient_accumulation_steps, cfg, "gradient_accumulation_steps", 1
        ),
        "learning_rate": _cfg_value(args.learning_rate, cfg, "learning_rate", 3e-4),
        "weight_decay": _cfg_value(args.weight_decay, cfg, "weight_decay", 0.0),
        "warmup_ratio": _cfg_value(args.warmup_ratio, cfg, "warmup_ratio", 0.03),
        "logging_steps": _cfg_value(args.logging_steps, cfg, "logging_steps", 10),
        "save_total_limit": _cfg_value(args.save_total_limit, cfg, "save_total_limit", None),
        "save_strategy": "epoch",
        "save_safetensors": True,
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
    }
    if args.early_stopping_patience is not None and not args.no_eval:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    if "ddp_find_unused_parameters" in sig.parameters:
        kwargs["ddp_find_unused_parameters"] = True
    return TrainingArguments(**kwargs)


def main() -> None:
    args = _parse_args()

    from transformers import EarlyStoppingCallback, set_seed  # noqa: WPS433
    from framo_amp.generation import GenerationSettings, run_generation  # noqa: WPS433
    from framo_amp.generation.distributed import get_rank, init_distributed_if_needed  # noqa: WPS433
    from framo_amp.generation.lora import LoraFineTuneSettings, train_lora  # noqa: WPS433

    init_distributed_if_needed()
    rank = get_rank()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.save_dir / "train.log"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][rank={rank}] %(levelname)s: %(message)s",
        handlers=handlers,
    )

    cfg = load_config(args.config, default_path=REPO_ROOT / "configs" / "default.yaml")
    gen_cfg = cfg.get("generation", {})
    lora_cfg = cfg.get("generation_lora", {})

    base_model_dir = args.base_model_dir or Path(gen_cfg.get("model_dir", REPO_ROOT / "AMP_generation"))
    if not Path(base_model_dir).is_absolute():
        base_model_dir = REPO_ROOT / base_model_dir

    set_seed(args.seed)
    fine_tune_settings = LoraFineTuneSettings(
        base_model_dir=base_model_dir,
        fasta_file=args.fasta,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name,
        max_length=_cfg_value(args.max_length, lora_cfg, "max_length", 256),
        train_ratio=_cfg_value(args.train_ratio, lora_cfg, "train_ratio", 0.95),
        seed=args.seed,
        use_fast_tokenizer=not args.slow_tokenizer,
        local_files_only=not args.allow_download and bool(gen_cfg.get("local_files_only", True)),
        lora_r=_cfg_value(args.lora_r, lora_cfg, "lora_r", 16),
        lora_alpha=_cfg_value(args.lora_alpha, lora_cfg, "lora_alpha", 32),
        lora_dropout=_cfg_value(args.lora_dropout, lora_cfg, "lora_dropout", 0.05),
        lora_target_modules=_cfg_value(
            args.lora_target_modules, lora_cfg, "lora_target_modules", "qkv_proj,out_proj"
        ),
        lora_bias=_cfg_value(args.lora_bias, lora_cfg, "lora_bias", "none"),
        lora_modules_to_save=args.lora_modules_to_save,
        mask_pad_labels=not args.no_mask_pad_labels,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        merge_and_save=args.merge_and_save,
    )
    training_args = _build_training_args(args, lora_cfg, args.save_dir)

    callbacks = []
    if args.early_stopping_patience is not None and not args.no_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    final_adapter = train_lora(fine_tune_settings, training_args, callbacks=callbacks)

    if args.generate_after_train:
        gen_out = args.generation_out or (args.save_dir / "generated_final.fasta")
        generation_settings = GenerationSettings(
            model_dir=base_model_dir,
            adapter_dir=final_adapter,
            out_fasta=gen_out,
            num_return_sequences=_cfg_value(args.num_return_sequences, gen_cfg, "num_return_sequences", 1000),
            min_seq_length=_cfg_value(args.min_seq_length, gen_cfg, "min_seq_length", 8),
            max_seq_length=_cfg_value(args.max_seq_length, gen_cfg, "max_seq_length", 25),
            batch_size=gen_cfg.get("batch_size", 32),
            seed=args.seed,
            temperature_min=gen_cfg.get("temperature_min", 0.8),
            temperature_max=gen_cfg.get("temperature_max", 1.0),
            top_k=gen_cfg.get("top_k", 30),
            top_p=gen_cfg.get("top_p", 0.90),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
            max_attempts_factor=gen_cfg.get("max_attempts_factor", 5),
            local_files_only=not args.allow_download and bool(gen_cfg.get("local_files_only", True)),
        )
        run_generation(generation_settings)


if __name__ == "__main__":
    main()
