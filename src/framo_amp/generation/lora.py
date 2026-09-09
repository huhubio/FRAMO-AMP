"""LoRA fine-tuning support for the bundled AMP generator."""
from __future__ import annotations

import csv
import logging
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback

from framo_amp.data import read_fasta
from framo_amp.generation.distributed import get_rank, is_distributed, unwrap_model

PathLike = Union[str, Path]

LOGGER = logging.getLogger("framo_amp.generation.lora")


@dataclass
class LoraFineTuneSettings:
    """Settings that are specific to FRAMO-AMP generator LoRA fine-tuning."""

    base_model_dir: PathLike = "./AMP_generation"
    fasta_file: PathLike = ""
    save_dir: PathLike = "./lora_runs/amp_generation_lora"
    dataset_name: str = "peptide"
    max_length: int = 256
    train_ratio: float = 0.95
    seed: int = 42
    use_fast_tokenizer: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "qkv_proj,out_proj"
    lora_bias: str = "none"
    lora_modules_to_save: Optional[str] = None
    mask_pad_labels: bool = True
    gradient_checkpointing: bool = True
    merge_and_save: bool = False


def clean_training_sequence(seq: str) -> str:
    """Keep only uppercase letters in a peptide training sequence."""
    return re.sub(r"[^A-Z]", "", seq.upper())


def load_training_sequences(fasta_file: PathLike) -> list[str]:
    """Read a FASTA file into cleaned non-empty peptide sequences."""
    _, seqs = read_fasta(fasta_file)
    return [seq for seq in (clean_training_sequence(seq) for seq in seqs) if seq]


def split_sequences(
    sequences: Sequence[str],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Shuffle and split sequences into train/validation lists."""
    if not sequences:
        raise ValueError("No non-empty sequences were found in the training FASTA.")
    seqs = list(sequences)
    random.Random(seed).shuffle(seqs)
    split_idx = int(len(seqs) * train_ratio)
    split_idx = min(max(split_idx, 1), len(seqs))
    train_seqs = seqs[:split_idx]
    val_seqs = seqs[split_idx:]
    return train_seqs, val_seqs


class FastaPeptideDataset(Dataset):
    """Causal-LM dataset over peptide strings."""

    def __init__(
        self,
        sequences: Sequence[str],
        tokenizer,
        *,
        max_length: int,
        mask_pad_labels: bool = True,
    ) -> None:
        self.samples = [seq for seq in sequences if seq]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.mask_pad_labels = mask_pad_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.samples[idx]
        enc = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        if self.mask_pad_labels:
            labels = labels.masked_fill(attention_mask == 0, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_lora_config(settings: LoraFineTuneSettings):
    """Build a PEFT LoRA config for ProGen attention projections."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "LoRA fine-tuning requires peft. Install it with `pip install peft`."
        ) from exc

    target_modules = [
        item.strip()
        for item in settings.lora_target_modules.split(",")
        if item.strip()
    ]
    modules_to_save = None
    if settings.lora_modules_to_save:
        modules_to_save = [
            item.strip()
            for item in settings.lora_modules_to_save.split(",")
            if item.strip()
        ]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=target_modules,
        bias=settings.lora_bias,
        modules_to_save=modules_to_save,
        inference_mode=False,
    )


def _enable_gradient_checkpointing(model) -> None:  # noqa: ANN001
    model.config.use_cache = False
    candidates = [model]
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for candidate in list(candidates):
        if hasattr(candidate, "transformer"):
            candidates.append(candidate.transformer)

    for candidate in candidates:
        if hasattr(candidate, "gradient_checkpointing_enable"):
            candidate.gradient_checkpointing_enable()
            return
        if hasattr(candidate, "_set_gradient_checkpointing"):
            candidate._set_gradient_checkpointing(enable=True)
            return
    LOGGER.warning("Gradient checkpointing is not available on this model.")


def _log_trainable_parameters(model) -> None:  # noqa: ANN001
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    LOGGER.info(
        "trainable params: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total if total else 0.0,
    )


class SaveEpochAdapterCallback(TrainerCallback):
    """Save PEFT adapters to save_dir/epoch_k at the end of each epoch."""

    def __init__(self, save_dir: PathLike, tokenizer) -> None:  # noqa: ANN001
        self.save_dir = Path(save_dir)
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001, ANN201
        if not state.is_world_process_zero or model is None:
            return
        epoch = int(round(state.epoch or 0))
        if epoch < 1:
            return
        epoch_dir = self.save_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("saving LoRA adapter -> %s", epoch_dir)
        unwrap_model(model).save_pretrained(epoch_dir)
        self.tokenizer.save_pretrained(epoch_dir)


class EvalLossCsvCallback(TrainerCallback):
    """Append evaluation metrics to eval_losses.csv."""

    def __init__(self, save_dir: PathLike, dataset_name: str) -> None:
        self.csv_path = Path(save_dir) / "eval_losses.csv"
        self.dataset_name = dataset_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001, ANN201
        if not state.is_world_process_zero or metrics is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "epoch",
                    "dataset",
                    "eval_loss",
                    "eval_runtime",
                    "eval_samples_per_second",
                    "eval_steps_per_second",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": state.epoch,
                    "dataset": self.dataset_name,
                    "eval_loss": metrics.get("eval_loss"),
                    "eval_runtime": metrics.get("eval_runtime"),
                    "eval_samples_per_second": metrics.get("eval_samples_per_second"),
                    "eval_steps_per_second": metrics.get("eval_steps_per_second"),
                }
            )


def train_lora(
    settings: LoraFineTuneSettings,
    training_args,
    *,
    callbacks: Optional[list[TrainerCallback]] = None,
) -> Path:  # noqa: ANN001
    """Fine-tune the AMP generator with PEFT LoRA and return the final adapter path."""
    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA fine-tuning requires peft. Install it with `pip install peft`."
        ) from exc

    base_model_dir = Path(settings.base_model_dir)
    if not base_model_dir.is_dir():
        raise FileNotFoundError(base_model_dir)
    save_dir = Path(settings.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        use_fast=settings.use_fast_tokenizer,
        **load_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    LOGGER.info("loading base generator from %s", base_model_dir)
    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), **load_kwargs)
    lora_config = build_lora_config(settings)
    model = get_peft_model(model, lora_config)
    if settings.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    if get_rank() == 0:
        _log_trainable_parameters(model)

    sequences = load_training_sequences(settings.fasta_file)
    train_seqs, val_seqs = split_sequences(
        sequences,
        train_ratio=settings.train_ratio,
        seed=settings.seed,
    )
    if get_rank() == 0:
        lengths = [len(seq) for seq in sequences]
        LOGGER.info(
            "loaded %s sequences for %s: %s train, %s val; length min=%s max=%s mean=%.2f median=%.1f",
            len(sequences),
            settings.dataset_name,
            len(train_seqs),
            len(val_seqs),
            min(lengths),
            max(lengths),
            statistics.mean(lengths),
            statistics.median(lengths),
        )

    train_dataset = FastaPeptideDataset(
        train_seqs,
        tokenizer,
        max_length=settings.max_length,
        mask_pad_labels=settings.mask_pad_labels,
    )
    eval_dataset = (
        FastaPeptideDataset(
            val_seqs,
            tokenizer,
            max_length=settings.max_length,
            mask_pad_labels=settings.mask_pad_labels,
        )
        if val_seqs and getattr(training_args, "do_eval", False)
        else None
    )

    training_args.output_dir = str(save_dir)
    training_args.save_strategy = "epoch"
    training_args.save_safetensors = True
    if is_distributed() and getattr(training_args, "ddp_find_unused_parameters", None) is None:
        training_args.ddp_find_unused_parameters = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.add_callback(SaveEpochAdapterCallback(save_dir, tokenizer))
    if eval_dataset is not None:
        trainer.add_callback(EvalLossCsvCallback(save_dir, settings.dataset_name))
    for callback in callbacks or []:
        trainer.add_callback(callback)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    final_dir = save_dir / "final"
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model(trainer.model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        LOGGER.info("final LoRA adapter saved -> %s", final_dir)

        if settings.merge_and_save:
            merged_dir = save_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model = unwrap_model(trainer.model).merge_and_unload()
            merged_model.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            LOGGER.info("merged full model saved -> %s", merged_dir)

    return final_dir
