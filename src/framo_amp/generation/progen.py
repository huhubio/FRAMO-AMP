"""Inference helpers for the bundled ProGen-based AMP generator."""
from __future__ import annotations

import json
import logging
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from framo_amp.data import write_fasta
from framo_amp.generation.distributed import (
    all_gather_object,
    barrier,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed_if_needed,
    is_rank_zero,
)

PathLike = Union[str, Path]

LOGGER = logging.getLogger("framo_amp.generation")


@dataclass
class GenerationSettings:
    """Sampling settings for AMP sequence generation."""

    model_dir: PathLike = "./AMP_generation"
    adapter_dir: Optional[PathLike] = None
    out_fasta: PathLike = "./results/generated_amp.fasta"
    out_json: Optional[PathLike] = None
    num_return_sequences: int = 1000
    min_seq_length: int = 8
    max_seq_length: int = 25
    batch_size: int = 32
    seed: int = 42
    temperature_min: float = 0.8
    temperature_max: float = 1.0
    top_k: int = 30
    top_p: float = 0.90
    repetition_penalty: float = 1.1
    max_attempts_factor: int = 5
    device: Optional[str] = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    merge_adapter: bool = True


def clean_generated_sequence(seq: str) -> str:
    """Keep only uppercase amino-acid-like letters from decoded model text."""
    return re.sub(r"[^A-Z]", "", seq.upper())


def _model_device(model) -> torch.device:  # noqa: ANN001
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _load_peft_model(base_model, adapter_dir: PathLike, load_kwargs: dict, merge: bool):  # noqa: ANN001, ANN202
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading LoRA adapters requires peft. Install it with `pip install peft`."
        ) from exc

    model = PeftModel.from_pretrained(base_model, str(adapter_dir), **load_kwargs)
    if merge:
        model = model.merge_and_unload()
    return model


def load_generation_model(settings: GenerationSettings):
    """Load tokenizer and causal language model, optionally with a LoRA adapter."""
    model_dir = Path(settings.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    load_kwargs = {
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
    }

    # Custom ProGen code is loaded from the local model directory. Under
    # torchrun, rank 0 warms the cache first to avoid file-copy races.
    if get_world_size() > 1:
        if is_rank_zero():
            _ = AutoConfig.from_pretrained(str(model_dir), **load_kwargs)
            del _
        barrier()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    if settings.adapter_dir is not None:
        adapter_dir = Path(settings.adapter_dir)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(adapter_dir)
        model = _load_peft_model(model, adapter_dir, load_kwargs, settings.merge_adapter)

    device = _resolve_device(settings.device)
    model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    settings: GenerationSettings,
    seeds: list[int],
) -> list[Optional[str]]:
    """Generate a batch of candidate sequences and filter by requested length."""
    if not seeds:
        return []

    device = _model_device(model)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer must define a BOS or EOS token.")

    rngs = [random.Random(int(seed)) for seed in seeds]
    target_lengths = [rng.randint(settings.min_seq_length, settings.max_seq_length) for rng in rngs]
    temperatures = [rng.uniform(settings.temperature_min, settings.temperature_max) for rng in rngs]

    input_ids = torch.full((len(seeds), 1), int(bos_id), dtype=torch.long, device=device)
    max_target = max(target_lengths)
    temperature = max(1e-6, sum(temperatures) / len(temperatures))

    set_seed(int(seeds[0]) % (2**32 - 1))
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        min_length=max_target + 1,
        max_length=max_target + 4,
        do_sample=True,
        temperature=temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        repetition_penalty=settings.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    seqs: list[Optional[str]] = []
    for row in out:
        seq = clean_generated_sequence(tokenizer.decode(row, skip_special_tokens=True))
        if settings.min_seq_length <= len(seq) <= settings.max_seq_length:
            seqs.append(seq)
        else:
            seqs.append(None)
    return seqs


def _generate_rank_shard(model, tokenizer, settings: GenerationSettings) -> list[tuple[int, str]]:  # noqa: ANN001
    rank = get_rank()
    world = get_world_size()
    indices = list(range(rank, settings.num_return_sequences, world))
    pending = list(indices)
    shard: list[tuple[int, str]] = []
    max_attempts = len(indices) * max(1, settings.max_attempts_factor)
    attempts = 0
    batch_size = max(1, int(settings.batch_size))

    while pending and attempts < max_attempts:
        current = pending[:batch_size]
        pending = pending[batch_size:]
        seeds = [settings.seed + idx * 1000003 + attempts for idx in current]
        attempts += len(current)

        seqs = generate_batch(model, tokenizer, settings, seeds)
        failed: list[int] = []
        for idx, seq in zip(current, seqs):
            if seq is None:
                failed.append(idx)
            else:
                shard.append((idx, seq))
        pending.extend(failed)

        if is_rank_zero() and indices:
            done = len(shard)
            if done == len(indices) or done % max(1, len(indices) // 10) == 0:
                LOGGER.info("rank 0 progress: %s/%s", done, len(indices))

    if pending:
        LOGGER.warning("rank %s generated %s/%s valid sequences", rank, len(shard), len(indices))
    return shard


def _write_outputs(sequences: list[str], settings: GenerationSettings) -> None:
    out_fasta = Path(settings.out_fasta)
    write_fasta(
        ((f"Generated_sequence_{i}", seq) for i, seq in enumerate(sequences, 1)),
        out_fasta,
    )

    out_json = Path(settings.out_json) if settings.out_json else out_fasta.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(settings).items()
        },
        "sequences": sequences,
    }
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if sequences:
        lengths = [len(seq) for seq in sequences]
        LOGGER.info(
            "saved %s sequences -> %s; length min=%s max=%s mean=%.2f median=%.1f",
            len(sequences),
            out_fasta,
            min(lengths),
            max(lengths),
            statistics.mean(lengths),
            statistics.median(lengths),
        )
    else:
        LOGGER.warning("no valid sequences generated; wrote empty outputs to %s", out_fasta)
    LOGGER.info("generation metadata -> %s", out_json)


def run_generation(settings: GenerationSettings) -> list[str]:
    """Generate AMP candidates and write FASTA/JSON outputs.

    The function supports both single-process execution and torchrun/DDP. Each
    rank generates a disjoint subset; rank 0 gathers all shards and writes the
    final files.
    """
    init_distributed_if_needed()
    tokenizer, model = load_generation_model(settings)
    LOGGER.info(
        "generating %s sequences from %s (world_size=%s)",
        settings.num_return_sequences,
        settings.model_dir,
        get_world_size(),
    )

    shard = _generate_rank_shard(model, tokenizer, settings)
    gathered = all_gather_object(shard)

    sequences: list[str] = []
    if is_rank_zero():
        flat = [item for part in gathered for item in part]
        flat.sort(key=lambda item: item[0])
        sequences = [seq for _, seq in flat][: settings.num_return_sequences]
        _write_outputs(sequences, settings)

    barrier()
    return sequences
