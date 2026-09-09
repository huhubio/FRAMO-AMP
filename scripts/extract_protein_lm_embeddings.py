#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract mean/CLS pooled embeddings from HuggingFace protein language models.

Supported input modes:
- protbert: amino-acid sequence is converted to ProtBert's space-separated input.
- saprot: amino-acid sequence is converted to SaProt AA+structure-token format.
  If no structure TSV is provided, every residue uses a missing-structure token,
  default "#". This runs the model but should be replaced by real Foldseek/3Di
  structure tokens when available.

The output format intentionally matches scripts/extract_esm2_embeddings.py:
<stem>.<model_name>.<pooling>.npy and metadata TSV with feature_row/seq_id/sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_WITH_X = CANONICAL_AA | {"X"}


@dataclass
class Record:
    index: int
    seq_id: str
    header: str
    sequence: str
    model_input: str
    original_length: int
    replaced_unknown: int
    truncated_residues: int
    structure_sequence: str


@dataclass
class FileJob:
    input_path: str
    stem: str
    records: list[Record]
    output_npy: str
    metadata_tsv: str


def iter_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def safe_stem(path: Path) -> str:
    return path.name.rsplit(".", 1)[0].replace(" ", "_")


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in value.split(",") if item.strip() != ""]
    if not gpus:
        raise ValueError("--gpus must contain at least one logical CUDA id")
    return gpus


def sanitize_sequence(sequence: str, max_residues: int) -> tuple[str, int, int, int]:
    cleaned = []
    original_length = 0
    replaced_unknown = 0
    for char in sequence.upper():
        if char.isspace():
            continue
        original_length += 1
        if char in AA_WITH_X:
            cleaned.append(char)
        else:
            cleaned.append("X")
            replaced_unknown += 1
    truncated = max(0, len(cleaned) - max_residues)
    if truncated:
        cleaned = cleaned[:max_residues]
    return "".join(cleaned), original_length, replaced_unknown, truncated


def read_structure_tsv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"seq_id", "structure_sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            rows[row["seq_id"]] = row["structure_sequence"].strip()
    return rows


def make_model_input(
    sequence: str,
    seq_id: str,
    model_type: str,
    structure_by_id: dict[str, str],
    missing_structure_token: str,
) -> tuple[str, str]:
    if model_type == "protbert":
        return " ".join(sequence), ""
    if model_type == "saprot":
        structure = structure_by_id.get(seq_id)
        if structure is None:
            structure = missing_structure_token * len(sequence)
        structure = structure.strip()
        if len(structure) < len(sequence):
            structure = structure + (missing_structure_token * (len(sequence) - len(structure)))
        if len(structure) > len(sequence):
            structure = structure[: len(sequence)]
        # SaProt tokens pair an uppercase amino acid with a lowercase structure token.
        # The missing token "#" is kept as-is, matching the SaProt convention.
        pieces = []
        for aa, ss in zip(sequence, structure):
            pieces.append(aa + (ss if ss == "#" else ss.lower()))
        return "".join(pieces), structure
    raise ValueError(f"Unsupported model_type={model_type}")


def read_records(
    path: Path,
    model_type: str,
    max_residues: int,
    structure_by_id: dict[str, str],
    missing_structure_token: str,
    limit: int | None,
) -> list[Record]:
    records = []
    for index, (header, raw_sequence) in enumerate(iter_fasta(path)):
        seq_id = header.split()[0] if header.split() else f"seq_{index}"
        sequence, original_length, replaced_unknown, truncated = sanitize_sequence(raw_sequence, max_residues=max_residues)
        if not sequence:
            raise ValueError(f"{path}: empty sequence after sanitization for {header!r}")
        model_input, structure_sequence = make_model_input(
            sequence=sequence,
            seq_id=seq_id,
            model_type=model_type,
            structure_by_id=structure_by_id,
            missing_structure_token=missing_structure_token,
        )
        records.append(
            Record(
                index=index,
                seq_id=seq_id,
                header=header,
                sequence=sequence,
                model_input=model_input,
                original_length=original_length,
                replaced_unknown=replaced_unknown,
                truncated_residues=truncated,
                structure_sequence=structure_sequence,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def split_by_residue_load(records: list[Record], workers: int) -> list[list[Record]]:
    buckets = [[] for _ in range(workers)]
    loads = [0 for _ in range(workers)]
    for record in sorted(records, key=lambda r: len(r.sequence), reverse=True):
        worker = min(range(workers), key=lambda i: loads[i])
        buckets[worker].append(record)
        loads[worker] += len(record.sequence)
    for bucket in buckets:
        bucket.sort(key=lambda r: r.index)
    return buckets


def batches(records: list[Record], batch_size: int) -> Iterable[list[Record]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def infer_hidden_size(model_dir: Path) -> int:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ["hidden_size", "d_model", "embed_dim"]:
        if key in config:
            return int(config[key])
    raise KeyError(f"Could not infer hidden size from {model_dir / 'config.json'}")


def infer_model_max_residues(model_dir: Path, model_type: str, fallback: int) -> int:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    max_positions = config.get("max_position_embeddings") or config.get("max_positions")
    if not max_positions:
        return fallback
    # SaProt consumes one token per AA+structure pair, not two tokenizer positions.
    return max(1, int(max_positions) - 2)


def worker_main(
    worker_rank: int,
    gpu_id: int,
    model_dir: str,
    model_type: str,
    file_jobs: list[dict[str, object]],
    hidden_size: int,
    batch_size: int,
    pooling: str,
    model_dtype: str,
    output_dtype: str,
    temp_dir: str,
    trust_remote_code: bool,
    log_every: int,
    status_queue,
) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        dtype_lookup = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_lookup[model_dtype]
        np_dtype = np.dtype(output_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model_class = AutoModelForMaskedLM if model_type == "saprot" else AutoModel
        model = model_class.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        model.eval()

        for job in file_jobs:
            records: list[Record] = job["records"]  # type: ignore[assignment]
            if not records:
                continue
            embeddings = np.empty((len(records), hidden_size), dtype=np_dtype)
            indices = np.empty((len(records),), dtype=np.int64)
            write_offset = 0

            for batch_number, batch_records in enumerate(batches(records, batch_size), start=1):
                encoded = tokenizer(
                    [record.model_input for record in batch_records],
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                special_mask = encoded.pop("special_tokens_mask")
                encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                special_mask = special_mask.to(device, non_blocking=True)

                with torch.inference_mode():
                    if model_type == "saprot":
                        output = model(**encoded, output_hidden_states=True)
                        hidden = output.hidden_states[-1]
                    else:
                        output = model(**encoded)
                        hidden = output.last_hidden_state
                    if pooling == "cls":
                        pooled = hidden[:, 0, :]
                    elif pooling == "mean":
                        residue_mask = (encoded["attention_mask"] > 0) & (special_mask == 0)
                        token_counts = residue_mask.sum(dim=1).clamp_min(1)
                        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / token_counts.unsqueeze(1)
                    else:
                        raise ValueError(f"Unsupported pooling={pooling}")

                pooled_np = pooled.float().cpu().numpy().astype(np_dtype, copy=False)
                batch_len = len(batch_records)
                embeddings[write_offset : write_offset + batch_len] = pooled_np
                indices[write_offset : write_offset + batch_len] = [record.index for record in batch_records]
                write_offset += batch_len

                if log_every > 0 and batch_number % log_every == 0:
                    print(
                        f"[worker {worker_rank} cuda:{gpu_id}] {job['stem']} {write_offset}/{len(records)}",
                        flush=True,
                    )

            prefix = Path(temp_dir) / f"{job['stem']}.worker{worker_rank}"
            np.save(f"{prefix}.embeddings.npy", embeddings)
            np.save(f"{prefix}.indices.npy", indices)
            print(f"[worker {worker_rank} cuda:{gpu_id}] finished {job['stem']} records={len(records)}", flush=True)

        status_queue.put({"worker": worker_rank, "ok": True})
    except Exception:
        status_queue.put({"worker": worker_rank, "ok": False, "traceback": traceback.format_exc()})
        raise


def write_metadata(job: FileJob, model_type: str) -> None:
    with Path(job.metadata_tsv).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "feature_row",
            "seq_id",
            "header",
            "length",
            "original_length",
            "replaced_unknown",
            "truncated_residues",
            "sequence",
            "structure_sequence",
            "model_type",
            "source_fasta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in sorted(job.records, key=lambda r: r.index):
            writer.writerow(
                {
                    "feature_row": record.index,
                    "seq_id": record.seq_id,
                    "header": record.header,
                    "length": len(record.sequence),
                    "original_length": record.original_length,
                    "replaced_unknown": record.replaced_unknown,
                    "truncated_residues": record.truncated_residues,
                    "sequence": record.sequence,
                    "structure_sequence": record.structure_sequence,
                    "model_type": model_type,
                    "source_fasta": job.input_path,
                }
            )


def merge_worker_outputs(
    job: FileJob,
    worker_count: int,
    hidden_size: int,
    output_dtype: str,
    temp_dir: Path,
    keep_temp: bool,
    model_type: str,
) -> None:
    final = np.lib.format.open_memmap(
        job.output_npy,
        mode="w+",
        dtype=np.dtype(output_dtype),
        shape=(len(job.records), hidden_size),
    )
    seen = np.zeros((len(job.records),), dtype=bool)
    for worker_rank in range(worker_count):
        prefix = temp_dir / f"{job.stem}.worker{worker_rank}"
        embeddings_path = Path(f"{prefix}.embeddings.npy")
        indices_path = Path(f"{prefix}.indices.npy")
        if not embeddings_path.exists():
            continue
        embeddings = np.load(embeddings_path, mmap_mode="r")
        indices = np.load(indices_path)
        final[indices] = embeddings
        seen[indices] = True
        if not keep_temp:
            embeddings_path.unlink(missing_ok=True)
            indices_path.unlink(missing_ok=True)
    final.flush()
    if not seen.all():
        missing = np.where(~seen)[0][:10].tolist()
        raise RuntimeError(f"{job.input_path}: missing embeddings for rows {missing}")
    write_metadata(job, model_type=model_type)


def build_jobs(args, max_residues: int) -> list[FileJob]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure_by_id = read_structure_tsv(args.structure_tsv)
    jobs = []
    suffix = f"{args.model_name}.{args.pooling}"
    for fasta in args.fastas:
        fasta_path = Path(fasta)
        stem = safe_stem(fasta_path)
        output_npy = args.output_dir / f"{stem}.{suffix}.npy"
        metadata_tsv = args.output_dir / f"{stem}.{suffix}.metadata.tsv"
        if not args.overwrite and (output_npy.exists() or metadata_tsv.exists()):
            raise FileExistsError(f"Output already exists for {fasta_path}; use --overwrite to replace it")
        records = read_records(
            fasta_path,
            model_type=args.model_type,
            max_residues=max_residues,
            structure_by_id=structure_by_id,
            missing_structure_token=args.missing_structure_token,
            limit=args.limit_per_file,
        )
        jobs.append(
            FileJob(
                input_path=str(fasta_path),
                stem=stem,
                records=records,
                output_npy=str(output_npy),
                metadata_tsv=str(metadata_tsv),
            )
        )
        print(
            f"loaded {fasta_path}: records={len(records)} "
            f"replaced_unknown={sum(r.replaced_unknown for r in records)} "
            f"truncated_residues={sum(r.truncated_residues for r in records)}",
            flush=True,
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fastas", nargs="+")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-type", choices=["protbert", "saprot"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--output-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--max-residues", type=int, default=None)
    parser.add_argument("--max-residues-fallback", type=int, default=1022)
    parser.add_argument("--structure-tsv", type=Path, default=None)
    parser.add_argument("--missing-structure-token", default="#")
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import torch

    mp.set_start_method("spawn", force=True)
    gpus = parse_gpus(args.gpus)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if max(gpus) >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPUs {gpus}, but torch sees {torch.cuda.device_count()} CUDA devices")

    hidden_size = infer_hidden_size(args.model_dir)
    model_max = infer_model_max_residues(args.model_dir, args.model_type, fallback=args.max_residues_fallback)
    max_residues = args.max_residues or model_max
    if max_residues > model_max:
        raise ValueError(f"--max-residues cannot exceed model limit {model_max}")

    jobs = build_jobs(args, max_residues=max_residues)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{args.model_name}_extract_", dir=str(args.output_dir)))
    print(f"temporary worker output: {temp_dir}", flush=True)

    worker_jobs: list[list[dict[str, object]]] = [[] for _ in gpus]
    for job in jobs:
        split_records = split_by_residue_load(job.records, len(gpus))
        for worker_rank, records in enumerate(split_records):
            worker_jobs[worker_rank].append({"stem": job.stem, "records": records})

    status_queue = mp.Queue()
    processes = []
    for worker_rank, gpu_id in enumerate(gpus):
        process = mp.Process(
            target=worker_main,
            args=(
                worker_rank,
                gpu_id,
                str(args.model_dir),
                args.model_type,
                worker_jobs[worker_rank],
                hidden_size,
                args.batch_size,
                args.pooling,
                args.model_dtype,
                args.output_dtype,
                str(temp_dir),
                args.trust_remote_code,
                args.log_every,
                status_queue,
            ),
        )
        process.start()
        processes.append(process)

    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(process.exitcode)

    statuses = []
    while not status_queue.empty():
        statuses.append(status_queue.get())
    errors = [status for status in statuses if not status.get("ok")]
    if failures or errors:
        for error in errors:
            print(error.get("traceback", ""))
        raise RuntimeError(f"Embedding extraction failed: worker_exit_codes={failures}")

    for job in jobs:
        merge_worker_outputs(
            job,
            worker_count=len(gpus),
            hidden_size=hidden_size,
            output_dtype=args.output_dtype,
            temp_dir=temp_dir,
            keep_temp=args.keep_temp,
            model_type=args.model_type,
        )
        print(f"wrote {job.output_npy}", flush=True)
        print(f"wrote {job.metadata_tsv}", flush=True)

    if not args.keep_temp:
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    summary_path = args.output_dir / f"{args.model_name}.{args.pooling}.summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_fasta",
                "records",
                "embedding_npy",
                "metadata_tsv",
                "model_type",
                "pooling",
                "output_dtype",
                "max_residues",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "input_fasta": job.input_path,
                    "records": len(job.records),
                    "embedding_npy": job.output_npy,
                    "metadata_tsv": job.metadata_tsv,
                    "model_type": args.model_type,
                    "pooling": args.pooling,
                    "output_dtype": args.output_dtype,
                    "max_residues": max_residues,
                }
            )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
