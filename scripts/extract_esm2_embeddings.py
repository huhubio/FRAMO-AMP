#!/usr/bin/env python3
"""Extract mean-pooled ESM2 embeddings from FASTA files with multi-GPU workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ESM_RESIDUE_TOKENS = set("LAGVSERTIDPKQNFYMHWCXBZUO.-")
CANONICAL_TOKENS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Record:
    index: int
    seq_id: str
    header: str
    sequence: str
    original_length: int
    replaced_unknown: int
    truncated_residues: int


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
    with path.open() as handle:
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


def sanitize_sequence(
    sequence: str,
    max_residues: int,
    canonical_only: bool,
) -> tuple[str, int, int, int]:
    allowed = CANONICAL_TOKENS | {"X"} if canonical_only else ESM_RESIDUE_TOKENS
    cleaned: list[str] = []
    replaced = 0
    original_length = 0

    for char in sequence.upper():
        if char.isspace():
            continue
        original_length += 1
        if char in allowed:
            cleaned.append(char)
        else:
            cleaned.append("X")
            replaced += 1

    truncated = max(0, len(cleaned) - max_residues)
    if truncated:
        cleaned = cleaned[:max_residues]

    return "".join(cleaned), original_length, replaced, truncated


def read_records(path: Path, max_residues: int, canonical_only: bool, limit: int | None) -> list[Record]:
    records: list[Record] = []
    for index, (header, raw_sequence) in enumerate(iter_fasta(path)):
        sequence, original_length, replaced, truncated = sanitize_sequence(
            raw_sequence,
            max_residues=max_residues,
            canonical_only=canonical_only,
        )
        if not sequence:
            raise ValueError(f"{path}: empty sequence after sanitizing FASTA record {header!r}")
        records.append(
            Record(
                index=index,
                seq_id=header.split()[0],
                header=header,
                sequence=sequence,
                original_length=original_length,
                replaced_unknown=replaced,
                truncated_residues=truncated,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def safe_stem(path: Path) -> str:
    return path.name.rsplit(".", 1)[0].replace(" ", "_")


def load_hidden_size(model_dir: Path) -> int:
    with (model_dir / "config.json").open() as handle:
        config = json.load(handle)
    return int(config["hidden_size"])


def load_max_residues(model_dir: Path) -> int:
    with (model_dir / "config.json").open() as handle:
        config = json.load(handle)
    return int(config.get("max_position_embeddings", 1026)) - 2


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in value.split(",") if item.strip() != ""]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def split_by_residue_load(records: list[Record], workers: int) -> list[list[Record]]:
    buckets: list[list[Record]] = [[] for _ in range(workers)]
    loads = [0 for _ in range(workers)]
    for record in sorted(records, key=lambda item: len(item.sequence), reverse=True):
        worker_index = min(range(workers), key=lambda item: loads[item])
        buckets[worker_index].append(record)
        loads[worker_index] += len(record.sequence)
    for bucket in buckets:
        bucket.sort(key=lambda item: item.index)
    return buckets


def batches(records: list[Record], batch_size: int) -> Iterable[list[Record]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def worker_main(
    worker_rank: int,
    gpu_id: int,
    model_dir: str,
    file_jobs: list[dict],
    hidden_size: int,
    batch_size: int,
    pooling: str,
    model_dtype: str,
    output_dtype: str,
    temp_dir: str,
    log_every: int,
    status_queue,
) -> None:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

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

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        checkpoint_model = AutoModelForMaskedLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        model = checkpoint_model.esm
        model.to(device)
        model.eval()

        for job in file_jobs:
            records: list[Record] = job["records"]
            if not records:
                continue

            embeddings = np.empty((len(records), hidden_size), dtype=np_dtype)
            indices = np.empty((len(records),), dtype=np.int64)
            write_offset = 0

            for batch_number, batch_records in enumerate(batches(records, batch_size), start=1):
                sequences = [record.sequence for record in batch_records]
                lengths = torch.tensor([len(record.sequence) for record in batch_records], device=device)
                encoded = tokenizer(
                    sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}

                with torch.inference_mode():
                    hidden = model(**encoded).last_hidden_state
                    if pooling == "mean":
                        positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
                        residue_mask = (positions >= 1) & (positions <= lengths.unsqueeze(1))
                        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1)
                        pooled = pooled / lengths.clamp_min(1).unsqueeze(1)
                    elif pooling == "cls":
                        pooled = hidden[:, 0, :]
                    else:
                        raise ValueError(f"Unsupported pooling mode: {pooling}")

                pooled_np = pooled.float().cpu().numpy().astype(np_dtype, copy=False)
                batch_len = len(batch_records)
                embeddings[write_offset : write_offset + batch_len] = pooled_np
                indices[write_offset : write_offset + batch_len] = [record.index for record in batch_records]
                write_offset += batch_len

                if log_every > 0 and batch_number % log_every == 0:
                    print(
                        f"[worker {worker_rank} cuda:{gpu_id}] {job['stem']} "
                        f"{write_offset}/{len(records)}",
                        flush=True,
                    )

            prefix = Path(temp_dir) / f"{job['stem']}.worker{worker_rank}"
            np.save(f"{prefix}.embeddings.npy", embeddings)
            np.save(f"{prefix}.indices.npy", indices)
            print(
                f"[worker {worker_rank} cuda:{gpu_id}] finished {job['stem']} "
                f"records={len(records)}",
                flush=True,
            )

        status_queue.put({"worker": worker_rank, "ok": True})
    except Exception:
        status_queue.put({"worker": worker_rank, "ok": False, "traceback": traceback.format_exc()})
        raise


def write_metadata(job: FileJob) -> None:
    with Path(job.metadata_tsv).open("w", newline="") as handle:
        fieldnames = [
            "feature_row",
            "seq_id",
            "header",
            "length",
            "original_length",
            "replaced_unknown",
            "truncated_residues",
            "sequence",
            "source_fasta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in sorted(job.records, key=lambda item: item.index):
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

    write_metadata(job)


def build_jobs(args, max_residues: int) -> list[FileJob]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[FileJob] = []
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
            max_residues=max_residues,
            canonical_only=args.canonical_only,
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
        replaced = sum(record.replaced_unknown for record in records)
        truncated = sum(record.truncated_residues for record in records)
        print(
            f"loaded {fasta_path}: records={len(records)} "
            f"replaced_unknown={replaced} truncated_residues={truncated}",
            flush=True,
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fastas", nargs="+", help="Input FASTA files")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local facebook/esm2_t33_650M_UR50D snapshot (download_backbones.py creates it)",
    )
    parser.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--output-dir", type=Path, default=Path("esm2_features"))
    parser.add_argument("--gpus", default="0", help="Comma-separated logical CUDA ids, e.g. 0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--output-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--max-residues", type=int, default=None)
    parser.add_argument("--canonical-only", action="store_true", help="Convert non-canonical residues to X")
    parser.add_argument("--limit-per-file", type=int, default=None, help="Debug option: process only N records per FASTA")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--log-every", type=int, default=200, help="Worker progress print frequency in batches; 0 disables")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import multiprocessing as mp
    import torch

    mp.set_start_method("spawn", force=True)

    gpus = parse_gpus(args.gpus)
    visible_cuda_count = torch.cuda.device_count()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run on a GPU node or remove GPU-only settings")
    if max(gpus) >= visible_cuda_count:
        raise RuntimeError(
            f"Requested GPU ids {gpus}, but torch sees only {visible_cuda_count} CUDA devices. "
            "Check CUDA_VISIBLE_DEVICES."
        )

    hidden_size = load_hidden_size(args.model_dir)
    model_max_residues = load_max_residues(args.model_dir)
    max_residues = args.max_residues or model_max_residues
    if max_residues > model_max_residues:
        raise ValueError(f"--max-residues cannot exceed model limit {model_max_residues}")

    jobs = build_jobs(args, max_residues=max_residues)
    temp_dir = Path(tempfile.mkdtemp(prefix="esm2_extract_", dir=str(args.output_dir)))
    print(f"temporary worker output: {temp_dir}", flush=True)

    worker_jobs: list[list[dict]] = [[] for _ in gpus]
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
                worker_jobs[worker_rank],
                hidden_size,
                args.batch_size,
                args.pooling,
                args.model_dtype,
                args.output_dtype,
                str(temp_dir),
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
            print(error.get("traceback", ""), file=sys.stderr)
        raise RuntimeError(f"Embedding extraction failed: worker_exit_codes={failures}")

    for job in jobs:
        merge_worker_outputs(
            job,
            worker_count=len(gpus),
            hidden_size=hidden_size,
            output_dtype=args.output_dtype,
            temp_dir=temp_dir,
            keep_temp=args.keep_temp,
        )
        print(f"wrote {job.output_npy}", flush=True)
        print(f"wrote {job.metadata_tsv}", flush=True)

    if not args.keep_temp:
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    summary_path = args.output_dir / f"{args.model_name}.{args.pooling}.summary.tsv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_fasta", "records", "embedding_npy", "metadata_tsv", "pooling", "output_dtype"],
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
                    "pooling": args.pooling,
                    "output_dtype": args.output_dtype,
                }
            )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
