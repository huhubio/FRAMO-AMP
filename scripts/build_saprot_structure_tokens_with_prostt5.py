#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict Foldseek/ProstT5 3Di tokens from FASTA for SaProt inputs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_FOLDSEEK = Path("foldseek")
THREEDI_ALPHABET = set("pynwrqhgdlvtmfsaeikc#x")


def iter_fasta(path: Path):
    header = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).replace(" ", "").upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks).replace(" ", "").upper()


def read_records(path: Path) -> list[dict[str, str]]:
    records = []
    for index, (header, sequence) in enumerate(iter_fasta(path)):
        seq_id = header.split()[0] if header.split() else f"seq_{index}"
        records.append({"seq_id": seq_id, "header": header, "sequence": sequence})
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    return records


def run_cmd(cmd: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def make_db(
    foldseek_bin: Path,
    fasta: Path,
    db_prefix: Path,
    prostt5_model: Path,
    threads: int,
    use_gpu: bool,
    log_path: Path | None,
    overwrite: bool,
) -> None:
    if overwrite:
        for path in db_prefix.parent.glob(db_prefix.name + "*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    if Path(str(db_prefix) + ".dbtype").exists():
        return
    db_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(foldseek_bin),
        "createdb",
        str(fasta),
        str(db_prefix),
        "--prostt5-model",
        str(prostt5_model),
        "--threads",
        str(threads),
        "-v",
        "3",
    ]
    if use_gpu:
        cmd.extend(["--gpu", "1"])
    run_cmd(cmd, log_path=log_path)


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    rows = []
    header = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    rows.append((header, "".join(chunks).replace(" ", "")))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        rows.append((header, "".join(chunks).replace(" ", "")))
    return rows


def read_lookup(db_prefix: Path) -> dict[str, str]:
    lookup_path = Path(str(db_prefix) + ".lookup")
    if not lookup_path.exists():
        return {}
    lookup = {}
    with lookup_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                lookup[parts[0]] = parts[1]
    return lookup


def _read_single_indexed_sequence_db(db_path: Path, index_path: Path) -> dict[str, str]:
    rows = {}
    with index_path.open(encoding="utf-8", errors="replace") as index_handle, db_path.open("rb") as db_handle:
        for line in index_handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key, offset_text, length_text = parts[:3]
            db_handle.seek(int(offset_text))
            raw = db_handle.read(int(length_text))
            sequence = raw.replace(b"\x00", b"").strip().decode("ascii", errors="replace")
            if sequence:
                rows[key] = sequence
    return rows


def read_indexed_sequence_db(db_prefix: Path) -> dict[str, str]:
    index_path = Path(str(db_prefix) + ".index")
    db_path = db_prefix
    if index_path.exists() and db_path.exists():
        return _read_single_indexed_sequence_db(db_path=db_path, index_path=index_path)

    rows = {}
    shard_indexes = sorted(
        db_prefix.parent.glob(db_prefix.name + ".index.*"),
        key=lambda path: int(path.name.rsplit(".", 1)[-1]) if path.name.rsplit(".", 1)[-1].isdigit() else path.name,
    )
    for shard_index in shard_indexes:
        suffix = shard_index.name.rsplit(".", 1)[-1]
        shard_db = Path(str(db_prefix) + "." + suffix)
        if shard_db.exists():
            rows.update(_read_single_indexed_sequence_db(db_path=shard_db, index_path=shard_index))
    return rows


def read_structure_rows_from_ss_db(
    db_prefix: Path,
    records: list[dict[str, str]],
) -> tuple[list[tuple[str, str]], list[dict[str, str]]] | None:
    ss_prefix = Path(str(db_prefix) + "_ss")
    indexed_rows = read_indexed_sequence_db(ss_prefix)
    if not indexed_rows:
        return None

    lookup = read_lookup(db_prefix)
    record_by_id = {record["seq_id"]: record for record in records}
    rows_by_seq_id = {}
    for key, sequence in indexed_rows.items():
        seq_id = lookup.get(key)
        if seq_id:
            rows_by_seq_id[seq_id] = sequence

    rows = []
    if len(rows_by_seq_id) == len(records):
        for record in records:
            rows.append((record["seq_id"], rows_by_seq_id[record["seq_id"]]))
    elif len(indexed_rows) == len(records):
        for key in sorted(indexed_rows, key=lambda value: int(value) if value.isdigit() else value):
            rows.append((records[len(rows)]["seq_id"], indexed_rows[key]))
    else:
        return None

    ok, score = is_structure_candidate(rows, records)
    candidates = [
        {
            "prefix": str(ss_prefix),
            "converted_fasta": "<indexed_db_direct>",
            "rows": len(rows),
            "score": f"{score:.6f}",
            "selected_candidate": str(ok),
        }
    ]
    if not ok:
        return None
    for seq_id, structure in rows:
        if seq_id in record_by_id and len(structure.strip()) != len(record_by_id[seq_id]["sequence"]):
            return None
    return rows, candidates


def convert_candidate(foldseek_bin: Path, prefix: Path, out_fasta: Path, log_path: Path | None) -> bool:
    try:
        run_cmd(
            [
                str(foldseek_bin),
                "convert2fasta",
                str(prefix),
                str(out_fasta),
                "--use-header-file",
                "1",
                "-v",
                "1",
            ],
            log_path=log_path,
        )
        return out_fasta.exists() and out_fasta.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def is_structure_candidate(rows: list[tuple[str, str]], records: list[dict[str, str]]) -> tuple[bool, float]:
    if len(rows) != len(records):
        return False, 0.0
    length_matches = 0
    non_aa_matches = 0
    alphabet_hits = 0
    total_chars = 0
    for (_header, sequence), record in zip(rows, records):
        seq = sequence.strip().lower()
        if len(seq) == len(record["sequence"]):
            length_matches += 1
        if seq.upper() != record["sequence"].upper():
            non_aa_matches += 1
        for char in seq:
            total_chars += 1
            if char in THREEDI_ALPHABET:
                alphabet_hits += 1
    if not rows or total_chars == 0:
        return False, 0.0
    score = (
        (length_matches / len(records))
        + (non_aa_matches / len(records))
        + (alphabet_hits / total_chars)
    )
    ok = length_matches == len(records) and non_aa_matches > 0 and alphabet_hits / total_chars > 0.95
    return ok, score


def discover_structure_fasta(
    foldseek_bin: Path,
    db_prefix: Path,
    records: list[dict[str, str]],
    tmp_dir: Path,
    log_path: Path | None,
) -> tuple[Path, list[tuple[str, str]], list[dict[str, str]]]:
    direct_rows = read_structure_rows_from_ss_db(db_prefix, records)
    if direct_rows is not None:
        rows, candidates = direct_rows
        return Path(str(db_prefix) + "_ss"), rows, candidates

    candidates = []
    for dbtype in sorted(db_prefix.parent.glob(db_prefix.name + "*.dbtype")):
        prefix = Path(str(dbtype)[: -len(".dbtype")])
        out_fasta = tmp_dir / f"{prefix.name}.converted.fasta"
        if not convert_candidate(foldseek_bin, prefix, out_fasta, log_path=log_path):
            continue
        rows = parse_fasta(out_fasta)
        ok, score = is_structure_candidate(rows, records)
        candidates.append(
            {
                "prefix": str(prefix),
                "converted_fasta": str(out_fasta),
                "rows": len(rows),
                "score": f"{score:.6f}",
                "selected_candidate": str(ok),
            }
        )
        if ok:
            return out_fasta, rows, candidates
    if candidates:
        best = max(candidates, key=lambda row: float(row["score"]))
        raise RuntimeError(f"No valid 3Di candidate found; best={best}")
    raise RuntimeError(f"No convertible Foldseek DB candidates found under {db_prefix.parent}")


def normalize_structure_sequence(sequence: str, length: int) -> str:
    seq = sequence.strip().lower()
    seq = "".join(char if char in THREEDI_ALPHABET else "#" for char in seq)
    if len(seq) < length:
        seq = seq + ("#" * (length - len(seq)))
    if len(seq) > length:
        seq = seq[:length]
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--prostt5_model", type=Path, required=True)
    parser.add_argument("--out_tsv", type=Path, required=True)
    parser.add_argument("--audit_tsv", type=Path, default=None)
    parser.add_argument("--tmp_dir", type=Path, required=True)
    parser.add_argument("--foldseek_bin", type=Path, default=DEFAULT_FOLDSEEK)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    foldseek_bin = args.foldseek_bin
    if not foldseek_bin.exists():
        resolved = shutil.which(str(foldseek_bin))
        if resolved:
            foldseek_bin = Path(resolved)
    if not foldseek_bin.exists():
        raise FileNotFoundError(f"foldseek not found: {args.foldseek_bin}")
    if not args.prostt5_model.exists():
        raise FileNotFoundError(f"ProstT5 model path not found: {args.prostt5_model}")
    if args.out_tsv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out_tsv}; use --overwrite")

    records = read_records(args.fasta)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_tsv is not None:
        args.audit_tsv.parent.mkdir(parents=True, exist_ok=True)

    db_prefix = args.tmp_dir / "foldseek_prostt5_db"
    make_db(
        foldseek_bin=foldseek_bin,
        fasta=args.fasta,
        db_prefix=db_prefix,
        prostt5_model=args.prostt5_model,
        threads=args.threads,
        use_gpu=args.gpu,
        log_path=args.log,
        overwrite=args.overwrite,
    )
    structure_fasta, structure_rows, candidates = discover_structure_fasta(
        foldseek_bin=foldseek_bin,
        db_prefix=db_prefix,
        records=records,
        tmp_dir=args.tmp_dir,
        log_path=args.log,
    )

    output_rows = []
    audit_rows = []
    for record, (structure_header, raw_structure) in zip(records, structure_rows):
        structure = normalize_structure_sequence(raw_structure, len(record["sequence"]))
        output_rows.append({"seq_id": record["seq_id"], "structure_sequence": structure})
        audit_rows.append(
            {
                "seq_id": record["seq_id"],
                "length": len(record["sequence"]),
                "structure_length": len(structure),
                "structure_header": structure_header,
                "source_structure_fasta": str(structure_fasta),
                "header": record["header"],
                "sequence": record["sequence"],
                "structure_sequence": structure,
            }
        )

    with args.out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_id", "structure_sequence"], delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    if args.audit_tsv is not None:
        with args.audit_tsv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "seq_id",
                "length",
                "structure_length",
                "structure_header",
                "source_structure_fasta",
                "header",
                "sequence",
                "structure_sequence",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(audit_rows)
        candidate_path = args.audit_tsv.with_suffix(args.audit_tsv.suffix + ".candidate_dbs.tsv")
        with candidate_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["prefix", "converted_fasta", "rows", "score", "selected_candidate"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(candidates)

    print(
        f"wrote {args.out_tsv} records={len(output_rows)} "
        f"structure_fasta={structure_fasta}",
        flush=True,
    )


if __name__ == "__main__":
    main()
