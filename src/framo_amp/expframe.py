"""Interpretable ExpFrame post-prediction prioritization utilities."""

from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from framo_amp.data import iter_fasta, write_fasta


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Candidate:
    seq_id: str
    sequence: str
    ensemble_score: float


@dataclass(frozen=True)
class SimilarityHit:
    target: str
    value: float


@dataclass(frozen=True)
class EmbeddingEvidence:
    centroid_distance: float
    centroid_threshold: float
    nearest_distance: float
    nearest_threshold: float
    nearest_reference: str

    @property
    def passes(self) -> bool:
        return (
            self.centroid_distance <= self.centroid_threshold
            and self.nearest_distance <= self.nearest_threshold
        )


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "amp", "positive"}


def load_high_confidence_candidates(
    path: Path,
    min_ensemble_score: float = 2.0 / 3.0,
) -> tuple[list[Candidate], list[dict[str, str]]]:
    """Load AMP calls, reject non-canonical sequences, and remove duplicates."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"prediction table is empty: {path}")
    required = {"seq_id", "sequence", "ensemble_prediction", "ensemble_score"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")

    candidates: list[Candidate] = []
    audit: list[dict[str, str]] = []
    seen_sequences: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        seq_id = row["seq_id"].strip()
        sequence = "".join(row["sequence"].split()).upper()
        try:
            score = float(row["ensemble_score"])
        except ValueError as exc:
            raise ValueError(f"invalid ensemble_score on row {row_number}: {row['ensemble_score']!r}") from exc
        reason = "retained_for_expframe"
        if not seq_id:
            reason = "missing_seq_id"
        elif seq_id in seen_ids:
            raise ValueError(f"duplicate seq_id in predictions: {seq_id}")
        elif not _is_true(row["ensemble_prediction"]):
            reason = "ensemble_non_amp"
        elif score + 1e-12 < min_ensemble_score:
            reason = "below_min_ensemble_score"
        elif not sequence:
            reason = "empty_sequence"
        elif not set(sequence).issubset(CANONICAL_AMINO_ACIDS):
            reason = "non_canonical_amino_acid"
        elif sequence in seen_sequences:
            reason = f"duplicate_sequence_of:{seen_sequences[sequence]}"
        else:
            candidates.append(Candidate(seq_id, sequence, score))
            seen_sequences[sequence] = seq_id
        if seq_id:
            seen_ids.add(seq_id)
        audit.append(
            {
                "row": str(row_number),
                "seq_id": seq_id,
                "sequence": sequence,
                "ensemble_score": f"{score:.8f}",
                "decision": reason,
            }
        )
    if not candidates:
        raise ValueError("no high-confidence canonical non-redundant candidates remain")
    return candidates, audit


def load_reference_fasta(path: Path) -> list[tuple[str, str]]:
    """Load a unique, canonical positive AMP reference FASTA."""
    records: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for header, raw_sequence in iter_fasta(path):
        seq_id = header.split()[0]
        sequence = "".join(raw_sequence.split()).upper()
        if not seq_id or seq_id in seen_ids:
            raise ValueError(f"empty or duplicate reference ID: {seq_id!r}")
        if not sequence or not set(sequence).issubset(CANONICAL_AMINO_ACIDS):
            raise ValueError(f"reference {seq_id!r} is empty or contains non-canonical amino acids")
        seen_ids.add(seq_id)
        if sequence not in seen_sequences:
            records.append((seq_id, sequence))
            seen_sequences.add(sequence)
    if len(records) < 2:
        raise ValueError("positive reference FASTA must contain at least two unique sequences")
    return records


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_mmseqs(
    executable: str,
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    temp_dir: Path,
    threads: int,
    evalue_threshold: float,
) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(output_tsv),
            str(temp_dir),
            "--format-output",
            "query,target,evalue,bits,pident,qcov,tcov",
            "-e",
            str(evalue_threshold),
            "--threads",
            str(threads),
        ],
        check=True,
    )


def run_foldseek(
    executable: str,
    query_structures: Path,
    target_structures: Path,
    output_tsv: Path,
    temp_dir: Path,
    threads: int,
) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "easy-search",
            str(query_structures),
            str(target_structures),
            str(output_tsv),
            str(temp_dir),
            "--format-output",
            "query,target,evalue,alntmscore,qtmscore,ttmscore",
            "--threads",
            str(threads),
        ],
        check=True,
    )


def read_mmseqs_hits(path: Path, evalue_threshold: float) -> dict[str, SimilarityHit]:
    """Return the lowest-E-value hit for every query."""
    hits: dict[str, SimilarityHit] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0].lower() == "query":
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: expected query, target, evalue")
            try:
                evalue = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid E-value {parts[2]!r}") from exc
            query, target = parts[0], parts[1]
            if evalue < evalue_threshold and (query not in hits or evalue < hits[query].value):
                hits[query] = SimilarityHit(target, evalue)
    return hits


def read_foldseek_hits(path: Path, tm_threshold: float) -> dict[str, SimilarityHit]:
    """Return the highest-TM-score hit for every query."""
    hits: dict[str, SimilarityHit] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0].lower() == "query":
                continue
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_number}: expected query, target and TM-score")
            try:
                scores = [float(value) for value in parts[3:6] if value != ""]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid TM-score columns") from exc
            if not scores:
                continue
            score = max(scores)
            query, target = parts[0], parts[1]
            if score > tm_threshold and (query not in hits or score > hits[query].value):
                hits[query] = SimilarityHit(target, score)
    return hits


def load_embeddings(npy_path: Path, metadata_path: Path) -> tuple[list[str], np.ndarray]:
    matrix = np.asarray(np.load(npy_path), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D: {npy_path}")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != matrix.shape[0] or not rows or "seq_id" not in rows[0]:
        raise ValueError(f"embedding/metadata mismatch: {npy_path} and {metadata_path}")
    ids = [row["seq_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate seq_id in embedding metadata: {metadata_path}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"embedding matrix contains NaN or infinity: {npy_path}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"embedding matrix contains zero-length vectors: {npy_path}")
    return ids, matrix / norms


def _nearest_cosine_distances(
    queries: np.ndarray,
    references: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        similarities = queries[start : start + chunk_size] @ references.T
        nearest_indices = np.argmax(similarities, axis=1)
        nearest_similarities = similarities[np.arange(len(nearest_indices)), nearest_indices]
        distances.append(1.0 - nearest_similarities)
        indices.append(nearest_indices)
    return np.concatenate(distances), np.concatenate(indices)


def embedding_evidence(
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    reference_ids: list[str],
    reference_embeddings: np.ndarray,
    quantile: float = 0.95,
    chunk_size: int = 2048,
) -> dict[str, EmbeddingEvidence]:
    """Compare candidates with positive AMP centroid and neighbour distributions.

    Cosine-distance cutoffs are learned exclusively from the positive references:
    the selected quantile of reference-to-centroid distances and leave-one-out
    nearest-neighbour distances. A candidate must pass both cutoffs.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("embedding quantile must be in (0, 1]")
    if len(reference_embeddings) < 2:
        raise ValueError("at least two positive reference embeddings are required")
    if candidate_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("candidate and reference embedding dimensions differ")
    if len(candidate_ids) != len(candidate_embeddings) or len(reference_ids) != len(reference_embeddings):
        raise ValueError("embedding IDs and rows differ")

    centroid = reference_embeddings.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if not math.isfinite(centroid_norm) or centroid_norm == 0:
        raise ValueError("positive reference centroid is invalid")
    centroid /= centroid_norm
    reference_centroid_distances = 1.0 - reference_embeddings @ centroid
    centroid_threshold = float(np.quantile(reference_centroid_distances, quantile))

    reference_similarity = reference_embeddings @ reference_embeddings.T
    np.fill_diagonal(reference_similarity, -np.inf)
    reference_nn_distances = 1.0 - np.max(reference_similarity, axis=1)
    nearest_threshold = float(np.quantile(reference_nn_distances, quantile))

    candidate_centroid_distances = 1.0 - candidate_embeddings @ centroid
    candidate_nn_distances, candidate_nn_indices = _nearest_cosine_distances(
        candidate_embeddings, reference_embeddings, chunk_size
    )
    return {
        seq_id: EmbeddingEvidence(
            centroid_distance=float(candidate_centroid_distances[index]),
            centroid_threshold=centroid_threshold,
            nearest_distance=float(candidate_nn_distances[index]),
            nearest_threshold=nearest_threshold,
            nearest_reference=reference_ids[int(candidate_nn_indices[index])],
        )
        for index, seq_id in enumerate(candidate_ids)
    }


def prioritize_candidates(
    candidates: list[Candidate],
    sequence_hits: dict[str, SimilarityHit],
    structure_hits: dict[str, SimilarityHit],
    embeddings: dict[str, EmbeddingEvidence],
) -> list[dict[str, object]]:
    """Assign mutually exclusive ExpFrame classes in evidence-priority order."""
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        sequence_hit = sequence_hits.get(candidate.seq_id)
        structure_hit = structure_hits.get(candidate.seq_id)
        embedding = embeddings.get(candidate.seq_id)
        if sequence_hit is not None:
            expframe_class = "sequence_similarity"
        elif structure_hit is not None:
            expframe_class = "structure_similarity"
        elif embedding is not None and embedding.passes:
            expframe_class = "divergent_generation"
        else:
            expframe_class = "not_retained"
        rows.append(
            {
                "seq_id": candidate.seq_id,
                "sequence": candidate.sequence,
                "length": len(candidate.sequence),
                "ensemble_score": f"{candidate.ensemble_score:.8f}",
                "sequence_hit": int(sequence_hit is not None),
                "sequence_target": sequence_hit.target if sequence_hit else "",
                "sequence_evalue": f"{sequence_hit.value:.6g}" if sequence_hit else "",
                "structure_hit": int(structure_hit is not None),
                "structure_target": structure_hit.target if structure_hit else "",
                "structure_tm_score": f"{structure_hit.value:.8f}" if structure_hit else "",
                "centroid_distance": f"{embedding.centroid_distance:.8f}" if embedding else "",
                "centroid_threshold": f"{embedding.centroid_threshold:.8f}" if embedding else "",
                "nearest_distance": f"{embedding.nearest_distance:.8f}" if embedding else "",
                "nearest_threshold": f"{embedding.nearest_threshold:.8f}" if embedding else "",
                "nearest_reference": embedding.nearest_reference if embedding else "",
                "embedding_pass": int(embedding.passes) if embedding else "",
                "expframe_class": expframe_class,
                "retained": int(expframe_class != "not_retained"),
            }
        )
    return rows


RESULT_FIELDS = [
    "seq_id",
    "sequence",
    "length",
    "ensemble_score",
    "sequence_hit",
    "sequence_target",
    "sequence_evalue",
    "structure_hit",
    "structure_target",
    "structure_tm_score",
    "centroid_distance",
    "centroid_threshold",
    "nearest_distance",
    "nearest_threshold",
    "nearest_reference",
    "embedding_pass",
    "expframe_class",
    "retained",
]


def write_expframe_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    audit: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "expframe_results.tsv", rows, RESULT_FIELDS)
    write_tsv(
        output_dir / "input_audit.tsv",
        audit,
        ["row", "seq_id", "sequence", "ensemble_score", "decision"],
    )
    retained = [row for row in rows if row["retained"] == 1]
    write_fasta(((str(row["seq_id"]), str(row["sequence"])) for row in retained), output_dir / "retained_amp.fasta")
    for class_name in ("sequence_similarity", "structure_similarity", "divergent_generation"):
        selected = [row for row in rows if row["expframe_class"] == class_name]
        write_fasta(
            ((str(row["seq_id"]), str(row["sequence"])) for row in selected),
            output_dir / f"{class_name}.fasta",
        )
    summary_rows = []
    for class_name in ("sequence_similarity", "structure_similarity", "divergent_generation", "not_retained"):
        summary_rows.append(
            {"expframe_class": class_name, "count": sum(row["expframe_class"] == class_name for row in rows)}
        )
    summary_rows.append({"expframe_class": "retained_total", "count": len(retained)})
    write_tsv(output_dir / "summary.tsv", summary_rows, ["expframe_class", "count"])
