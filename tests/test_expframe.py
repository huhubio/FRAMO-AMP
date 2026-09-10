import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from framo_amp.expframe import (
    Candidate,
    SimilarityHit,
    embedding_evidence,
    load_high_confidence_candidates,
    prioritize_candidates,
    read_foldseek_hits,
    read_mmseqs_hits,
    write_expframe_outputs,
)


class ExpFrameTests(unittest.TestCase):
    def test_candidate_filtering_and_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.tsv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["seq_id", "sequence", "ensemble_prediction", "ensemble_score"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"seq_id": "seq", "sequence": "ACDE", "ensemble_prediction": "1", "ensemble_score": "1"},
                        {"seq_id": "dup", "sequence": "ACDE", "ensemble_prediction": "1", "ensemble_score": "1"},
                        {"seq_id": "bad", "sequence": "ACDX", "ensemble_prediction": "1", "ensemble_score": "1"},
                        {"seq_id": "neg", "sequence": "KKLL", "ensemble_prediction": "0", "ensemble_score": "0.33"},
                    ]
                )
            candidates, audit = load_high_confidence_candidates(path)
            self.assertEqual([item.seq_id for item in candidates], ["seq"])
            self.assertEqual(
                [row["decision"] for row in audit],
                ["retained_for_expframe", "duplicate_sequence_of:seq", "non_canonical_amino_acid", "ensemble_non_amp"],
            )

    def test_hit_thresholds_and_evidence_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            mmseqs = directory / "mmseqs.tsv"
            mmseqs.write_text("seq1\tref1\t0.0009\nseq2\tref2\t0.001\n", encoding="utf-8")
            foldseek = directory / "foldseek.tsv"
            foldseek.write_text("seq2\tref2\t1e-5\t0.51\t0.4\t0.3\nseq3\tref3\t1e-4\t0.5\n", encoding="utf-8")
            seq_hits = read_mmseqs_hits(mmseqs, 1e-3)
            struct_hits = read_foldseek_hits(foldseek, 0.5)
            candidates = [Candidate(f"seq{i}", "ACDE", 1.0) for i in range(1, 5)]
            candidate_vectors = np.asarray([[1, 0], [1, 0], [1, 0], [-1, 0]], dtype=np.float32)
            reference_vectors = np.asarray([[1, 0], [0.99, 0.1], [0.98, -0.1]], dtype=np.float32)
            candidate_vectors /= np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
            reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)
            evidence = embedding_evidence(
                [item.seq_id for item in candidates],
                candidate_vectors,
                ["ref1", "ref2", "ref3"],
                reference_vectors,
            )
            rows = prioritize_candidates(candidates, seq_hits, struct_hits, evidence)
            self.assertEqual(
                [row["expframe_class"] for row in rows],
                ["sequence_similarity", "structure_similarity", "divergent_generation", "not_retained"],
            )
            write_expframe_outputs(directory / "out", rows, [])
            retained = (directory / "out" / "retained_amp.fasta").read_text(encoding="utf-8")
            self.assertIn(">seq1", retained)
            self.assertNotIn(">seq4", retained)

    def test_precomputed_hit_objects(self) -> None:
        rows = prioritize_candidates(
            [Candidate("x", "ACDE", 1.0)],
            {"x": SimilarityHit("ref", 1e-6)},
            {},
            {},
        )
        self.assertEqual(rows[0]["retained"], 1)


if __name__ == "__main__":
    unittest.main()
