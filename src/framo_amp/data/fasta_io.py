"""FASTA reading and writing utilities.

This module replaces the original ``fa2seqs`` / ``fa2seqs_name`` / ``seqs2fa``
helpers with a more pythonic, well-typed API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Union

PathLike = Union[str, Path]


def iter_fasta(path: PathLike) -> Iterator[Tuple[str, str]]:
    """Stream a FASTA file as ``(header, sequence)`` tuples.

    The header keeps the leading ``>`` stripped but preserves any inline
    description. Sequences are concatenated across line breaks and stripped of
    whitespace. The iterator is memory-friendly for very large files.
    """
    path = Path(path)
    header: str | None = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def read_fasta(path: PathLike) -> Tuple[List[str], List[str]]:
    """Read a FASTA file and return ``(headers, sequences)`` as parallel lists.

    Parameters
    ----------
    path:
        Path to a FASTA file.

    Returns
    -------
    headers, sequences:
        Two parallel lists. ``headers[i]`` is the description line (without the
        leading ``>``) for ``sequences[i]``.
    """
    headers: List[str] = []
    seqs: List[str] = []
    for h, s in iter_fasta(path):
        headers.append(h)
        seqs.append(s)
    return headers, seqs


def write_fasta(
    records: Iterable[Tuple[str, str]] | Iterable[str],
    path: PathLike,
    *,
    width: int | None = None,
) -> None:
    """Write ``(header, sequence)`` pairs (or bare sequences) to a FASTA file.

    Parameters
    ----------
    records:
        Either an iterable of ``(header, sequence)`` tuples, or an iterable of
        sequence strings. When bare sequences are supplied, headers default to
        the 0-based index.
    path:
        Output FASTA path. Parent directories are created if missing.
    width:
        Optional line wrap width for sequence lines. ``None`` means write each
        sequence on a single line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            if isinstance(rec, str):
                header, seq = str(i), rec
            else:
                header, seq = rec
            header = header.lstrip(">").strip()
            fh.write(f">{header}\n")
            if width is None or width <= 0:
                fh.write(seq + "\n")
            else:
                for k in range(0, len(seq), width):
                    fh.write(seq[k : k + width] + "\n")


def strip_terminal_stop(seqs: Iterable[str]) -> List[str]:
    """Drop a trailing ``*`` stop codon from each sequence, if present."""
    return [s[:-1] if s.endswith("*") else s for s in seqs]
