"""Synthetic reference + source-VCF re-encoding for the toy test fixtures.

Replaces the hg38 download and hand-authored ``source.vcf``. The reference is
random ACGT with bases at each variant locus overwritten to match the source
VCF's REF alleles, plus single-base flank guards 5' of every indel anchor so
``bcftools norm`` cannot left-shift indels off their hardcoded positions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# (contig, length) — sized to span every variant locus with headroom.
CONTIGS: list[tuple[str, int]] = [("chr19", 1_300_000), ("chr20", 1_300_000)]

# (contig, 1-based pos, REF) — the longest REF at each shared position.
REF_OVERWRITES: list[tuple[str, int, str]] = [
    ("chr19", 111, "N"),
    ("chr19", 1010696, "GAGACGGGGCC"),
    ("chr19", 1110696, "A"),
    ("chr19", 1210696, "C"),
    ("chr19", 1210697, "T"),
    ("chr20", 14370, "N"),
    ("chr20", 17330, "N"),
    ("chr20", 1110696, "G"),
    ("chr20", 1234567, "A"),
]

# (contig, 1-based anchor pos, guard base) — overwrite the base at index pos-2
# (immediately 5' of the anchor) to break any leftward repeat. Guard base is
# chosen != anchor base and != REF tail base for that record.
FLANK_GUARDS: list[tuple[str, int, str]] = [
    ("chr19", 1010696, "T"),  # anchor G, REF tail C -> T is safe
    ("chr19", 1110696, "G"),  # anchor A, ALT TTT/REF A -> G is safe
    ("chr20", 1234567, "T"),  # anchor A, REF A / ALTs GA,AC -> T is safe
]

_BASES = np.frombuffer(b"ACGT", dtype="S1")


def write_synthetic_reference(path: str | Path, seed: int = 0) -> Path:
    """Write a bgzipped, faidx-indexed synthetic reference to *path*.

    Returns the path to the bgzipped FASTA (``.fa.bgz``).
    """
    path = Path(path)
    if path.suffix != ".bgz" or path.with_suffix("").suffix != ".fa":
        raise ValueError(f"path must end in .fa.bgz, got {path}")
    rng = np.random.default_rng(seed)

    seqs: dict[str, np.ndarray] = {}
    for contig, length in CONTIGS:
        seqs[contig] = rng.choice(_BASES, size=length)

    for contig, pos, ref in REF_OVERWRITES:
        arr = seqs[contig]
        start = pos - 1  # 0-based
        ref_bytes = np.frombuffer(ref.encode(), dtype="S1")
        arr[start : start + len(ref)] = ref_bytes

    for contig, pos, guard in FLANK_GUARDS:
        # Base immediately 5' of the anchor (0-based index pos-2).
        seqs[contig][pos - 2] = guard.encode()

    # Write plain FASTA (60-col wrapped), then bgzip + faidx via samtools.
    plain = path.with_suffix("")  # strip .bgz -> .fa
    with open(plain, "w") as f:
        for contig, _ in CONTIGS:
            f.write(f">{contig}\n")
            seq = seqs[contig].tobytes().decode()
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

    try:
        subprocess.run(["bgzip", "-f", "-o", str(path), str(plain)], check=True)
    finally:
        plain.unlink(missing_ok=True)
    subprocess.run(["samtools", "faidx", str(path)], check=True)
    return path
