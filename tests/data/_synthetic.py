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
import pysam
from vcfixture import Number, Type, VcfBuilder

# (contig, length) — sized to span every variant locus with headroom.
CONTIGS: list[tuple[str, int]] = [
    ("chr1", 1_300_000),
    ("chr2", 100_000),
    ("chr19", 1_300_000),
    ("chr20", 1_300_000),
]

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

    # hg38 parity: chr1 begins with an N-masked telomere. tests/unit/dataset/
    # test_ref_ds.py expects chr1:0-150 to read as N.
    seqs["chr1"][0:150] = np.frombuffer(b"N" * 150, dtype="S1")

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


def build_source_vcf(reference_path: str | Path) -> "object":
    """Re-encode the canonical source VCF against the synthetic reference.

    Returns a built ``vcfixture`` document (has ``.render()`` / ``.write()``).
    Contigs (names + lengths) are read from ``reference_path``'s FASTA index so
    the VCF header always matches the reference. REF alleles are taken verbatim
    from the records below and match the engineered reference bases written by
    ``write_synthetic_reference``.
    """
    with pysam.FastaFile(str(reference_path)) as fa:
        contigs = list(zip(fa.references, fa.lengths))
    # VCFv4.0 to match the original hand-authored source.vcf: under VCFv4.4+
    # an INFO field with Number=. (used here for AC/AF) is rejected by the
    # noodles VCF parser that genoray's SparseVar.from_vcf relies on via oxbow.
    b = VcfBuilder(
        samples=["NA00001", "NA00002", "NA00003"],
        contigs=contigs,
        fileformat="VCFv4.0",
    )
    # INFO defs (must match the original header).
    b.info("NS", Number.ONE, Type.INTEGER)
    b.info("AN", Number.ONE, Type.INTEGER)
    b.info("AC", Number.DOT, Type.INTEGER)
    b.info("DP", Number.ONE, Type.INTEGER)
    b.info("AF", Number.DOT, Type.FLOAT)
    b.info("AA", Number.ONE, Type.STRING)
    b.info("DB", Number.FLAG, Type.FLAG)
    b.info("H2", Number.FLAG, Type.FLAG)
    # FORMAT defs.
    b.fmt("GT")
    b.fmt("VAF", Number.A, Type.FLOAT)
    b.fmt("GQ", Number.ONE, Type.INTEGER)
    b.fmt("DP", Number.ONE, Type.INTEGER)
    b.fmt("HQ", Number.fixed(2), Type.INTEGER)
    # FILTER defs.
    b.filter("q10", "Quality below 10")
    b.filter("s50", "Less than 50% of samples have data")

    # chr19 block — FORMAT GT:VAF:HQ. Non-GT FORMAT values are not assertion-
    # critical (no test reads them); GT is reproduced exactly.
    b.record("chr19", 111, ref="N", alt=["C"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1010696, ref="GAGA", alt=["G"], gt=["1|0", "0|0", "0/0"])
    b.record("chr19", 1010696, ref="GAGACGG", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1010696, ref="GAGACGGGGCC", alt=["G"], gt=["0|1", "1|1", "0/0"])
    b.record("chr19", 1110696, ref="A", alt=["TTT"], gt=["0|1", "1|1", "0/0"])
    b.record("chr19", 1110696, ref="A", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1210696, ref="C", alt=["G"], gt=["1|.", "0/1", "1|1"])
    b.record("chr19", 1210696, ref="C", alt=["G"], gt=[".|1", "0|0", "0/0"])
    b.record("chr19", 1210697, ref="T", alt=["G"], gt=["0/0", "1|0", "0/1"])
    b.record("chr19", 1210697, ref="T", alt=["A"], gt=["0/0", "1|0", "0/1"])

    # chr20 block — carries INFO (test_sitesonly) and IDs/FILTERs.
    b.record(
        "chr20", 14370, ref="N", alt=["A"], ids=["rs6054257"], qual=29.0, filter=(),
        gt=["0|0", "1|0", "1/1"], info={"NS": 3, "DP": 14, "AF": [0.5], "DB": True, "H2": True},
    )
    b.record(
        "chr20", 17330, ref="N", alt=["A"], qual=3.0, filter=["q10"],
        gt=["0|0", "0|1", "0/0"], info={"NS": 3, "DP": 11, "AF": [0.017]},
    )
    b.record(
        "chr20", 1110696, ref="G", alt=["A", "T"], ids=["rs6040355"], qual=67.0, filter=(),
        gt=["1|2", "2|1", "2/2"],
        info={"NS": 2, "DP": 10, "AF": [0.333, 0.667], "AA": "T", "DB": True},
    )
    b.record(
        "chr20", 1234567, ref="A", alt=["GA", "AC"], ids=["microsat1"], qual=50.0, filter=(),
        gt=["0/1", "0/2", "./."],
        info={"NS": 3, "DP": 9, "AA": "G", "AN": 6, "AC": [3, 1]},
    )
    return b.build()
