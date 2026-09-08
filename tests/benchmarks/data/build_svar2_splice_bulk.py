"""Build a *bulk* SVAR2 spliced-variant dataset for benchmarking at cohort scale.

Companion to ``build_svar_splice.py``. That builder is pinned to the committed
5-sample chr22 PGEN, so it cannot vary cohort size -- the dominant axis for the
SVAR2 spliced *variant* decode path added in PR #286, whose Rayon chunking is
bounded per-worker. This builder synthesizes arbitrarily large cohorts with the
vcfixture-rs ``bulk`` CLI so ``n_samples`` can grow toward AoU scale.

Genotypes are drawn i.i.d. from a fitted 1kGP site-frequency spectrum (no LD),
which vcfixture's own ablation shows is a ~0x lever on BCF parse-bound readers --
irrelevant to a decode-throughput benchmark. Exon boundaries are synthetic: this
benchmarks the spliced *code path*, not gene models.

``SparseVar2.from_vcf(..., no_reference=True, skip_out_of_scope=True)`` avoids
REF-vs-FASTA validation (vcfixture draws REF i.i.d.) and drops symbolic/breakend
SV records the short-read codec can't represent. A synthetic reference FASTA
(correct contig name + length) is still written for ``gvl.write`` / ``Dataset.open``.

Requires the vcfixture-rs ``bulk`` CLI (``VCFIXTURE_BIN`` env var or ``vcfixture``
on PATH) plus ``bcftools`` and ``samtools``. ``build()`` raises
``FileNotFoundError`` when a tool is missing; the benchmark fixture turns that
into a skip.

Run standalone to populate a cache dir:

    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev python \
        tests/benchmarks/data/build_svar2_splice_bulk.py /tmp/svar2_bulk 2000 64
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_RECORDS = 20_000
DEFAULT_SEED = 42
EXONS_PER_TX = 3
EXON_BP = 200
INTRON_BP = 10


@dataclass(frozen=True)
class BulkSpliceFixture:
    gvl_path: Path
    reference: Path
    splice_bed: pl.DataFrame
    n_samples: int
    n_transcripts: int
    records: int


def _which_vcfixture() -> str:
    cand = os.environ.get("VCFIXTURE_BIN") or shutil.which("vcfixture")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError(
            "vcfixture-rs bulk CLI not found: set VCFIXTURE_BIN or put `vcfixture` "
            "on PATH (build with `cargo build --release --features cli` in the "
            "vcfixture-rs repo)."
        )
    return str(cand)


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FileNotFoundError(f"required tool not on PATH: {tool}")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _contig_span(bcf: Path, contig: str) -> int:
    """vcfixture declares ``##contig length`` = populated span (last record POS)."""
    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(bcf)], check=True, capture_output=True, text=True
    ).stdout
    pat = re.compile(
        r"##contig=<[^>]*ID=" + re.escape(contig) + r"[^>]*length=(\d+)", re.IGNORECASE
    )
    for line in hdr.splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1))
    raise ValueError(f"no ##contig length for {contig!r} in {bcf} header")


def gen_cohort(
    out_dir: Path,
    n_samples: int,
    records: int,
    *,
    contig: str = "chr1",
    seed: int = DEFAULT_SEED,
    profile: str = "germline-1kgp",
    payload: str = "gt-only",
) -> tuple[Path, int]:
    """Generate + bi-allelic-normalize a cohort BCF. Returns ``(bcf, span)``."""
    vcfixture = _which_vcfixture()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cohort_s{n_samples}_r{records}_seed{seed}"
    raw = out_dir / f"{stem}.raw.bcf"
    norm = out_dir / f"{stem}.bcf"
    if not norm.exists():
        _run(
            [
                vcfixture,
                "bulk",
                "--profile",
                profile,
                "--samples",
                str(n_samples),
                "--contigs",
                contig,
                "--records",
                str(records),
                "--payload",
                payload,
                "--seed",
                str(seed),
                "-o",
                str(raw),
            ]
        )
        # GVL requires bi-allelic, atomized variants (from_vcf skips SV/symbolic ALTs).
        _run(["bcftools", "norm", "-m-", "-Ob", "-o", str(norm), str(raw)])
        _run(["bcftools", "index", "-f", str(norm)])
    return norm, _contig_span(norm, contig)


def make_splice_bed(
    contig: str,
    span: int,
    n_transcripts: int,
    *,
    exons_per_tx: int = EXONS_PER_TX,
    exon_len: int = EXON_BP,
) -> pl.DataFrame:
    """Tile ``n_transcripts`` transcripts of ``exons_per_tx`` exons inside ``[0, span)``.

    Strand alternates so the minus-strand reverse-complement path is exercised.
    """
    stride = max(exon_len * (exons_per_tx + 1), 1)
    usable = max(span - stride, stride)
    rows = []
    for t in range(n_transcripts):
        base = (t * stride) % usable
        strand = "+" if t % 2 == 0 else "-"
        for e in range(exons_per_tx):
            start = base + e * (exon_len + INTRON_BP)
            end = min(start + exon_len, span)
            if start >= end:
                continue
            rows.append(
                {
                    "chrom": contig,
                    "chromStart": start,
                    "chromEnd": end,
                    "strand": strand,
                    "transcript_id": f"T{t}",
                    "exon_number": e + 1,
                }
            )
    return pl.DataFrame(rows)


def _write_reference(path: Path, contig: str, length: int, seed: int = 0) -> None:
    """A synthetic reference is fine: no_reference=True skips REF validation, and the
    spliced *variant* read returns stored REF/ALT, not reference lookups."""
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.frombuffer(b"ACGT", "S1"), size=length).tobytes().decode()
    path.write_text(f">{contig}\n{seq}\n")
    _run(["samtools", "faidx", str(path)])


def build(
    out_dir: Path,
    *,
    n_samples: int,
    n_transcripts: int,
    records: int = DEFAULT_RECORDS,
    seed: int = DEFAULT_SEED,
    contig: str = "chr1",
) -> BulkSpliceFixture:
    """Build one bulk SVAR2 spliced-variant dataset. Idempotent (cached by shape)."""
    import genvarloader as gvl
    from genoray import SparseVar2

    _require("bcftools")
    _require("samtools")

    root = Path(out_dir) / f"s{n_samples}_r{records}_{n_transcripts}tx"
    gvl_path = root / "ds.gvl"
    ref = root / "ref.fa"

    bcf, span = gen_cohort(
        Path(out_dir) / "cohorts", n_samples, records, contig=contig, seed=seed
    )
    bed = make_splice_bed(contig, span, n_transcripts)

    if gvl_path.exists() and ref.exists():
        return BulkSpliceFixture(gvl_path, ref, bed, n_samples, n_transcripts, records)

    root.mkdir(parents=True, exist_ok=True)
    _write_reference(ref, contig, span + EXON_BP)

    svar2 = root / "store.svar2"
    if not (svar2 / "meta.json").exists():
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
    gvl.write(gvl_path, bed, variants=SparseVar2(svar2), overwrite=True)
    return BulkSpliceFixture(gvl_path, ref, bed, n_samples, n_transcripts, records)


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("svar2_bulk.cache")
    n_records = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RECORDS
    n_tx = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    fx = build(out, n_samples=200, n_transcripts=n_tx, records=n_records)
    print(f"gvl: {fx.gvl_path}")
    print(
        f"splice bed: {fx.splice_bed.height} exon rows, "
        f"{fx.splice_bed['transcript_id'].n_unique()} transcripts"
    )
