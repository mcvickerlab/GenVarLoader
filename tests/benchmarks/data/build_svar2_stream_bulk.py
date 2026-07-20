"""Build a *bulk* SVAR2 dataset (contiguous regions) for streaming benchmarks.

vcfixture-rs ``bulk`` synthesizes arbitrarily large cohorts (``n_samples`` toward
AoU scale, the dominating axis) drawn i.i.d. from a fitted 1kGP site-frequency
spectrum. Genotypes have no LD -- irrelevant to a decode/gather-throughput
benchmark, per vcfixture's own ablation. ``SparseVar2.from_vcf(no_reference=True,
skip_out_of_scope=True)`` avoids REF-vs-FASTA validation (vcfixture draws REF
i.i.d.) and drops symbolic/breakend records the short-read codec can't represent.
A synthetic reference (correct contig name + length) is written for gvl.write /
Dataset.open. ``build`` raises FileNotFoundError when a tool is missing; the
benchmark turns that into a skip.

Run standalone:
    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev python \
        tests/benchmarks/data/build_svar2_stream_bulk.py /tmp/svar2_stream_bulk 2000 200
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


@dataclass(frozen=True)
class BulkStreamFixture:
    gvl_path: Path
    svar2_path: Path
    reference: Path
    bed: pl.DataFrame
    n_samples: int
    records: int


def _which_vcfixture() -> str:
    cand = os.environ.get("VCFIXTURE_BIN") or shutil.which("vcfixture")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError(
            "vcfixture-rs bulk CLI not found: set VCFIXTURE_BIN or put `vcfixture` "
            "on PATH (build with `cargo build --release --features cli` in vcfixture-rs)."
        )
    return str(cand)


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FileNotFoundError(f"required tool not on PATH: {tool}")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _contig_span(bcf: Path, contig: str) -> int:
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
    """Generate + bi-allelic-normalize a cohort BCF. Returns (bcf, span). Cached by shape."""
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


def make_contiguous_bed(
    contig: str, span: int, n_regions: int, region_len: int
) -> pl.DataFrame:
    """Tile n_regions non-overlapping windows of region_len inside [0, span)."""
    stride = max(span // max(n_regions, 1), region_len)
    rows = []
    for i in range(n_regions):
        start = (i * stride) % max(span - region_len, 1)
        rows.append(
            {"chrom": contig, "chromStart": start, "chromEnd": start + region_len}
        )
    return pl.DataFrame(rows)


def _write_reference(path: Path, contig: str, length: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.frombuffer(b"ACGT", "S1"), size=length).tobytes().decode()
    path.write_text(f">{contig}\n{seq}\n")
    _run(["samtools", "faidx", str(path)])


def build(
    out_dir: Path,
    *,
    n_samples: int,
    records: int = DEFAULT_RECORDS,
    seed: int = DEFAULT_SEED,
    contig: str = "chr1",
    region_len: int = 1000,
    n_regions: int = 64,
) -> BulkStreamFixture:
    """Build one bulk SVAR2 streaming dataset. Idempotent (cached by shape)."""
    import genvarloader as gvl
    from genoray import SparseVar2

    _require("bcftools")
    _require("samtools")

    root = Path(out_dir) / f"s{n_samples}_r{records}_{n_regions}x{region_len}"
    gvl_path = root / "ds.gvl"
    ref = root / "ref.fa"
    svar2 = root / "store.svar2"

    bcf, span = gen_cohort(
        Path(out_dir) / "cohorts", n_samples, records, contig=contig, seed=seed
    )
    bed = make_contiguous_bed(contig, span, n_regions, region_len)

    if gvl_path.exists() and ref.exists() and (svar2 / "meta.json").exists():
        return BulkStreamFixture(gvl_path, svar2, ref, bed, n_samples, records)

    root.mkdir(parents=True, exist_ok=True)
    _write_reference(ref, contig, span + region_len)
    if not (svar2 / "meta.json").exists():
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
    gvl.write(gvl_path, bed, variants=SparseVar2(svar2), overwrite=True)
    return BulkStreamFixture(gvl_path, svar2, ref, bed, n_samples, records)


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("svar2_stream_bulk.cache")
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    records = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RECORDS
    fx = build(out, n_samples=n_samples, records=records)
    print(f"gvl oracle: {fx.gvl_path}")
    print(f"svar2 store: {fx.svar2_path}")
    print(f"bed: {fx.bed.height} regions")
