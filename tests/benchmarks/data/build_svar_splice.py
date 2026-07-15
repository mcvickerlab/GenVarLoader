"""Build svar1- and svar2-backed *spliced* haplotype datasets for benchmarking.

Both datasets are built from the committed hap-safe chr22 PGEN
(``chr22_5s_hapsafe.{pgen,pvar,psam}``) so the only variable between them is the
genotype backend (genoray ``SparseVar`` vs ``SparseVar2``); the reconstructed
spliced haplotypes are byte-identical (asserted by the benchmark's parity test).

A multi-exon splice BED is synthesized from ``chr22_egenes.bed`` by chopping each
stranded eGene window into ``N_EXONS`` exons with intronic gaps, so
``var_filter="exonic"`` actually drops variants and the splice machinery
concatenates several exons per transcript. Exon boundaries are not biologically
real — this benchmarks the spliced *code path*, not gene models.

Requires the full test reference (``tests/data/fasta/hg38.fa.bgz``, fetched by
``pixi run -e dev gen``): ``SparseVar2`` conversion validates every variant's REF
against the FASTA, so the committed masked chr22 (Ns outside eGene windows) can't
be used. ``build()`` raises ``FileNotFoundError`` when inputs are missing; the
benchmark fixture turns that into a skip.

Run standalone to populate a cache dir:

    pixi run -e dev python tests/benchmarks/data/build_svar_splice.py /tmp/svar_splice
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
HAPSAFE_PGEN = DATA / "chr22_5s_hapsafe.pgen"
EGENES = DATA / "chr22_egenes.bed"
REF = REPO / "tests" / "data" / "fasta" / "hg38.fa.bgz"

N_EXONS = 4  # exons per transcript
EXON_BP = 2000  # width of each synthesized exon


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def make_splice_bed() -> pl.DataFrame:
    """Chop each stranded eGene window into ``N_EXONS`` exons spread across it."""
    egenes = pl.read_csv(
        EGENES,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )
    rows = []
    for chrom, start, end, name, _score, strand in egenes.select(
        "chrom", "start", "end", "name", "score", "strand"
    ).iter_rows():
        span = end - start
        if span < N_EXONS * EXON_BP:
            continue
        stride = span // N_EXONS
        for i in range(N_EXONS):
            es = start + i * stride
            rows.append(
                {
                    "chrom": chrom,
                    "chromStart": es,
                    "chromEnd": es + EXON_BP,
                    "strand": strand or "+",
                    "transcript_id": name,
                    "exon_number": i + 1,
                }
            )
    return pl.DataFrame(rows)


def build(out_dir: Path) -> tuple[Path, Path, Path]:
    """Build the svar1/svar2 spliced datasets into ``out_dir``.

    Returns ``(svar1_gvl, svar2_gvl, reference)``. Idempotent: skips steps whose
    outputs already exist. Raises ``FileNotFoundError`` if a required input or
    tool is missing.
    """
    import genvarloader as gvl
    from genoray import PGEN, SparseVar, SparseVar2

    for p in (HAPSAFE_PGEN, EGENES, REF):
        if not p.exists():
            raise FileNotFoundError(f"required benchmark input missing: {p}")
    for tool in ("plink2",):
        if not _have_tool(tool):
            raise FileNotFoundError(f"required tool not on PATH: {tool}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # The committed hap-safe PGEN encodes chr22 as bare '22'; rename to 'chr22'
    # (plink2 --output-chr chrM) so it matches the hg38 reference contig.
    chr_prefix = out_dir / "chr22_chr"
    if not chr_prefix.with_suffix(".pgen").exists():
        subprocess.run(
            [
                "plink2",
                "--pfile",
                str(HAPSAFE_PGEN.with_suffix("")),
                "--output-chr",
                "chrM",
                "--make-pgen",
                "--out",
                str(chr_prefix),
            ],
            check=True,
            capture_output=True,
        )
    chr_pgen = chr_prefix.with_suffix(".pgen")

    svar1 = out_dir / "chr22.svar"
    svar2 = out_dir / "chr22.svar2"
    if not svar1.exists():
        SparseVar.from_pgen(svar1, PGEN(chr_pgen), max_mem="4g", overwrite=True)
    if not (svar2 / "meta.json").exists():
        SparseVar2.from_pgen(svar2, chr_pgen, reference=REF, overwrite=True)

    bed = make_splice_bed()
    ds1 = out_dir / "svar1_splice.gvl"
    ds2 = out_dir / "svar2_splice.gvl"
    if not ds1.exists():
        gvl.write(ds1, bed, variants=SparseVar(svar1), overwrite=True)
    if not ds2.exists():
        gvl.write(ds2, bed, variants=SparseVar2(svar2), overwrite=True)
    return ds1, ds2, REF


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "svar_splice.cache"
    d1, d2, _ = build(out)
    bed = make_splice_bed()
    print(
        f"splice bed: {bed.height} exon rows, "
        f"{bed['transcript_id'].n_unique()} transcripts"
    )
    print(f"built svar1 dataset: {d1}")
    print(f"built svar2 dataset: {d2}")
