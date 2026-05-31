"""Regenerate the committed chr22 1kGP + GEUVADIS benchmark slice.

Run once on a host with /carter mounted:

    pixi run -e dev python tests/benchmarks/data/build_realistic.py

Produces (all under tests/benchmarks/data/):
  - samples.txt           the 5 chosen sample IDs
  - chr22_egenes.bed      benchmark regions (copied)
  - chr22.masked.fa.gz     masked chr22 reference (+ .fai, .gzi)
  - chr22_geuv.gvl/        the gvl dataset (5 samples, chr22, read-depth tracks)

Requires the test reference fasta (run `pixi run -e dev gen` first to fetch it).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent

PLINK_PREFIX = Path("/carter/users/dlaub/data/1kGP/plink2/hg38.norm")
RNA_DIR = Path("/carter/users/dlaub/data/1kGP-rna-seq")
SAMPLE_MAP = RNA_DIR / "sample_id_to_bigwig.csv"
BW_CHR22_DIR = RNA_DIR / "bw_chr22"
EGENES_BED = RNA_DIR / "chr22_egenes.bed"
REF_FASTA = REPO / "tests" / "data" / "fasta" / "hg38.fa.bgz"

N_SAMPLES = 5
N_REGIONS = 300  # cap region count to keep the committed dataset small
# eGene BED rows are zero-width TSS points; expand each into a window so that
# gvl.write captures the variants/tracks the benchmarks reconstruct over. The
# window matches the read length used by the regression hot paths (with_len).
WINDOW = 16384
CHR22_LEN = 50_818_468


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def choose_samples() -> list[str]:
    """Deterministically pick N samples present in both genotypes and bigwigs."""
    psam = pl.read_csv(str(PLINK_PREFIX) + ".psam", separator="\t", infer_schema=False)
    # plink psam first column is "#IID" or "IID".
    iid_col = "#IID" if "#IID" in psam.columns else "IID"
    geno_samples = set(psam[iid_col].to_list())

    smap = pl.read_csv(SAMPLE_MAP)
    bw_samples = set(smap["sample"].to_list())

    overlap = sorted(geno_samples & bw_samples)
    if len(overlap) < N_SAMPLES:
        raise SystemExit(
            f"Only {len(overlap)} samples overlap genotypes+bigwigs; need {N_SAMPLES}."
        )
    chosen = overlap[:N_SAMPLES]
    (DATA / "samples.txt").write_text("\n".join(chosen) + "\n")
    print(f"Chosen samples: {chosen}")
    return chosen


def slice_pgen(samples: list[str], bed_path: Path) -> Path:
    """plink2 slice: chr22, chosen samples, variants within benchmark windows.

    Restricting to variants that overlap the benchmark regions keeps the
    committed genotypes/variants.arrow small (only ~tens of thousands of
    variants instead of the full ~1M-variant chr22 table).
    """
    keep = DATA / "_keep.txt"
    keep.write_text("#IID\n" + "\n".join(samples) + "\n")
    out_prefix = DATA / "chr22_5s"
    run(
        [
            "plink2",
            "--pgen",
            str(PLINK_PREFIX) + ".pgen",
            "--pvar",
            str(PLINK_PREFIX) + ".pvar.zst",
            "--psam",
            str(PLINK_PREFIX) + ".psam",
            "--chr",
            "chr22",
            "--keep",
            str(keep),
            # The widened chr22_egenes.bed is standard 0-based half-open BED;
            # plink2 reads its first 3 columns to restrict to overlapping vars.
            "--extract",
            "bed0",
            str(bed_path),
            "--make-pgen",
            "--out",
            str(out_prefix),
        ]
    )
    keep.unlink()
    return out_prefix.with_suffix(".pgen")


def copy_regions() -> Path:
    """Copy the chr22 egenes BED, capped to N_REGIONS rows."""
    bed = pl.read_csv(
        EGENES_BED,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )
    if bed.height > N_REGIONS:
        bed = bed.head(N_REGIONS)
    # Expand zero-width TSS points into centered WINDOW-bp regions, clipped to
    # the chromosome. Wider-than-zero rows are left untouched.
    half = WINDOW // 2
    center = (pl.col("start") + pl.col("end")) // 2
    bed = bed.with_columns(
        start=(center - half).clip(0, CHR22_LEN),
        end=(center + half).clip(0, CHR22_LEN),
    )
    out = DATA / "chr22_egenes.bed"
    bed.write_csv(out, include_header=False, separator="\t")
    print(f"Wrote {bed.height} regions to {out}")
    return out


def build_masked_reference(bed_path: Path) -> Path:
    """Extract chr22, mask everything outside benchmark regions to N, bgzip.

    A mostly-N chr22 compresses to a few MB while staying coordinate-correct
    over the benchmark windows used for haplotype reconstruction.
    """
    import numpy as np
    import pysam

    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )

    fa = pysam.FastaFile(str(REF_FASTA))
    chr22 = np.frombuffer(fa.fetch("chr22").encode("ascii"), dtype="S1").copy()
    masked = np.full_like(chr22, b"N")
    pad = 0  # regions already include flanks; no extra pad needed
    for start, end in bed.select("start", "end").iter_rows():
        s = max(0, int(start) - pad)
        e = min(chr22.size, int(end) + pad)
        masked[s:e] = chr22[s:e]

    out_plain = DATA / "chr22.masked.fa"
    with open(out_plain, "w") as f:
        f.write(">chr22\n")
        seq = masked.tobytes().decode("ascii")
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")

    out_bgz = DATA / "chr22.masked.fa.gz"
    if out_bgz.exists():
        out_bgz.unlink()
    run(["bgzip", str(out_plain)])  # -> chr22.masked.fa.gz
    run(["samtools", "faidx", str(out_bgz)])  # -> .fai + .gzi
    print(f"Wrote masked reference {out_bgz}")
    return out_bgz


def build_dataset(samples: list[str], pgen: Path, bed_path: Path) -> Path:
    import genvarloader as gvl
    from genoray import PGEN

    smap = pl.read_csv(SAMPLE_MAP)
    paths: dict[str, str] = {}
    for sample, full_path in smap.select("sample", "path").iter_rows():
        if sample not in samples:
            continue
        bw = BW_CHR22_DIR / Path(full_path).name
        if not bw.exists():
            raise SystemExit(f"Missing chr22 bigwig for {sample}: {bw}")
        paths[sample] = str(bw)
    assert set(paths) == set(samples), set(samples) - set(paths)

    tracks = gvl.BigWigs("read-depth", paths)

    out = DATA / "chr22_geuv.gvl"
    if out.exists():
        shutil.rmtree(out)
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=tracks,
        samples=samples,
        overwrite=True,
    )
    print(f"Wrote dataset {out}")
    return out


def main() -> None:
    if not REF_FASTA.exists():
        raise SystemExit(
            f"Reference {REF_FASTA} not found. Run `pixi run -e dev gen` first."
        )
    DATA.mkdir(parents=True, exist_ok=True)
    samples = choose_samples()
    # Build the widened regions BED first: slice_pgen now restricts the PGEN
    # to variants overlapping these benchmark windows.
    bed_path = copy_regions()
    pgen = slice_pgen(samples, bed_path)
    build_masked_reference(bed_path)
    build_dataset(samples, pgen, bed_path)
    print("Done.")


if __name__ == "__main__":
    main()
