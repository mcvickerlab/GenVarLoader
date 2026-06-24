"""Time gvl.write() (and best-effort gvl.update()) on the 1kg chr21/chr22 corpus.

Usage:
    pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op write
    pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op update

Run under memray for peak RSS:
    pixi run -e dev memray-write

Prints: op=... corpus=1kg-chr21chr22 wall=<s>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure repo root is on sys.path so `tests` is importable when run standalone
# (pytest adds "." via pythonpath config, but pixi run doesn't).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ONE_KG_DIR = Path(__file__).resolve().parents[2] / "data" / "1kg"
FILTERED_BCF = ONE_KG_DIR / "filtered.bcf"
REGIONS_BED = ONE_KG_DIR / "regions.bed"

CORPUS_TAG = "1kg-chr21chr22"


def _check_inputs() -> None:
    missing = [p for p in (FILTERED_BCF, REGIONS_BED) if not p.exists()]
    if missing:
        raise SystemExit(
            f"1kg inputs missing: {[str(p) for p in missing]}\n"
            "Run `pixi run -e dev gen-1kg` first."
        )


def run_write(out: Path) -> float:
    """Write the 1kg VCF dataset; return wall-clock seconds."""
    import genvarloader as gvl
    from genoray import VCF

    vcf = VCF(FILTERED_BCF)
    if not vcf._valid_index():
        vcf._write_gvi_index()
    _ = vcf._load_index()

    t0 = time.perf_counter()
    gvl.write(path=out, bed=REGIONS_BED, variants=vcf, max_mem="4g")
    return time.perf_counter() - t0


def run_update(ds_path: Path) -> tuple[float, str]:
    """Add a synthetic annotation track to an existing dataset; return (wall, info).

    Uses a small polars DataFrame covering chr21+chr22 as an annotation track
    (sample-independent, so no sample-matching is needed).
    """
    import polars as pl

    import genvarloader as gvl

    # Build a tiny synthetic annotation: 50 bp intervals sprinkled across chr21/chr22.
    # gvl.update expects BED-style columns: chrom, chromStart (0-based), chromEnd, score.
    rows = []
    for chrom in ("chr21", "chr22"):
        for start in range(5_000_000, 20_000_000, 500_000):
            rows.append(
                {
                    "chrom": chrom,
                    "chromStart": start,
                    "chromEnd": start + 50,
                    "score": 1.0,
                }
            )
    annot_df = pl.DataFrame(rows)

    t0 = time.perf_counter()
    gvl.update(ds_path, annot_tracks={"syn_annot": annot_df}, max_mem="4g")
    wall = time.perf_counter() - t0
    return wall, f"annot_rows={len(rows)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["write", "update"], required=True)
    args = p.parse_args()

    _check_inputs()

    if args.op == "write":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "1kg_baseline.gvl"
            wall = run_write(out)
        print(f"op=write corpus={CORPUS_TAG} wall={wall:.3f}s")

    elif args.op == "update":
        # Write first so we have something to update.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "1kg_baseline.gvl"
            try:
                _write_wall = run_write(out)
            except Exception as exc:
                raise SystemExit(f"update baseline: write step failed: {exc}") from exc

            try:
                wall, info = run_update(out)
            except Exception as exc:
                print(
                    f"op=update corpus={CORPUS_TAG} wall=N/A "
                    f"blocked: update step failed: {exc}"
                )
                sys.exit(0)

        print(f"op=update corpus={CORPUS_TAG} wall={wall:.3f}s ({info})")


if __name__ == "__main__":
    main()
