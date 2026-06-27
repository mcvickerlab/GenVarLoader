"""Time gvl.write() and a real per-sample BigWigs gvl.update() on the chr22_geuv corpus.

Exercises the full Rust write path (genoray sparse genotypes + Rust bigWig
streaming writer). Prep (sample choice, plink2 slice) runs untimed; only the
gvl.write / gvl.update call is measured.

Usage (needs /carter sources or GVL_BENCH_SOURCE bundle):
    pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op write
    pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op update

Peak RSS:
    NUMBA_NUM_THREADS=1 .pixi/envs/dev/bin/memray run -o w.bin \\
        tests/benchmarks/profiling/profile_write_realistic.py --op write
    .pixi/envs/dev/bin/memray stats w.bin
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.benchmarks.data import build_realistic as br  # noqa: E402

CORPUS_TAG = "chr22_geuv"


def _resolve_bigwig_paths(samples: list[str]) -> dict[str, str]:
    """Resolve per-sample chr22 bigWig paths exactly as build_realistic.build_dataset."""
    smap = pl.read_csv(br.SAMPLE_MAP)
    paths: dict[str, str] = {}
    for sample, full_path in smap.select("sample", "path").iter_rows():
        if sample not in samples:
            continue
        bw = br.BW_CHR22_DIR / Path(full_path).name
        if not bw.exists():
            raise SystemExit(f"Missing chr22 bigwig for {sample}: {bw}")
        paths[sample] = str(bw)
    assert set(paths) == set(samples), set(samples) - set(paths)
    return paths


def _prep() -> tuple[list[str], Path, Path, dict[str, str]]:
    """Untimed prep: choose samples, build regions BED, slice + filter PGEN, resolve bigwigs."""
    samples = br.choose_samples()
    bed_path = br.copy_regions()
    pgen = br.slice_pgen(samples, bed_path)
    pgen = br.drop_unsupported_variants(pgen)
    paths = _resolve_bigwig_paths(samples)
    return samples, pgen, bed_path, paths


def run_write(out: Path) -> float:
    import genvarloader as gvl
    from genoray import PGEN

    samples, pgen, bed_path, paths = _prep()
    tracks = gvl.BigWigs("read-depth", paths)
    t0 = time.perf_counter()
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=tracks,
        samples=samples,
        overwrite=True,
        extend_to_length=False,
    )
    return time.perf_counter() - t0


def run_update(out: Path) -> tuple[float, str]:
    import genvarloader as gvl
    from genoray import PGEN

    samples, pgen, bed_path, paths = _prep()
    # Build a base dataset (untimed) to update.
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=gvl.BigWigs("read-depth", paths),
        samples=samples,
        overwrite=True,
        extend_to_length=False,
    )
    # Timed: add a SECOND per-sample BigWigs track via update (Rust bigWig writer).
    add = gvl.BigWigs("read-depth-2", paths)
    t0 = time.perf_counter()
    gvl.update(out, tracks=add, max_mem="4g")
    wall = time.perf_counter() - t0
    return wall, f"track=read-depth-2 samples={len(samples)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["write", "update"], required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
        out = Path(tmp) / "chr22_geuv_bench.gvl"
        if args.op == "write":
            wall = run_write(out)
            print(f"op=write corpus={CORPUS_TAG} wall={wall:.3f}s")
        else:
            wall, info = run_update(out)
            print(f"op=update corpus={CORPUS_TAG} wall={wall:.3f}s ({info})")


if __name__ == "__main__":
    main()
