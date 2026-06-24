"""Profiling driver for the haplotype, track, and variant hot paths.

Run single-threaded under py-spy or memray via the pixi tasks, e.g.:

    pixi run -e dev profile-tracks
    pixi run -e dev memray-haps

Modes:
  haplotypes  with_seqs("haplotypes")               (heavily-used path)
  tracks      with_seqs(None).with_tracks(...)       (REGRESSIONS.md target)
  variants    with_seqs("variants")                  (RaggedVariants assembly)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Force single numba thread BEFORE importing numba-backed code.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

DATA = Path(__file__).resolve().parents[1] / "data"
DS_PATH = DATA / "chr22_geuv.gvl"
REF_PATH = DATA / "chr22.masked.fa.gz"
SEQLEN = 16384
BATCH = 32
N_BATCHES = 2000
BURN_IN = 5


def build(ds, mode: str):
    if mode == "haplotypes":
        return ds.with_seqs("haplotypes").with_len(SEQLEN)
    if mode == "tracks":
        return ds.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    if mode == "variants":
        return ds.with_seqs("variants").with_len(SEQLEN)
    raise SystemExit(f"unknown mode {mode!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["haplotypes", "tracks", "variants"], required=True
    )
    p.add_argument("--n-batches", type=int, default=N_BATCHES)
    args = p.parse_args()

    if not DS_PATH.exists():
        raise SystemExit(
            f"Dataset {DS_PATH} not built. Run "
            "`pixi run -e dev python tests/benchmarks/data/build_realistic.py`."
        )

    import genvarloader as gvl

    ds = build(gvl.Dataset.open(DS_PATH, REF_PATH), args.mode)
    n_regions, n_samples = ds.shape[0], ds.shape[1]
    # Mirrors tests.benchmarks._indices.batch_indices (inlined to keep this a
    # standalone script that py-spy/memray can run without repo-root on sys.path).
    n = min(BATCH, n_regions * n_samples)
    regions = [i % n_regions for i in range(n)]
    samples = [i % n_samples for i in range(n)]

    print(
        f"mode={args.mode} threads={os.environ['NUMBA_NUM_THREADS']} "
        f"batches={args.n_batches} batch={n}"
    )
    # Burn-in (numba JIT warm-up, cache priming) is excluded from the timing.
    for _ in range(BURN_IN):
        _ = ds[regions, samples]
    t0 = time.perf_counter()
    for _ in range(args.n_batches):
        _ = ds[regions, samples]
    wall = time.perf_counter() - t0
    batches_per_s = args.n_batches / wall
    print(
        f"done wall={wall:.3f}s "
        f"throughput={batches_per_s:.1f} batch/s "
        f"({wall / args.n_batches * 1e3:.3f} ms/batch, "
        f"{batches_per_s * n:.0f} item/s)"
    )


if __name__ == "__main__":
    main()
