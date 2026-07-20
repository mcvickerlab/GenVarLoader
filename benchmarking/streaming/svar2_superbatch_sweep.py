"""Sweep `_super_batch_rows` on a vcfixture cohort-scale SVAR2 store to size
`SUPERBATCH_TARGET_ROWS` (`python/genvarloader/_dataset/_streaming.py`).

The super-batch is the rayon dispatch grain of the Phase-2 SVAR2 streaming reconstruct
(`_Svar2Backend._fill_super_batch` -> `svar2_reconstruct_super_batch`). This harness
times the reconstruct kernel *in isolation* (only the `_fill_super_batch` calls, not
`read_window`/gather/drain), forced serial vs forced parallel, so the reconstruct's
scaling is not swamped by the serial/GIL-bound read overhead that dominates an
end-to-end `to_iter` timing. Perf is secondary color on a shared node (see the
roadmap's perf-gate convention); read `speedup` (par vs serial) and `par cpu/wall`
(cores engaged) together.

Measured 2026-07-19 (2000 samples x 20000 records, 64x1000bp, 8 cores, best-of-3):
    sb_rows   serial_s      par_s   par_cpu/wall   speedup
        256      0.194      0.212         2.36      0.91   # parallel slower (fork/join)
       1024      0.157      0.142         2.01      1.11
       4096      0.208      0.181         1.54      1.15
      16384      0.278      0.212         1.51      1.31   # best speedup
      65536      0.329      0.287         1.32      1.15
The kernel is memory-bandwidth-bound: only ~1.1-1.3x on 8 cores, and big super-batches
hurt cache locality (serial time rises). Default 4096 kept -- at the `should_parallelize`
byte-gate boundary and near the wall optimum. End-to-end wall is dominated by the serial
read, so PR 3 (relaxed-order read<->reconstruct overlap) is the real throughput lever.
See docs/roadmaps/streaming-dataset.md (SVAR2 Phase 2 PR 2 row).

Run:
    VCFIXTURE_BIN=/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture \
        pixi run -e dev python benchmarking/streaming/svar2_superbatch_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import genvarloader as gvl
from genvarloader._threads import should_parallelize
from genvarloader.genvarloader import Svar2ReconBuf

# `tests/` is not an installed package; put the repo root on the path so the bulk
# cohort builder imports (same pattern as the sibling streaming benchmarks).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the bulk cohort/bed/reference builders (NOT `build`, which also runs the heavy
# gvl.write oracle the sweep doesn't need).
from tests.benchmarks.data.build_svar2_stream_bulk import (  # noqa: E402
    _write_reference,
    gen_cohort,
    make_contiguous_bed,
)

SUPERBATCH_GRID = [256, 1024, 4096, 16384, 65536]
REPEATS = 3

# Cohort scale: n_samples is the dominating axis (toward AoU scale). 64 regions x
# 1000bp matches the streaming default window shape.
N_SAMPLES = 2000
N_RECORDS = 20_000
N_REGIONS = 64
REGION_LEN = 1000
CONTIG = "chr1"


def build_store(cache: Path):
    """(bed, reference_path, svar2_path) for a cohort-scale SVAR2 store. Idempotent."""
    from genoray import SparseVar2

    root = cache / f"s{N_SAMPLES}_r{N_RECORDS}_{N_REGIONS}x{REGION_LEN}"
    root.mkdir(parents=True, exist_ok=True)
    ref = root / "ref.fa"
    svar2 = root / "store.svar2"

    bcf, span = gen_cohort(cache / "cohorts", N_SAMPLES, N_RECORDS, contig=CONTIG)
    bed = make_contiguous_bed(CONTIG, span, N_REGIONS, REGION_LEN)
    if not ref.exists():
        _write_reference(ref, CONTIG, span + REGION_LEN)
    if not (svar2 / "meta.json").exists():
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
    return bed, ref, svar2


def _reconstruct_only(
    sds, sb_rows: int, *, parallel: bool | None
) -> tuple[float, float]:
    """Drive every window's super-batch fills (the reconstruct kernel), timing ONLY the
    `_fill_super_batch` calls -- not `read_window`, gather, or drain. This isolates the
    reconstruct from the serial/GIL-bound read overhead that swamps an end-to-end
    `to_iter` timing. `parallel`: None = production gating (`should_parallelize`), else
    force serial/parallel to measure the reconstruct's raw scaling.

    Returns (fill_wall_s, fill_cpu_s) summed over all windows.
    """
    backend = sds._backend
    buf = Svar2ReconBuf(backend.ploidy)
    fill_wall, fill_cpu = 0.0, 0.0
    for r_idx, s_idx in sds._plan():
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)
        for sb_lo in range(0, n_rows, sb_rows):
            sb_hi = min(sb_lo + sb_rows, n_rows)
            par = (
                should_parallelize(backend._est_out_bytes(r_idx, sb_hi - sb_lo))
                if parallel is None
                else parallel
            )
            w0, c0 = time.perf_counter(), time.process_time()
            backend._fill_super_batch(r_idx, s_idx, window, sb_lo, sb_hi, buf, par)
            fill_wall += time.perf_counter() - w0
            fill_cpu += time.process_time() - c0
    return fill_wall, fill_cpu


def bench(sds, sb_rows: int) -> dict[str, float]:
    """Best-of-REPEATS reconstruct-only wall for serial vs parallel fills at this sb
    size, plus the cpu/wall of the parallel run (core engagement)."""
    sds._backend._super_batch_rows = sb_rows
    best_serial = float("inf")
    best_par_wall, best_par_cpu = float("inf"), 0.0
    for _ in range(REPEATS):
        sw, _sc = _reconstruct_only(sds, sb_rows, parallel=False)
        best_serial = min(best_serial, sw)
        pw, pc = _reconstruct_only(sds, sb_rows, parallel=True)
        if pw < best_par_wall:
            best_par_wall, best_par_cpu = pw, pc
    return {
        "serial_wall": best_serial,
        "par_wall": best_par_wall,
        "par_cpu_over_wall": best_par_cpu / best_par_wall if best_par_wall > 0 else 0.0,
        "speedup": best_serial / best_par_wall if best_par_wall > 0 else 0.0,
    }


def main() -> None:
    cache = Path(__file__).resolve().parent / ".svar2_superbatch_sweep.cache"
    bed, ref, svar2 = build_store(cache)
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs(
        "haplotypes"
    )

    for _batch in sds.to_iter(batch_size=32):  # warm import/JIT/FASTA-cache off clock
        break

    import os

    print(
        f"# SVAR2 super-batch sweep: {N_SAMPLES} samples x {N_RECORDS} records, "
        f"{N_REGIONS}x{REGION_LEN}bp, best-of-{REPEATS}, "
        f"{len(os.sched_getaffinity(0))} cores"
    )
    print("# reconstruct-only timing (excludes read_window/gather/drain)")
    print(
        f"{'sb_rows':>10} {'serial_s':>10} {'par_s':>10} "
        f"{'par_cpu/wall':>13} {'speedup':>9}"
    )
    for sb in SUPERBATCH_GRID:
        r = bench(sds, sb)
        print(
            f"{sb:>10} {r['serial_wall']:>10.3f} {r['par_wall']:>10.3f} "
            f"{r['par_cpu_over_wall']:>13.2f} {r['speedup']:>9.2f}"
        )


if __name__ == "__main__":
    main()
