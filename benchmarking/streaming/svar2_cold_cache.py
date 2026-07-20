"""SVAR2 streaming: cold-cache A/B ("sync" vs "svar2_engine") + the original
synchronous-path IO-vs-CPU-bound split.

MEASURES ONLY -- perf is secondary color here, never a pass/fail gate (see the
project's perf-gate convention, CLAUDE.md / docs/roadmaps/streaming-dataset.md).
Uses the vcfixture-rs bulk builder for cohort-scale stores; skips cleanly if
vcfixture/bcftools/samtools are absent. Reports best-of-N on a shared, noisy node --
treat the reported ratio/ranges as color, not a verdict on their own; the SHIP
decision rule below is what turns a measurement into a decision.

**Ship rule (PR-3 Task 4, mirrors the SVAR1 Task-9 A-vs-C bar):** the "svar2_engine"
strategy ships as the SVAR2 default ONLY IF its best-of-N wall-clock range is
ENTIRELY BELOW "sync"'s (non-overlapping ranges) -- i.e. clearly outside this node's
run-to-run noise, not merely a smaller minimum or a favorable-looking average. If the
ranges overlap, or a clean best-of-N can't be obtained, "sync" stays the default and
the reason is recorded in the roadmap; a marginal or noisy "win" does not count.

The default action (no flags) is the A/B sweep. `--legacy-iosplit` reruns the
original single-strategy ("sync" only) cold-cache timing + a `pyinstrument`
find_ranges-vs-gather/kernel breakdown for one rep, kept for IO/CPU-bound
investigation independent of the A/B decision.

Run:
    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev \
        python benchmarking/streaming/svar2_cold_cache.py --samples 500 2000 --records 20000
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import genvarloader as gvl

_STRATEGIES = ("sync", "svar2_engine")


def _evict_page_cache(root: Path) -> None:
    """Evict `root`'s regular files from the kernel page cache via
    ``posix_fadvise(..., POSIX_FADV_DONTNEED)``, so a "cold" timed pass doesn't ride
    the store build's own write-back cache. Combined with building under a fresh
    ``tempfile.mkdtemp()`` per timed run (a never-faulted inode -- the same
    no-root-needed convention `cold_cache_overlap.py` uses), this is belt-and-
    suspenders cold: fresh inode AND explicitly evicted. Best-effort: unreadable or
    special files are skipped silently -- this is a benchmark aid, not a correctness
    gate, and a failed eviction only biases a run warm, which the ship rule's
    non-overlapping-ranges bar already guards against.
    """
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        try:
            fd = os.open(str(f), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)


def _time_strategy(fx, batch_size: int, strategy: str) -> float:
    """Time one full `list(sds.to_iter(...))` sweep under `strategy`. Construction
    (reading the store's static variant table) happens OUTSIDE the timed region --
    only the sweep itself (window read + fill + drain) is timed, matching what a
    training loop actually pays per epoch."""
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    object.__setattr__(sds, "_prefetch_strategy", strategy)

    t0 = time.perf_counter()
    n_rows = 0
    for _data, r_idx, _s_idx in sds.to_iter(batch_size=batch_size):
        n_rows += len(r_idx)
    elapsed = time.perf_counter() - t0
    assert n_rows == sds.shape[0] * sds.shape[1], (
        f"sweep yielded {n_rows} rows, expected {sds.shape[0] * sds.shape[1]} "
        "-- harness is not exercising the full dataset"
    )
    return elapsed


def _fmt_range(vals: list[float]) -> str:
    return f"[{min(vals):.3f}, {max(vals):.3f}]"


def _ab_sweep(
    samples_list: list[int], records: int, repeats: int, batch_size: int
) -> None:
    """Cold-cache A/B: "sync" (current SVAR2 default) vs "svar2_engine" (PR-3's
    producer-thread pipeline engine). Per rep, per `n_samples`, EACH strategy gets
    its OWN freshly-built store (own `tempfile.mkdtemp()` inode, own
    `posix_fadvise(DONTNEED)` eviction pass) -- mirroring `cold_cache_overlap.py`'s
    per-(rep, strategy) fresh-build discipline exactly, so neither strategy can ride
    the other's warm page cache within a rep."""
    import sys

    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from tests.benchmarks.data.build_svar2_stream_bulk import build

    # Fixed per-strategy seed offset (same rationale as cold_cache_overlap.py):
    # PYTHONHASHSEED randomizes hash(strategy) per process, so a stable offset keeps
    # each strategy's per-rep store deterministic while still differing between
    # strategies (never comparing the exact same genotype draw across strategies,
    # which would let a lucky/unlucky draw masquerade as a strategy effect).
    strategy_offset = {s: i * 500 for i, s in enumerate(_STRATEGIES)}

    for n in samples_list:
        timings: dict[str, list[float]] = {s: [] for s in _STRATEGIES}
        for rep in range(repeats):
            for strategy in _STRATEGIES:
                tmp = Path(tempfile.mkdtemp(prefix="gvl_svar2_ab_"))
                try:
                    seed = 3000 + rep * 1000 + strategy_offset[strategy]
                    fx = build(tmp, n_samples=n, records=records, seed=seed)
                    _evict_page_cache(tmp)
                    elapsed = _time_strategy(fx, batch_size, strategy)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                timings[strategy].append(elapsed)
                print(
                    f"n_samples={n:>6} rep={rep + 1}/{repeats} "
                    f"{strategy:>12s}: {elapsed:.3f}s"
                )

        print(f"\n--- n_samples={n} summary (batch_size={batch_size}) ---")
        for strategy in _STRATEGIES:
            vals = timings[strategy]
            runs_str = " ".join(f"{v:.3f}" for v in vals)
            print(
                f"  {strategy:>12s}: runs=[{runs_str}]  "
                f"best={min(vals):.3f}s  range={_fmt_range(vals)}"
            )

        sync_vals = timings["sync"]
        eng_vals = timings["svar2_engine"]
        sync_range = (min(sync_vals), max(sync_vals))
        eng_range = (min(eng_vals), max(eng_vals))
        non_overlapping_below = eng_range[1] < sync_range[0]
        print(
            f"  ship rule: svar2_engine range {_fmt_range(eng_vals)} entirely below "
            f"sync range {_fmt_range(sync_vals)}? {non_overlapping_below}"
        )
        print(
            "  -> "
            + (
                "svar2_engine clears the non-overlapping-ranges bar at this n_samples."
                if non_overlapping_below
                else "ranges overlap (or engine is not faster) -- keep 'sync' as the "
                "default at this n_samples per the ship rule."
            )
        )


def _legacy_iosplit(samples_list: list[int], records: int, repeats: int) -> None:
    """Original single-strategy ("sync") cold-cache timing + one pyinstrument
    find_ranges-vs-gather/kernel breakdown per `n_samples`. Kept for IO/CPU-bound
    investigation independent of the A/B ship decision above."""
    import sys

    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from tests.benchmarks.data.build_svar2_stream_bulk import build
    import pyinstrument

    for n in samples_list:
        best = float("inf")
        tmp = None
        try:
            for rep in range(repeats):
                tmp = Path(tempfile.mkdtemp())  # fresh inode -> never-faulted
                fx = build(tmp, n_samples=n, records=records, seed=1000 + rep)
                sds = gvl.StreamingDataset(
                    fx.bed, reference=fx.reference, variants=fx.svar2_path
                ).with_seqs("haplotypes")
                t0 = time.perf_counter()
                for _ in sds.to_iter(batch_size=32):
                    pass
                best = min(best, time.perf_counter() - t0)
                # Keep the LAST rep's store alive for the pyinstrument sweep below;
                # earlier reps' stores are done being used, so clean them up now.
                if rep < repeats - 1:
                    shutil.rmtree(tmp, ignore_errors=True)
            print(f"n_samples={n:>6}: synchronous best-of-{repeats} = {best:.3f}s")

            # IO-vs-CPU split on the LAST rep's store, one sweep under pyinstrument.
            prof = pyinstrument.Profiler()
            prof.start()
            for _ in sds.to_iter(batch_size=32):
                pass
            prof.stop()
            print(prof.output_text(unicode=True, color=False))
            print(
                "  -> read `_find_ranges` (search) vs gather+kernel (fill) share above."
            )
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--records", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--legacy-iosplit",
        action="store_true",
        help="run the original single-strategy ('sync') cold-cache timing + "
        "pyinstrument IO-vs-CPU breakdown instead of the sync-vs-svar2_engine A/B",
    )
    args = ap.parse_args()
    try:
        if args.legacy_iosplit:
            _legacy_iosplit(args.samples, args.records, args.repeats)
        else:
            _ab_sweep(args.samples, args.records, args.repeats, args.batch_size)
    except FileNotFoundError as e:
        print(f"SKIP: {e}")


if __name__ == "__main__":
    main()
