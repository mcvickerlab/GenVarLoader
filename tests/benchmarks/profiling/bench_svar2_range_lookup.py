"""Benchmark gate for the sparse svar2 range-cache probe (#357).

Synthetic: needs no All of Us access and runs in ~8 s end-to-end (measured on
the machine this was authored on; the spec's "~30 s" estimate did not hold).
Dominated by `build()`'s N = 10**8 row alone, ~5 s of the ~6 s spent inside
`main()`, from `rng.choice` + `np.sort` + `np.concatenate` over ~2.4 GB of
`cell_vk` plus its int64 concatenate temporaries -- not by the lookups
themselves. Run it directly, not under pytest:

    pixi run -e dev python tests/benchmarks/profiling/bench_svar2_range_lookup.py

THE GATE: sparse lookup <= 2% of batch wall on the GLOBALLY SHUFFLED
distribution, against `BATCH_MS = 171.0` -- the single-threaded spliced batch
recorded in `Svar2Haps._readbound_gather`'s docstring, exactly as the spec
states it. `BATCH_MS_8T = 88.0` (the same batch on 8 threads) is reported
alongside as a note, not a second gate -- see "Two figures, one gate" below.

The shuffled row is the training case, not a pathological one:
`to_dataloader(shuffle=True)` (_impl.py:1825) shuffles the flat (R*S) index.
The clustered row is what `ds[:, :]` produces.

The baseline is a RESIDENT dense `_DenseRanges` at a size that fits: two int64
`(1000, 2000, 2, 2)` arrays, 64 MB apiece, 128 MB total -- the real shipped
dense layout (32 B/cell: two `(R, S, P, 2)` int64 arrays, not one), timed
through `_DenseRanges.lookup` so the baseline is the same two-gather work the
sparse probe replaces. Not the 6 KB fixture, and not the 128 GB array the real
comparison would need, which nobody can run. It is also the honest trade: at
the realized chr22 operating point (see below) sparse measures ~5.5x a
resident dense gather -- but the dense array cannot *be* resident at cohort
scale (128 GB, >99% empty, #357), which is the entire point of the change. A
slower probe against an alternative that cannot exist is not grounds to
reject the design.

If the two-level probe misses the gate, do NOT take the spec's flat-int64-key
fallback. It was written before the depth was measured and it makes the probe
slower, not faster: a flat key searches the whole table, raising the depth
`__post_init__` would compute -- `(width - 1).bit_length()` -- from 13 at the
widest per-region block (N/R ~= 5054 at All of Us chr22) to 25 over the whole
1.8e7-entry table, on the term measured at 84-87% of lookup cost (see
"Mechanism" below). Its only real effect is +4 bytes per entry.

The actual fallback is to fuse the search loop, which is where the time is: the
probe is ~93% numpy-pass-bound (measured -- the gather itself is the minority),
so a single-pass kernel over (region_ptr, cell_id) removes the per-iteration
temporaries the vectorized form cannot. That is a Rust #[pyfunction], deferred to
a follow-up issue (Task 10, Step 9) precisely because it is only worth doing if
this gate fails: at the measured numbers the whole probe is 0.5% of batch wall,
so a 3-5x relative win is worth ~0.35% of wall.

Estimator: `timeit` returns `(min_ms, max_ms)` over several trials, not the
mean of one. Scheduling noise on a shared machine only ever adds time, so the
minimum across trials is the least-biased estimate of true cost -- the same
principle `timeit.Timer.repeat`/`min` and `pyperf` use -- and the gate binds on
it. A mean-of-one-trial estimator made the worst-shuffled figure swing between
0.77% and 2.37% run to run and flip the verdict; min-over-trials reduces but
does NOT eliminate this (a contaminated run can still land above 2% -- see
"Mechanism" below), so every row also prints its own `max_ms` and `max/min`
instead of hiding the spread behind a scalar. A benchmark that lives in a repo
gets re-run on unknown, possibly busy machines, and a silent mean is exactly
what let one contaminated run become a verdict.

Mechanism: the instability is in the search loop, not the multi-GB memory
access, and it does not respond to warming the arrays. Splitting one lookup
into its parts (measured on `N = 1e8`, 16384 scattered queries): gathering
`cell_vk` (2.4 GB) costs 0.26-0.28 ms with a 1.02-1.09x spread and is 13-16% of
the full lookup, i.e. the search loop is the other 84-87% -- the same order as
the brief's "search is 91% of cost" and consistent with its conclusion, though
not the same number; gathering `cell_id` (400 MB) costs 0.03 ms at a 1.01-1.04x
spread; the full lookup costs 1.68-2.03 ms at a 1.12-1.71x spread. The search
loop runs over a 16384-element query block, is L2-resident, and is dominated
by per-call numpy/Python overhead -- exactly what CPU contention and
P-core/E-core scheduling perturb, and exactly what a multi-GB gather (bound by
RAM bandwidth, not overhead) is immune to. A sequential touch of both arrays
before timing does not narrow the spread (warm min/max 1.06x-3.18x vs cold
1.23x-2.25x, overlapping ranges), which rules out page residency as the cause:
the pages were already resident (the warm touch pass ran at ~20 GB/s, i.e. RAM
speed, on every trial). A row whose `max/min` is much above ~1.3x means the
machine was busy during that trial block, not that the probe itself is
unstable -- the gate wants a quiet machine, and this script cannot tell the
difference between "the probe is slow" and "the machine was busy" on its own;
that is why it prints the ratio instead of asserting one or the other.

Two figures, one gate: `BATCH_MS` (171 ms, single-threaded) is what the spec's
criterion names, so PASS/FAIL binds on it alone and nothing else can turn the
script's exit into FAIL. `BATCH_MS_8T` (88 ms, 8 threads) is printed as a note
beneath the verdict, never a second verdict, because the probe is serial
Python/numpy and does not speed up with threads while the batch does -- so its
share of the batch *rises* as thread count goes up. Both are recorded figures
from another machine, so read them as a break-even instead of trusting either
percentage outright: `worst_ms / GATE` is the batch wall below which the gate
fails, independent of which denominator is in fashion. A batch that is
actually 60 ms here turns a 1.9% PASS into a 5.4% FAIL -- the break-even number
says the same thing without needing a specific batch figure to compare against.

Operating point: the sweep's max -- and hence the gate -- comes from the
N = 1e8 row, which is 5.5x past what All of Us chr22 actually realizes: 504 MB
/ 28 B per entry = 1.8e7 entries, i.e. N ~= 1.8e7 at 0.45% fill. The N = 1e7
row is the nearest realized-scale point and confirms the spec's own "~0.5% of
batch wall" claim to the digit. N = 1e8 stays in the sweep as a headroom/stress
row nobody's dataset will actually reach -- the gate is a max over the whole
sweep, not the realized point alone, and dropping it to look better would be
weakening the gate to pass it.
"""

from __future__ import annotations

import time

import numpy as np

from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _DenseRanges, _SparseRanges

BATCH_MS = 171.0
"""Single-threaded spliced 8192-cell batch wall, from _readbound_gather's docstring.

This is the figure the spec's gate criterion names, so PASS/FAIL binds on this
alone (see BATCH_MS_8T and the module docstring's "Two figures, one gate"
paragraph for why the 8-thread figure is reported but never gates). It is a
recorded figure from another machine, so the percentage against it is only as
good as it is -- a batch that is actually 60 ms here turns a 1.9% PASS into a
5.4% FAIL. The break-even line this script prints (`worst_ms / GATE`) says the
same thing without depending on this specific number. The spec's companion
figure of 18.4 ms for a shuffled 8192-cell probe does NOT reproduce (measured
~13x lower); the table this script prints supersedes it.
"""
BATCH_MS_8T = 88.0
"""Same spliced 8192-cell batch, on 8 threads, from _readbound_gather's docstring
(_svar2_haps.py:1528-1529). Reported as an informational note beneath the
verdict, never a second gate: the probe is serial Python/numpy and does not
speed up with threads while the batch does, so the probe's percentage of the
batch rises as thread count goes up. Tightening the gate to bind on this would
double its strictness on this script's own authority, which the spec's stated
criterion does not do.
"""
GATE = 0.02


def build(R: int, S: int, P: int, N: int, seed: int = 0) -> _SparseRanges:
    """A CSR table with N entries spread over R regions."""
    rng = np.random.default_rng(seed)
    span = S * P
    per = np.bincount(rng.integers(0, R, N), minlength=R)
    per = np.minimum(per, span)
    cell = np.concatenate(
        [np.sort(rng.choice(span, size=c, replace=False)) for c in per]
    ).astype(np.int32)
    ent = np.zeros(len(cell), ENTRY_DTYPE)
    ent["snp_len"] = 1
    ptr = np.concatenate([[0], per.cumsum()]).astype(np.int64)
    return _SparseRanges(ptr, cell, ent, R, S, P)


def probes(R: int, S: int, n_q: int, how: str, seed: int = 1):
    rng = np.random.default_rng(seed)
    if how == "clustered":
        # What ds[:, :] produces: few regions, many sorted slots.
        n_r = max(1, n_q // 512)
        r_q = np.repeat(rng.integers(0, R, n_r), n_q // n_r)[:n_q]
        si_q = np.tile(np.sort(rng.choice(S, n_q // n_r, replace=False)), n_r)[:n_q]
    else:
        # What to_dataloader(shuffle=True) produces.
        r_q = rng.integers(0, R, n_q)
        si_q = rng.integers(0, S, n_q)
    return r_q.astype(np.int64), si_q.astype(np.int64)


def timeit(fn, reps: int = 20, trials: int = 7) -> tuple[float, float]:
    """Return (min_ms, max_ms) per-call over several trials, not the mean of one.

    Scheduling noise on a shared machine only ever adds time, so the minimum
    across trials is the least-biased estimate of true cost and is what the
    gate binds on; the maximum is reported alongside so a caller can see how
    much the machine perturbed this run instead of trusting a single scalar
    (see the module docstring's "Estimator" and "Mechanism" paragraphs).
    """
    fn()  # warm
    times = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        times.append((time.perf_counter() - t) / reps * 1000)
    return min(times), max(times)


def main():
    R, S, P = 3734, 535662, 2  # All of Us chr22
    print(
        f"{'N':>10} {'n_q':>6} {'dist':>10} {'min_ms':>8} {'max_ms':>8}"
        f" {'max/min':>8} {'% batch':>8}"
    )
    worst_ms = 0.0
    # N=1e7, n_q=8192 is the row nearest the realized chr22 operating point
    # (1.8e7 entries at 504 MB / 28 B per entry) -- see the module docstring's
    # "Operating point" paragraph.
    op_point_ms: float | None = None
    for N in (10**5, 10**6, 10**7, 10**8):
        table = build(R, S, P, N)
        for n_q in (512, 2048, 8192):
            for how in ("clustered", "shuffled"):
                r_q, si_q = probes(R, S, n_q, how)
                ms, ms_max = timeit(lambda: table.lookup(r_q, si_q, P))
                spread = ms_max / ms if ms > 0 else float("inf")
                pct = ms / BATCH_MS * 100
                if how == "shuffled":
                    worst_ms = max(worst_ms, ms)
                    if N == 10**7 and n_q == 8192:
                        op_point_ms = ms
                print(
                    f"{N:>10} {n_q:>6} {how:>10} {ms:>8.2f} {ms_max:>8.2f}"
                    f" {spread:>7.2f}x {pct:>7.2f}%"
                )

    # Baseline: the shipped dense layout at a size that actually fits -- two
    # int64 (R, S, P, 2) arrays, 64 MB each, 128 MB total, timed through
    # _DenseRanges.lookup (two gathers), not one gather over a single array.
    dense = _DenseRanges(
        np.zeros((1000, 2000, 2, 2), np.int64),
        np.zeros((1000, 2000, 2, 2), np.int64),
        1000,
        2000,
        2,
    )
    rng = np.random.default_rng(2)
    r_q = rng.integers(0, 1000, 8192)
    si_q = rng.integers(0, 2000, 8192)
    dense_ms, dense_ms_max = timeit(lambda: dense.lookup(r_q, si_q, 2))
    dense_spread = dense_ms_max / dense_ms if dense_ms > 0 else float("inf")
    print(
        f"\nresident dense baseline (128 MB across two int64 (R,S,P,2) arrays,"
        f" 8192 cells, via _DenseRanges.lookup): min {dense_ms:.2f} ms, max"
        f" {dense_ms_max:.2f} ms ({dense_spread:.2f}x)"
    )
    if op_point_ms is not None and dense_ms > 0:
        ratio = op_point_ms / dense_ms
        print(
            f"honest trade at the realized chr22 operating point (N=1e7, 8192"
            f" cells, shuffled): sparse {op_point_ms:.2f} ms vs resident dense"
            f" {dense_ms:.2f} ms ({ratio:.1f}x) -- dense cannot be resident at"
            f" real scale (128 GB, >99% empty), which is the point of the change"
        )

    # THE GATE binds on BATCH_MS (171 ms, single-threaded) alone, per the
    # spec's stated criterion -- nothing below can turn the exit into FAIL
    # except pct_1t.
    pct_1t = worst_ms / BATCH_MS * 100
    pct_8t = worst_ms / BATCH_MS_8T * 100
    breakeven_ms = worst_ms / GATE
    print(
        f"\nworst shuffled: {worst_ms:.2f} ms = {pct_1t:.2f}% of a {BATCH_MS:.0f} ms"
        f" batch (gate: {GATE:.0%})"
    )
    print(
        f"break-even batch wall: {breakeven_ms:.1f} ms"
        " (below this, the gate fails, regardless of which recorded batch figure is used)"
    )
    print(
        "PASS" if pct_1t <= GATE * 100 else "FAIL -- take the spec's flat-key fallback"
    )

    # Note only -- never a second gate. The probe is serial and does not speed
    # up with threads while the batch does, so its share rises at 8 threads.
    print(
        f"note (informational, does not gate): at {BATCH_MS_8T:.0f} ms (8"
        f" threads, the real training config) the same worst cell is"
        f" {pct_8t:.2f}% of batch, vs {pct_1t:.2f}% at {BATCH_MS:.0f} ms"
        " single-threaded -- the probe's share rises as thread count goes up"
        " because it does not itself parallelize."
    )
    if pct_8t > GATE * 100:
        print(
            "the 8-thread figure already exceeds the single-threaded gate's"
            " 2% threshold -- this belongs in the fused-kernel follow-up issue"
            " (tests/benchmarks/profiling/), not this script's verdict."
        )


if __name__ == "__main__":
    main()
