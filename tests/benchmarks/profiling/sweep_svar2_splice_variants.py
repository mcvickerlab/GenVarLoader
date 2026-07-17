"""Sweep driver for the SVAR2 spliced-variant decode optimization loop.

Times the spliced ``with_seqs("variants")`` getitem across the two dominating
dimensions -- cohort size (``n_samples``) and transcripts/batch
(``n_transcripts``) -- using vcfixture-rs bulk cohorts
(``data/build_svar2_splice_bulk.py``). Emits median/min/spread to CSV.

This is the moving-target harness for the performant-py-rust loop; the committed
regression benchmark is ``tests/benchmarks/test_e2e_svar2_splice_variants.py``.

Thread control MUST be set before genvarloader is imported: this driver sets
``GVL_FORCE_PARALLEL=1`` (so every sweep point runs the *same* parallel code path,
not flipping at GVL's size gate) and ``GVL_NUM_THREADS=--threads``. GVL memoizes
the worker count at first parallel call, so run once per thread count
(``--threads 1`` ~serial curve, ``--threads $(nproc)`` parallel curve).

    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev python \
        tests/benchmarks/profiling/sweep_svar2_splice_variants.py \
        --work /scratch/svar2bench --threads 8

    # correctness-oracle mode (freeze current output, re-read, assert byte-identical):
    ... sweep_svar2_splice_variants.py --work /scratch/svar2bench --check
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow direct invocation (`python tests/benchmarks/profiling/sweep_...py`): put the
# repo root on sys.path so `tests.benchmarks.data...` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLICE_INFO = ("transcript_id", "exon_number")


@dataclass
class BenchRow:
    n_samples: int
    n_transcripts: int
    records: int
    threads: int
    decoded_alleles: int
    median_ms: float
    min_ms: float
    spread_ms: float


def _open(fixture):
    import genvarloader as gvl

    return (
        gvl.Dataset.open(fixture.gvl_path, reference=fixture.reference)
        .with_settings(splice_info=SPLICE_INFO, var_filter="exonic")
        .with_seqs("variants")
    )


def _freeze(out) -> dict[str, tuple]:
    """Freeze every RaggedVariants field's data/offsets/str_offsets."""
    import numpy as np

    frozen: dict[str, tuple] = {}
    for name in out.fields:
        f = out[name]
        rl = getattr(f, "_rl", None)
        so = getattr(rl, "str_offsets", None) if rl is not None else None
        frozen[name] = (
            np.asarray(f.data).copy(),
            np.asarray(f.offsets).copy(),
            None if so is None else np.asarray(so).copy(),
        )
    return frozen


def _assert_equal(frozen: dict[str, tuple], out) -> None:
    import numpy as np

    assert set(out.fields) == set(frozen), (
        f"field set changed: {set(out.fields)} vs {set(frozen)}"
    )
    for name, (data, off, so) in frozen.items():
        f = out[name]
        assert np.array_equal(np.asarray(f.data), data), f"{name}: data differ"
        assert np.array_equal(np.asarray(f.offsets), off), f"{name}: offsets differ"
        rl = getattr(f, "_rl", None)
        cur = getattr(rl, "str_offsets", None) if rl is not None else None
        if so is None:
            assert cur is None, f"{name}: gained str_offsets"
        else:
            assert cur is not None and np.array_equal(np.asarray(cur), so), (
                f"{name}: str_offsets differ"
            )


def _decoded(out) -> int:
    import numpy as np

    return int(np.asarray(out.alt.offsets)[-1])


def bench(fixture, *, threads: int, reps: int, warmup: int) -> BenchRow:
    ds = _open(fixture)
    idx = (slice(0, fixture.n_transcripts), slice(0, fixture.n_samples))
    decoded = 0
    for _ in range(warmup):
        decoded = _decoded(ds[idx])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ds[idx]
        ts.append((time.perf_counter() - t0) * 1e3)
        decoded = _decoded(out)
    return BenchRow(
        fixture.n_samples,
        fixture.n_transcripts,
        fixture.records,
        threads,
        decoded,
        st.median(ts),
        min(ts),
        max(ts) - min(ts),
    )


def run_sweep(*, work, samples, transcripts, records, threads, reps, warmup, out_csv):
    from tests.benchmarks.data.build_svar2_splice_bulk import build

    results: list[BenchRow] = []
    for s in samples:
        for t in transcripts:
            fx = build(work / "fixtures", n_samples=s, n_transcripts=t, records=records)
            row = bench(fx, threads=threads, reps=reps, warmup=warmup)
            results.append(row)
            print(
                f"S={s:>7} T={t:>4} thr={threads:>3}: {row.median_ms:9.2f} ms "
                f"(min {row.min_ms:.2f}, spread {row.spread_ms:.2f}, "
                f"{row.decoded_alleles} alleles)",
                flush=True,
            )
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(results[0])))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"wrote {out_csv}", flush=True)
    return results


def run_check(*, work, samples, transcripts, records):
    """Freeze the current output, re-read, and assert byte-identical (oracle)."""
    from tests.benchmarks.data.build_svar2_splice_bulk import build

    for s in samples:
        for t in transcripts:
            fx = build(work / "fixtures", n_samples=s, n_transcripts=t, records=records)
            ds = _open(fx)
            idx = (slice(0, t), slice(0, s))
            frozen = _freeze(ds[idx])
            _assert_equal(frozen, ds[idx])
            print(f"OK oracle S={s} T={t}: {_decoded(ds[idx])} alleles", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--records", type=int, default=20_000)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--samples", type=int, nargs="+", default=[1000, 5000, 25_000])
    p.add_argument("--transcripts", type=int, nargs="+", default=[64, 256])
    p.add_argument("--check", action="store_true", help="oracle mode, no timing")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    os.environ["GVL_FORCE_PARALLEL"] = "1"
    os.environ["GVL_NUM_THREADS"] = str(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)

    if args.check:
        run_check(
            work=args.work,
            samples=args.samples,
            transcripts=args.transcripts,
            records=args.records,
        )
        return
    out = args.out or (args.work / f"results_thr{args.threads}.csv")
    run_sweep(
        work=args.work,
        samples=args.samples,
        transcripts=args.transcripts,
        records=args.records,
        threads=args.threads,
        reps=args.reps,
        warmup=args.warmup,
        out_csv=out,
    )


if __name__ == "__main__":
    main()
