"""Bench count_intervals equivalents using pyranges1 (py>=3.12).

Methods:
  xprod      : cross-product queries (composite Chromosome key, current impl)
  join_groupby: n_regions queries only; sample_id flows from table join side

Run:
    pixi run -e py312 python experiments/bench_table_overlap/bench_pyranges1.py
"""

from __future__ import annotations

import csv
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

assert sys.version_info >= (3, 12), "pyranges1 requires py>=3.12"
import pyranges1 as pr

sys.path.insert(0, str(Path(__file__).parent))
from _common import CASES, N_TRIALS, gen_queries, gen_table  # noqa: E402

OUT_CSV = Path(__file__).parent / "results_pyranges1.csv"


# ── methods ──────────────────────────────────────────────────────────────────

def count_xprod(table_df, samples, starts, ends, contig="chr1"):
    n_regions, n_samples = len(starts), len(samples)
    contig_subset = table_df.filter(pl.col("chrom") == contig)
    if contig_subset.height == 0:
        return np.zeros((n_regions, n_samples), np.int32)
    _si = np.repeat(np.arange(n_samples, dtype=np.int64), n_regions)
    _q  = np.tile(np.arange(n_regions, dtype=np.int64), n_samples)
    chrom_keys = [f"{contig}|{s}" for s in np.repeat(samples, n_regions)]
    queries_df = pd.DataFrame({
        "Chromosome": chrom_keys,
        "Start": np.tile(starts, n_samples),
        "End":   np.tile(ends,   n_samples),
        "_q": _q, "_si": _si,
    })
    tpd = contig_subset.select("sample_id", "start", "end").to_pandas()
    tpd["Chromosome"] = contig + "|" + tpd["sample_id"]
    tpd = tpd.rename(columns={"start": "Start", "end": "End"})[["Chromosome", "Start", "End"]]
    counts = pr.PyRanges(queries_df).count_overlaps(pr.PyRanges(tpd), overlap_col="Count")
    df = counts.sort_values(["_q", "_si"])
    return df["Count"].to_numpy().reshape(n_regions, n_samples).astype(np.int32)


def count_join_groupby(table_df, samples, starts, ends, contig="chr1"):
    n_regions, n_samples = len(starts), len(samples)
    sample_to_si = {s: i for i, s in enumerate(samples)}
    contig_subset = table_df.filter(
        (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
    )
    if contig_subset.height == 0:
        return np.zeros((n_regions, n_samples), np.int32)
    queries_df = pd.DataFrame({
        "Chromosome": np.full(n_regions, contig, dtype=object).astype(str),
        "Start": starts, "End": ends,
        "_q": np.arange(n_regions, dtype=np.int64),
    })
    tpd = contig_subset.select("sample_id", "start", "end").to_pandas()
    tpd = tpd.rename(columns={"start": "Start", "end": "End"})
    tpd["Chromosome"] = contig
    tpd = tpd[["Chromosome", "Start", "End", "sample_id"]]
    joined = pr.PyRanges(queries_df).join_overlaps(pr.PyRanges(tpd), suffix="_b")
    out = np.zeros((n_regions, n_samples), np.int32)
    if len(joined) == 0:
        return out
    q_idx  = joined["_q"].to_numpy()
    si_idx = joined["sample_id"].map(sample_to_si).to_numpy()
    np.add.at(out, (q_idx, si_idx), 1)
    return out


# ── harness ──────────────────────────────────────────────────────────────────

def time_and_mem(fn, *args):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6, result


METHODS = [
    ("xprod",        count_xprod),
    ("join_groupby", count_join_groupby),
]


def main():
    print(f"pyranges1 {pr.__version__}  polars {pl.__version__}  python {sys.version.split()[0]}")
    print(f"{'case':>8} {'n_reg':>6} {'n_s':>5} {'ipp':>5} {'method':>18} {'time_s':>9} {'peak_MB':>9}")
    print("-" * 68)

    rows = []
    for case in CASES:
        n_r, n_s, ipp = case["n_regions"], case["n_samples"], case["ipp"]
        table = gen_table(n_s, ipp)
        starts, ends = gen_queries(n_r)
        samples = [f"s{i}" for i in range(n_s)]

        for m_name, fn in METHODS:
            try:
                fn(table, samples, starts, ends)  # warm-up
                times, peaks = [], []
                for trial in range(N_TRIALS):
                    t_data = gen_table(n_s, ipp, seed=trial * 31)
                    s_arr, e_arr = gen_queries(n_r, seed=trial * 31 + 1)
                    t, p, _ = time_and_mem(fn, t_data, samples, s_arr, e_arr)
                    times.append(t); peaks.append(p)
                t_med = float(np.median(times))
                t_std = float(np.std(times))
                p_med = float(np.median(peaks))
                print(f"{case['name']:>8} {n_r:>6} {n_s:>5} {ipp:>5} {m_name:>18} {t_med:>9.3f}±{t_std:.3f} {p_med:>9.1f}")
                for i, (t, p) in enumerate(zip(times, peaks)):
                    rows.append({"backend": "pyranges1", "method": m_name,
                                 "case": case["name"], "n_regions": n_r,
                                 "n_samples": n_s, "ipp": ipp,
                                 "trial": i, "time_s": t, "peak_MB": p})
            except Exception as e:
                print(f"{case['name']:>8} {n_r:>6} {n_s:>5} {ipp:>5} {m_name:>18} FAIL ({e!s:.40})")
                rows.append({"backend": "pyranges1", "method": m_name,
                              "case": case["name"], "n_regions": n_r,
                              "n_samples": n_s, "ipp": ipp,
                              "trial": 0, "time_s": None, "peak_MB": None})

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend","method","case","n_regions","n_samples","ipp","trial","time_s","peak_MB"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {OUT_CSV}")


if __name__ == "__main__":
    main()
