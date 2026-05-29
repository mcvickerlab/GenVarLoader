"""Bench count_intervals equivalents using polars-bio (py3.10 dev env).

Methods:
  per_sample_loop  : original code (pb.count_overlaps per sample)
  lazy_filter      : single pb.overlap lazy + filter on sample_id equality
  no_xprod_overlap : n_regions queries only; sample_id flows from table side

Run:
    pixi run -e dev python experiments/bench_table_overlap/bench_polars_bio.py
"""

from __future__ import annotations

import csv
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
import polars_bio as pb

sys.path.insert(0, str(Path(__file__).parent))
from _common import CASES, N_TRIALS, gen_queries, gen_table

pb.set_option("datafusion.bio.coordinate_system_check", "false")
pb.set_option("datafusion.bio.coordinate_system_zero_based", True)

OUT_CSV = Path(__file__).parent / "results_polars_bio.csv"


# ── methods ──────────────────────────────────────────────────────────────────


def count_per_sample_loop(table_df, samples, starts, ends, contig="chr1"):
    n_regions, n_samples = len(starts), len(samples)
    contig_subset = table_df.filter(pl.col("chrom") == contig)
    if contig_subset.height == 0:
        return np.zeros((n_regions, n_samples), np.int32)
    queries = pl.DataFrame({
        "chrom": np.full(n_regions, contig, dtype=object).astype(str),
        "start": starts.astype(np.int64),
        "end": ends.astype(np.int64),
        "_q": np.arange(n_regions, dtype=np.int64),
    })
    out = np.zeros((n_regions, n_samples), np.int32)
    for si, s in enumerate(samples):
        sub_s = contig_subset.filter(pl.col("sample_id") == s).select(
            "chrom", "start", "end"
        )
        if sub_s.height == 0:
            continue
        counts_df = pb.count_overlaps(
            queries,
            sub_s,
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            output_type="polars.DataFrame",
        )
        sorted_counts = (
            counts_df.sort("_q")["count"].to_numpy().astype(np.int32, copy=False)
        )
        if len(sorted_counts) < n_regions:
            idx_df = pl.DataFrame({"_q": np.arange(n_regions, dtype=np.int64)})
            sorted_counts = (
                idx_df
                .join(counts_df.select("_q", "count"), on="_q", how="left")
                .fill_null(0)["count"]
                .to_numpy()
                .astype(np.int32, copy=False)
            )
        out[:, si] = sorted_counts
    return out


def count_lazy_filter(table_df, samples, starts, ends, contig="chr1"):
    n_regions, n_samples = len(starts), len(samples)
    sample_to_si = {s: i for i, s in enumerate(samples)}
    contig_subset = table_df.filter(
        (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
    )
    if contig_subset.height == 0:
        return np.zeros((n_regions, n_samples), np.int32)
    _n = n_regions * n_samples
    queries = pl.DataFrame({
        "chrom": np.full(_n, contig, dtype=object).astype(str),
        "start": np.tile(starts, n_samples).astype(np.int64),
        "end": np.tile(ends, n_samples).astype(np.int64),
        "_q": np.tile(np.arange(n_regions, dtype=np.int64), n_samples),
        "sample_id": np.repeat(np.array(samples, dtype=object), n_regions).astype(str),
    })
    result = (
        pb
        .overlap(
            queries.lazy(),
            contig_subset.select("chrom", "start", "end", "sample_id").lazy(),
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            output_type="polars.LazyFrame",
        )
        .filter(pl.col("sample_id_1") == pl.col("sample_id_2"))
        .collect()
    )
    out = np.zeros((n_regions, n_samples), np.int32)
    if result.height == 0:
        return out
    q_idx = result["_q_1"].to_numpy()
    si_idx = np.fromiter(
        (sample_to_si[s] for s in result["sample_id_1"].to_list()), dtype=np.int64
    )
    np.add.at(out, (q_idx, si_idx), 1)
    return out


def count_no_xprod(table_df, samples, starts, ends, contig="chr1"):
    n_regions, n_samples = len(starts), len(samples)
    sample_to_si = {s: i for i, s in enumerate(samples)}
    contig_subset = table_df.filter(
        (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
    )
    if contig_subset.height == 0:
        return np.zeros((n_regions, n_samples), np.int32)
    queries = pl.DataFrame({
        "chrom": np.full(n_regions, contig, dtype=object).astype(str),
        "start": starts.astype(np.int64),
        "end": ends.astype(np.int64),
        "_q": np.arange(n_regions, dtype=np.int64),
    })
    result = pb.overlap(
        queries,
        contig_subset.select("chrom", "start", "end", "sample_id"),
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
        output_type="polars.DataFrame",
    )
    out = np.zeros((n_regions, n_samples), np.int32)
    if result.height == 0:
        return out
    q_idx = result["_q_1"].to_numpy()
    si_idx = np.fromiter(
        (sample_to_si[s] for s in result["sample_id_2"].to_list()), dtype=np.int64
    )
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
    ("per_sample_loop", count_per_sample_loop),
    ("lazy_filter", count_lazy_filter),
    ("no_xprod", count_no_xprod),
]


def main():
    print(
        f"polars-bio {pb.__version__}  polars {pl.__version__}  python {sys.version.split()[0]}"
    )
    print(
        f"{'case':>8} {'n_reg':>6} {'n_s':>5} {'ipp':>5} {'method':>18} {'time_s':>9} {'peak_MB':>9}"
    )
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
                    times.append(t)
                    peaks.append(p)
                t_med = float(np.median(times))
                t_std = float(np.std(times))
                p_med = float(np.median(peaks))
                print(
                    f"{case['name']:>8} {n_r:>6} {n_s:>5} {ipp:>5} {m_name:>18} {t_med:>9.3f}±{t_std:.3f} {p_med:>9.1f}"
                )
                for i, (t, p) in enumerate(zip(times, peaks)):
                    rows.append({
                        "backend": "polars_bio",
                        "method": m_name,
                        "case": case["name"],
                        "n_regions": n_r,
                        "n_samples": n_s,
                        "ipp": ipp,
                        "trial": i,
                        "time_s": t,
                        "peak_MB": p,
                    })
            except Exception as e:
                print(
                    f"{case['name']:>8} {n_r:>6} {n_s:>5} {ipp:>5} {m_name:>18} FAIL ({e!s:.40})"
                )
                rows.append({
                    "backend": "polars_bio",
                    "method": m_name,
                    "case": case["name"],
                    "n_regions": n_r,
                    "n_samples": n_s,
                    "ipp": ipp,
                    "trial": 0,
                    "time_s": None,
                    "peak_MB": None,
                })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "method",
                "case",
                "n_regions",
                "n_samples",
                "ipp",
                "trial",
                "time_s",
                "peak_MB",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {OUT_CSV}")


if __name__ == "__main__":
    main()
