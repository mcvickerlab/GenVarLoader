"""Property-based parity: Rust Table vs a brute-force numpy oracle."""

import numpy as np
import polars as pl
from genvarloader import Table
from genvarloader._utils import lengths_to_offsets
from hypothesis import given, settings
from hypothesis import strategies as st


def _rand_table(rng, n_samples, n_contigs, n_intervals):
    rows = []
    for _ in range(n_intervals):
        s = int(rng.integers(0, n_samples))
        c = int(rng.integers(0, n_contigs))
        start = int(rng.integers(0, 500))
        width = int(rng.integers(1, 50))  # positive width only
        rows.append((f"s{s}", f"chr{c}", start, start + width, float(rng.random())))
    df = pl.DataFrame(
        rows, schema=["sample_id", "chrom", "start", "end", "value"], orient="row"
    )
    return df


def _brute(df, stored_df, contig, starts, ends, samples):
    """Brute-force oracle.

    `df` is the original user DataFrame (used for counts — order-independent).
    `stored_df` is ``t._df``, the frame as sorted and stored by Table
    (``chrom, sample_id, start`` stable sort).  The Rust backend stores intervals
    in exactly this order, so the interval-value comparisons must use `stored_df`
    to match the within-equal-start tie-breaking that polars's stable sort produces.
    """
    counts = np.zeros((len(starts), len(samples)), np.int32)
    cells = {}
    for si, s in enumerate(samples):
        # counts: order-independent, use original df (any consistent filter is fine)
        sub_orig = df.filter((pl.col("sample_id") == s) & (pl.col("chrom") == contig))
        ts_orig = sub_orig["start"].to_numpy()
        te_orig = sub_orig["end"].to_numpy()
        # intervals: use stored order to match Rust's index-sort output
        sub_stored = stored_df.filter(
            (pl.col("sample_id") == s) & (pl.col("chrom") == contig)
        )
        ts = sub_stored["start"].to_numpy()
        te = sub_stored["end"].to_numpy()
        tv = sub_stored["value"].to_numpy()
        for ri, (rs, re_) in enumerate(zip(starts, ends)):
            mask_orig = (ts_orig < re_) & (te_orig > rs)
            counts[ri, si] = int(mask_orig.sum())
            mask = (ts < re_) & (te > rs)
            cells[(ri, si)] = (
                ts[mask].astype(np.int32),
                te[mask].astype(np.int32),
                tv[mask].astype(np.float32),
            )
    return counts, cells


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_samples=st.integers(1, 3),
    n_contigs=st.integers(1, 3),
    n_intervals=st.integers(0, 40),
    n_regions=st.integers(1, 6),
)
def test_count_and_intervals_match_oracle(
    seed, n_samples, n_contigs, n_intervals, n_regions
):
    rng = np.random.default_rng(seed)
    df = _rand_table(rng, n_samples, n_contigs, n_intervals)
    if df.height == 0:
        return
    t = Table("sig", df)
    samples = [f"s{i}" for i in range(n_samples)]
    for c in range(n_contigs):
        contig = f"chr{c}"
        starts = rng.integers(0, 500, n_regions).astype(np.int32)
        ends = (starts + rng.integers(1, 100, n_regions)).astype(np.int32)
        present = [s for s in samples if s in t.samples]
        if not present:
            continue
        counts = t.count_intervals(contig, starts, ends, sample=present)
        exp_counts, cells = _brute(df, t._df, contig, starts, ends, present)
        np.testing.assert_array_equal(counts, exp_counts)

        offsets = lengths_to_offsets(counts.ravel())
        itvs = t._intervals_from_offsets(contig, starts, ends, offsets, sample=present)
        n_sel = len(present)
        for ri in range(n_regions):
            for sj in range(n_sel):
                cell = ri * n_sel + sj
                lo, hi = int(offsets[cell]), int(offsets[cell + 1])
                exp_s, exp_e, exp_v = cells[(ri, sj)]
                np.testing.assert_array_equal(itvs.starts.data[lo:hi], exp_s)
                np.testing.assert_array_equal(itvs.ends.data[lo:hi], exp_e)
                np.testing.assert_array_equal(itvs.values.data[lo:hi], exp_v)
