"""Unit tests for build_splice_plan: permutation + offset math."""

import numpy as np
import pytest

from genvarloader._dataset._splice import build_splice_plan


def test_plan_no_inner_axes():
    """E=1 case (RefDataset): plan is essentially an identity grouping."""
    # 2 splice rows × 1 sample, row 0 has 2 elements, row 1 has 1 element.
    # B = 3 queries total.
    lengths = np.array([3, 4, 5], dtype=np.int32)  # shape (3,)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # No inner fixed, so perm is identity.
    np.testing.assert_array_equal(plan.permutation, [0, 1, 2])
    np.testing.assert_array_equal(plan.permuted_lengths, [3, 4, 5])
    np.testing.assert_array_equal(plan.permuted_out_offsets, [0, 3, 7, 12])
    # group_offsets at (row, sample) granularity: 2 entries + 1.
    np.testing.assert_array_equal(plan.group_offsets, [0, 7, 12])
    assert plan.out_shape == (2, 1, None)
    # flat_out_shape collapses (n_rows=2, n_samples=1) → n_pairs=2.
    assert plan.flat_out_shape == (2, None)


def test_plan_ploidy_2():
    """B=3 queries × P=2 ploidy. Each splice row's ploidies must be contiguous.

    Splice layout:
      row 0, sample 0 = elements [0, 1]  (2 elements)
      row 1, sample 0 = elements [2]     (1 element)

    Inner-fixed lengths (B, P) where B=3, P=2:
      query 0 (row 0 elem 0): ploidy lens [10, 11]
      query 1 (row 0 elem 1): ploidy lens [20, 21]
      query 2 (row 1 elem 0): ploidy lens [30, 31]

    Desired permuted order (row, sample, ploidy, element) C-order:
      (r=0, s=0, p=0, e=0), (r=0, s=0, p=0, e=1),
      (r=0, s=0, p=1, e=0), (r=0, s=0, p=1, e=1),
      (r=1, s=0, p=0, e=0),
      (r=1, s=0, p=1, e=0)

    k_idx in current layout is (query, ploidy) C-order:
      k = [(q0,p0), (q0,p1), (q1,p0), (q1,p1), (q2,p0), (q2,p1)]
        = [0, 1, 2, 3, 4, 5]

    So perm pulls k-indices in this order:
      k(q=0, p=0)=0, k(q=1, p=0)=2,   # row 0 sample 0 ploidy 0 elements
      k(q=0, p=1)=1, k(q=1, p=1)=3,   # row 0 sample 0 ploidy 1 elements
      k(q=2, p=0)=4,                  # row 1 sample 0 ploidy 0
      k(q=2, p=1)=5,                  # row 1 sample 0 ploidy 1
    """
    lengths = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.int32)  # (B=3, P=2)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    np.testing.assert_array_equal(plan.permutation, [0, 2, 1, 3, 4, 5])
    np.testing.assert_array_equal(plan.permuted_lengths, [10, 20, 11, 21, 30, 31])
    np.testing.assert_array_equal(
        plan.permuted_out_offsets, [0, 10, 30, 41, 62, 92, 123]
    )
    # group_offsets at (row, sample, ploidy) granularity: 2*1*2 = 4 cells + 1.
    # cell sums: row0,s0,p0 = 10+20=30; row0,s0,p1 = 11+21=32; row1,s0,p0 = 30; row1,s0,p1 = 31.
    np.testing.assert_array_equal(plan.group_offsets, [0, 30, 62, 92, 123])
    assert plan.out_shape == (2, 1, 2, None)
    # flat_out_shape collapses (n_rows=2, n_samples=1) → n_pairs=2.
    assert plan.flat_out_shape == (2, 2, None)


def test_plan_multi_sample_ploidy_2():
    """n_samples=2, ploidy=2. Verify (row, sample, ploidy) C-order."""
    # 1 splice row × 2 samples. Row has 2 elements.
    # B = 4 queries (row, sample, element) C-order:
    #   q0 = (r=0, s=0, e=0)
    #   q1 = (r=0, s=0, e=1)
    #   q2 = (r=0, s=1, e=0)
    #   q3 = (r=0, s=1, e=1)
    lengths = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)  # (B=4, P=2)
    splice_row_offsets = np.array([0, 2, 4], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=2,
        n_rows=1,
    )
    # k_idx = query * P + ploidy
    # Desired order: (r, s, p, e). For r=0:
    #   s=0, p=0: e=0 → k(q=0,p=0)=0; e=1 → k(q=1,p=0)=2
    #   s=0, p=1: e=0 → k(q=0,p=1)=1; e=1 → k(q=1,p=1)=3
    #   s=1, p=0: e=0 → k(q=2,p=0)=4; e=1 → k(q=3,p=0)=6
    #   s=1, p=1: e=0 → k(q=2,p=1)=5; e=1 → k(q=3,p=1)=7
    np.testing.assert_array_equal(plan.permutation, [0, 2, 1, 3, 4, 6, 5, 7])
    np.testing.assert_array_equal(plan.permuted_lengths, [1, 3, 2, 4, 5, 7, 6, 8])
    # group_offsets at (1, 2, 2) granularity = 4 cells + 1.
    # cell sums: 1+3=4, 2+4=6, 5+7=12, 6+8=14
    np.testing.assert_array_equal(plan.group_offsets, [0, 4, 10, 22, 36])
    assert plan.out_shape == (1, 2, 2, None)
    # flat_out_shape collapses (n_rows=1, n_samples=2) → n_pairs=2.
    assert plan.flat_out_shape == (2, 2, None)


def test_plan_total_bytes_consistent():
    """sum(lengths) == permuted_out_offsets[-1] == group_offsets[-1]."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 20, size=(6, 3), dtype=np.int32)
    splice_row_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=3,
    )
    total = int(lengths.sum())
    assert int(plan.permuted_out_offsets[-1]) == total
    assert int(plan.group_offsets[-1]) == total


def test_plan_single_element_rows():
    """Every splice row has exactly one element — no concatenation needed."""
    lengths = np.array([[5, 6], [7, 8]], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # With singleton splice rows the permutation still groups by (r, s, p).
    np.testing.assert_array_equal(plan.permutation, [0, 1, 2, 3])
    np.testing.assert_array_equal(plan.permuted_lengths, [5, 6, 7, 8])


def test_plan_inner_fixed_size_3():
    """E=3: e.g. a track axis of 3 stacked tracks. Verify general inner-fixed handling."""
    # 1 splice row × 1 sample × 2 elements × 3 tracks.
    # B = 2 queries × 3 tracks = 6 inner k-indices.
    lengths = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # (B=2, T=3)
    splice_row_offsets = np.array([0, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=1,
    )
    # k = query*T + t. Desired order (r=0, s=0, t, e):
    #   t=0: e=0 → k=0; e=1 → k=3
    #   t=1: e=0 → k=1; e=1 → k=4
    #   t=2: e=0 → k=2; e=1 → k=5
    np.testing.assert_array_equal(plan.permutation, [0, 3, 1, 4, 2, 5])
    np.testing.assert_array_equal(plan.permuted_lengths, [1, 4, 2, 5, 3, 6])
    np.testing.assert_array_equal(plan.group_offsets, [0, 5, 12, 21])
    assert plan.out_shape == (1, 1, 3, None)


def test_plan_dtype_invariants():
    """perm is intp, lengths is int32, offsets is int64-compatible."""
    lengths = np.array([3, 4], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    assert plan.permutation.dtype == np.intp
    assert plan.permuted_lengths.dtype == np.int32
    # offset arrays use seqpro's OFFSET_TYPE (int64).
    assert plan.permuted_out_offsets.dtype == np.int64
    assert plan.group_offsets.dtype == np.int64


def test_ref_call_with_plan_writes_per_element_layout():
    """Ref.__call__(splice_plan=...) returns a per-element Ragged whose
    offsets are plan.permuted_out_offsets and shape is (n_elements, None)."""
    from pathlib import Path

    import genvarloader as gvl
    import polars as pl

    from genvarloader._dataset._reconstruct import Ref
    from genvarloader._dataset._splice import build_splice_plan
    from genvarloader._dataset._utils import bed_to_regions

    # Find the data directory. In git worktrees the large binary test data
    # lives in the main project's tests/data/, not in the worktree checkout.
    import subprocess

    _here = Path(__file__).resolve()
    # Try: worktree-local first, then git common dir (main worktree root).
    _candidates = [
        _here.parent.parent / "data" / "fasta" / "hg38.fa.bgz",
        _here.parent.parent.parent / "data" / "fasta" / "hg38.fa.bgz",
    ]
    try:
        _git_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"],
                cwd=str(_here.parent),
                text=True,
            ).strip()
        ).parent
        _candidates.insert(0, _git_root / "tests" / "data" / "fasta" / "hg38.fa.bgz")
    except Exception:
        pass
    for _c in _candidates:
        if _c.exists():
            DDIR = _c.parent.parent
            break
    else:
        pytest.skip("hg38.fa.bgz not found — run from repo root or main project")

    reference = gvl.Reference.from_path(DDIR / "fasta" / "hg38.fa.bgz", in_memory=False)

    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
        }
    )

    regions = bed_to_regions(bed, reference.c_map)
    # Two splice rows: row 0 = elements [0, 1], row 1 = element [2].
    flat_r_idx = np.array([0, 1, 2], dtype=np.intp)
    splice_offsets = np.array([0, 2, 3], dtype=np.int64)
    lengths = (regions[flat_r_idx, 2] - regions[flat_r_idx, 1]).astype(np.int32)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_offsets,
        n_samples=1,
        n_rows=2,
    )

    reconstructor = Ref(reference=reference)
    out = reconstructor(
        idx=flat_r_idx,
        r_idx=flat_r_idx,
        regions=regions[flat_r_idx],
        output_length="ragged",
        jitter=0,
        rng=np.random.default_rng(0),
        deterministic=True,
        splice_plan=plan,
    )

    # Per-element shape: (n_elements=3, None) — NOT the grouped (2, 1, None).
    assert out.shape == (3, None), f"unexpected shape: {out.shape}"
    # Total byte count matches sum of per-region lengths (3 regions × 10 bp each).
    assert int(out.data.shape[0]) == int(lengths.sum())
    # Per-element offsets must match the plan's permuted_out_offsets exactly.
    np.testing.assert_array_equal(out.offsets, plan.permuted_out_offsets)


def test_tracks_call_float32_splice_plan():
    """Direct unit test for Tracks._call_float32 with a SplicePlan.

    Constructs a synthetic Tracks reconstructor with n_regions=2, n_samples=2,
    one SAMPLE track named 'synthetic' populated with constant-value intervals
    that cover the full extent of each region.  Verifies that:
    - The returned shape is (plan.permuted_lengths.shape[0], None)
    - out.data.shape[0] == sum of all lengths
    - out.offsets == plan.permuted_out_offsets
    - Each scatter slice holds the expected constant value (7.0)
    """
    from seqpro.rag import Ragged

    from genvarloader._dataset._reconstruct import TrackType, Tracks
    from genvarloader._dataset._splice import build_splice_plan
    from genvarloader._ragged import RaggedIntervals, RaggedTracks

    # ------------------------------------------------------------------
    # 1. Build synthetic intervals for 2 regions × 2 samples, 1 track.
    #    Region 0: genomic coords [1000, 1004)  → length 4
    #    Region 1: genomic coords [2000, 2006)  → length 6
    #    Each (region, sample) pair gets one interval that covers the full
    #    region width with value 7.0.
    # ------------------------------------------------------------------
    n_regions = 2
    n_samples = 2
    n_pairs = n_regions * n_samples  # 4
    TRACK_VALUE = 7.0

    # Flat array of intervals: one per (region, sample) pair.
    # The flat index order is C-order: (r0s0, r0s1, r1s0, r1s1).
    itv_starts = np.array([1000, 1000, 2000, 2000], dtype=np.int32)
    itv_ends = np.array([1004, 1004, 2006, 2006], dtype=np.int32)
    itv_values = np.full(n_pairs, TRACK_VALUE, dtype=np.float32)

    # offsets: one interval per slot → [0, 1, 2, 3, 4]
    offsets = np.arange(n_pairs + 1, dtype=np.int64)

    shape = (n_regions, n_samples, None)
    rag_starts = Ragged.from_offsets(itv_starts, shape, offsets)
    rag_ends = Ragged.from_offsets(itv_ends, shape, offsets)
    rag_values = Ragged.from_offsets(itv_values, shape, offsets)
    intervals = RaggedIntervals(rag_starts, rag_ends, rag_values)

    # ------------------------------------------------------------------
    # 2. Build the Tracks reconstructor.
    # ------------------------------------------------------------------
    track_name = "synthetic"
    tracks = Tracks(
        intervals={track_name: intervals},
        active_tracks={track_name: TrackType.SAMPLE},
        available_tracks={track_name: TrackType.SAMPLE},
        kind=RaggedTracks,
        n_regions=n_regions,
        n_samples=n_samples,
    )

    # ------------------------------------------------------------------
    # 3. Build regions array (chrom-col unused; only start/end matter).
    #    Shape: (B, 3) where columns are [chrom_idx, chromStart, chromEnd].
    #    B=4 queries: (r0,s0), (r0,s1), (r1,s0), (r1,s1) in C-order.
    #    For SAMPLE tracks the kernel uses flat idx = r*n_samples + s.
    # ------------------------------------------------------------------
    # B=4 queries; idx is the flat (region*n_samples + sample) index.
    flat_idx = np.array([0, 1, 2, 3], dtype=np.intp)  # r0s0, r0s1, r1s0, r1s1
    r_idx = np.array([0, 0, 1, 1], dtype=np.intp)  # region index per query

    # Regions: [chrom_idx, chromStart, chromEnd] — use int32 as the code does.
    regions = np.array(
        [
            [0, 1000, 1004],  # region 0, length 4
            [0, 1000, 1004],  # region 0 again (sample 1 → same coords)
            [0, 2000, 2006],  # region 1, length 6
            [0, 2000, 2006],  # region 1 again (sample 1 → same coords)
        ],
        dtype=np.int32,
    )

    # ------------------------------------------------------------------
    # 4. Build the SplicePlan.
    #    inner_fixed = (n_tracks=1,) so lengths shape is (B=4, T=1).
    #    splice_row_offsets: 2 splice rows, each with 2 elements.
    # ------------------------------------------------------------------
    lengths_bt = np.array([[4], [4], [6], [6]], dtype=np.int32)  # (B=4, T=1)
    splice_row_offsets = np.array([0, 2, 4], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths_bt,
        splice_row_offsets=splice_row_offsets,
        n_samples=n_samples,
        n_rows=2,
    )

    # ------------------------------------------------------------------
    # 5. Call _call_float32 directly with the splice plan.
    # ------------------------------------------------------------------
    out = tracks._call_float32(
        idx=flat_idx,
        r_idx=r_idx,
        regions=regions,
        output_length="ragged",
        splice_plan=plan,
    )

    # ------------------------------------------------------------------
    # 6. Assertions.
    # ------------------------------------------------------------------
    # Shape: (n_permuted_elements, None) where n_permuted_elements = B*T = 4.
    expected_n_elements = plan.permuted_lengths.shape[0]
    assert out.shape == (expected_n_elements, None), (
        f"unexpected shape: {out.shape!r}, expected ({expected_n_elements}, None)"
    )

    # Total bytes in the flat buffer == sum of all per-element lengths.
    total_bytes = int(lengths_bt.sum())
    assert int(out.data.shape[0]) == total_bytes, (
        f"data length {out.data.shape[0]} != {total_bytes}"
    )

    # Per-element offsets must exactly match plan.permuted_out_offsets.
    np.testing.assert_array_equal(out.offsets, plan.permuted_out_offsets)

    # Optional: verify scatter correctness — every output byte should be 7.0
    # because the synthetic intervals cover the full region for every query.
    np.testing.assert_array_equal(
        out.data,
        np.full(total_bytes, TRACK_VALUE, dtype=np.float32),
        err_msg="scatter loop placed unexpected values in the output buffer",
    )
