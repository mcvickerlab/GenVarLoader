"""Unit tests for Tracks._call_float32 with SplicePlan inputs."""

import numpy as np


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
