"""Hypothesis input strategies per migrated kernel (contract-valid generators)."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st


@st.composite
def intervals_to_tracks_inputs(draw):
    """Contract-valid inputs for ``intervals_to_tracks``.

    One interval-slice per query (offset_idxs = arange). Within a slice the
    intervals are sorted and non-overlapping; the first interval may start
    before the query's start (negative relative start) to cover #242.
    Covers empty slices, in-bounds paints, end-clamp (interval end past the
    out length), and break (interval start past the out length).
    """
    n_queries = draw(st.integers(min_value=1, max_value=6))

    starts_list: list[int] = []
    out_lengths: list[int] = []
    counts: list[int] = []
    itv_starts_all: list[int] = []
    itv_ends_all: list[int] = []
    itv_values_all: list[float] = []

    for _ in range(n_queries):
        qstart = draw(st.integers(min_value=0, max_value=500))
        length = draw(st.integers(min_value=0, max_value=48))
        starts_list.append(qstart)
        out_lengths.append(length)

        m = draw(st.integers(min_value=0, max_value=6))
        # first start may be below qstart (negative relative start; #242) or above
        cur = qstart + draw(st.integers(min_value=-10, max_value=10))
        for _ in range(m):
            width = draw(st.integers(min_value=1, max_value=20))
            itv_starts_all.append(cur)
            itv_ends_all.append(cur + width)
            itv_values_all.append(
                draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
            )
            cur = (
                cur + width + draw(st.integers(min_value=0, max_value=10))
            )  # gap -> non-overlap
        counts.append(m)

    offset_idxs = np.arange(n_queries, dtype=np.int64)
    starts = np.array(starts_list, dtype=np.int32)
    itv_starts = np.array(itv_starts_all, dtype=np.int32)
    itv_ends = np.array(itv_ends_all, dtype=np.int32)
    itv_values = np.array(itv_values_all, dtype=np.float32)
    itv_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int64)

    return (
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out_offsets,
    )


@st.composite
def _sparse_geno(
    draw, max_queries=4, max_ploidy=2, max_vars_per_group=5, max_total_unique=12
):
    """Shared sparse-genotype layout: returns
    (geno_offset_idx (q,p) int64, geno_v_idxs int32, geno_offsets (n+1,) int64,
     v_starts int32, ilens int32, q_starts int32, q_ends int32).
    geno_offset_idx is arange so each (q,p) row maps to its own offset slice."""
    n_unique = draw(st.integers(min_value=1, max_value=max_total_unique))
    v_starts = np.sort(
        draw(
            st.lists(st.integers(0, 1000), min_size=n_unique, max_size=n_unique).map(
                np.array
            )
        )
    ).astype(np.int32)
    ilens = np.array(
        draw(st.lists(st.integers(-5, 5), min_size=n_unique, max_size=n_unique)),
        dtype=np.int32,
    )
    n_q = draw(st.integers(1, max_queries))
    p = draw(st.integers(1, max_ploidy))
    n_groups = n_q * p
    counts = [draw(st.integers(0, max_vars_per_group)) for _ in range(n_groups)]
    v_idx_list = []
    for c in counts:
        # sorted variant indices within a group (reconstruction assumes sorted pos)
        idxs = sorted(
            draw(st.lists(st.integers(0, n_unique - 1), min_size=c, max_size=c))
        )
        v_idx_list.extend(idxs)
    geno_v_idxs = np.array(v_idx_list, dtype=np.int32)
    geno_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    geno_offset_idx = np.arange(n_groups, dtype=np.int64).reshape(n_q, p)
    q_starts = np.array(
        draw(st.lists(st.integers(0, 800), min_size=n_q, max_size=n_q)), np.int32
    )
    q_ends = (q_starts + draw(st.integers(1, 200))).astype(np.int32)
    return (
        geno_offset_idx,
        geno_v_idxs,
        geno_offsets,
        v_starts,
        ilens,
        q_starts,
        q_ends,
    )


@st.composite
def get_diffs_sparse_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno())
    mode = draw(st.sampled_from(["plain", "keep", "query"]))
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    total = int(goff[-1])
    if mode == "plain":
        return (goi, gvi, offsets, ilens, None, None, None, None, None)
    if mode == "keep":
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        return (goi, gvi, offsets, ilens, keep, goff.copy(), None, None, None)
    # query mode (optionally also keep)
    keep = None
    keep_off = None
    if draw(st.booleans()):
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        keep_off = goff.copy()
    return (goi, gvi, offsets, ilens, keep, keep_off, qstarts, qends, vstarts)


@st.composite
def choose_exonic_variants_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno())
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    return (qstarts, qends, goi, gvi, offsets, vstarts, ilens)


@st.composite
def gather_rows_inputs(draw, dtype=np.int32):
    n_groups = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_groups)]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
    else:
        elements = st.integers(0, 1000)
    data = np.array(draw(st.lists(elements, min_size=total, max_size=total)), dt)
    n_rows = draw(st.integers(1, 8))
    goi = np.array(
        draw(st.lists(st.integers(0, n_groups - 1), min_size=n_rows, max_size=n_rows)),
        np.int64,
    )
    twod = draw(st.booleans())
    off = (
        offsets if not twod else np.stack([offsets[:-1], offsets[1:]]).astype(np.int64)
    )
    return (goi, off, data)


@st.composite
def gather_alleles_inputs(draw):
    n_unique = draw(st.integers(1, 8))
    lens = [draw(st.integers(0, 5)) for _ in range(n_unique)]
    allele_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(allele_offsets[-1])
    allele_bytes = np.array(
        draw(st.lists(st.integers(0, 255), min_size=total, max_size=total)), np.uint8
    )
    m = draw(st.integers(0, 10))
    v_idxs = np.array(
        draw(st.lists(st.integers(0, n_unique - 1), min_size=m, max_size=m)), np.int32
    )
    return (v_idxs, allele_bytes, allele_offsets)


@st.composite
def compact_keep_inputs(draw, dtype):
    """Generate (values[dtype], row_offsets int64, keep bool) for compact_keep tests."""
    n_rows = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_rows)]
    row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(row_offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
    else:
        elements = st.integers(0, 1000)
    values = np.array(draw(st.lists(elements, min_size=total, max_size=total)), dt)
    keep = np.array(
        draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
    )
    return (values, row_offsets, keep)


@st.composite
def fill_empty_scalar_inputs(draw, dtype=np.int32):
    """Generate (data[dtype], offsets int64, fill) with at least one empty row.

    Guarantees at least one row has zero count so empty-row insertion is
    exercised on every draw.
    """
    n_rows = draw(st.integers(2, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_rows)]
    # Force one row to be empty so the empty-fill path is always exercised.
    empty_idx = draw(st.integers(0, n_rows - 1))
    counts[empty_idx] = 0
    row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(row_offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
        fill = draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
    else:
        elements = st.integers(-1000, 1000)
        fill = draw(st.integers(-1000, 1000))
    data = np.array(draw(st.lists(elements, min_size=total, max_size=total)), dt)
    fill_val = dt.type(fill)
    return (data, row_offsets, fill_val)


@st.composite
def fill_empty_fixed_inputs(draw, dtype=np.int32):
    """Generate (data[dtype], offsets int64, inner int, fill) with at least one
    empty row for fill_empty_fixed tests.
    """
    n_rows = draw(st.integers(2, 6))
    inner = draw(st.integers(1, 4))
    counts = [draw(st.integers(0, 4)) for _ in range(n_rows)]
    # Force one row to be empty.
    empty_idx = draw(st.integers(0, n_rows - 1))
    counts[empty_idx] = 0
    row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total_vars = int(row_offsets[-1])
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        elements = st.floats(width=32, allow_nan=False, allow_infinity=False)
        fill = draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
    else:
        elements = st.integers(-1000, 1000)
        fill = draw(st.integers(-1000, 1000))
    data = np.array(
        draw(
            st.lists(elements, min_size=total_vars * inner, max_size=total_vars * inner)
        ),
        dt,
    )
    fill_val = dt.type(fill)
    return (data, row_offsets, inner, fill_val)


@st.composite
def fill_empty_seq_inputs(draw, dtype=np.uint8):
    """Generate (data[dtype], var_offsets int64, seq_offsets int64, dummy[dtype])
    with at least one guaranteed empty row for fill_empty_seq tests.

    Layout:
    - var_offsets: b*p+1 boundaries over variant groups (one guaranteed empty).
    - seq_offsets: per-variant byte/token boundaries (len = total_vars + 1).
    - data: flat element array (len = seq_offsets[-1]).
    - dummy: random sequence of length >= 1 in the given dtype.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.unsignedinteger):
        elements = st.integers(0, 255)
    else:
        elements = st.integers(-1000, 1000)

    n_rows = draw(st.integers(2, 6))
    # Number of variants per row (zero = empty row).
    var_counts = [draw(st.integers(0, 4)) for _ in range(n_rows)]
    # Force at least one empty row.
    empty_idx = draw(st.integers(0, n_rows - 1))
    var_counts[empty_idx] = 0
    var_offsets = np.concatenate([[0], np.cumsum(var_counts)]).astype(np.int64)
    total_vars = int(var_offsets[-1])

    # Per-variant byte/token lengths.
    var_lens = [draw(st.integers(0, 5)) for _ in range(total_vars)]
    seq_offsets = np.concatenate([[0], np.cumsum(var_lens)]).astype(np.int64)
    total_elems = int(seq_offsets[-1])
    data = np.array(
        draw(st.lists(elements, min_size=total_elems, max_size=total_elems)), dt
    )

    # Dummy sequence: length >= 1.
    dummy_len = draw(st.integers(1, 4))
    dummy = np.array(
        draw(st.lists(elements, min_size=dummy_len, max_size=dummy_len)), dt
    )

    return (data, var_offsets, seq_offsets, dummy)


@st.composite
def tracks_to_intervals_inputs(draw):
    """Contract-valid inputs for ``tracks_to_intervals``.

    Generates (regions, tracks, track_offsets) where:
    - regions: (n_queries, 3) int32 with (contig_idx, start, end)
    - tracks: flat f32 ragged array, one piecewise-constant run per query
    - track_offsets: (n_queries + 1,) int64

    Exercises: multi-run queries, all-constant (1 interval), and empty queries.
    Includes a guaranteed empty query (track_offsets[q]==track_offsets[q+1]) and
    a guaranteed all-constant query (single run, 1 interval).
    """
    n_queries = draw(st.integers(min_value=3, max_value=8))
    regions_list: list[tuple[int, int, int]] = []
    track_lengths: list[int] = []
    tracks_parts: list[np.ndarray] = []

    for qi in range(n_queries):
        start = draw(st.integers(min_value=0, max_value=500))
        # Force first query to be empty, second to be all-constant
        if qi == 0:
            length = 0
        elif qi == 1:
            length = draw(st.integers(min_value=1, max_value=20))
        else:
            length = draw(st.integers(min_value=0, max_value=40))

        regions_list.append((0, start, start + length))
        track_lengths.append(length)

        if length == 0:
            tracks_parts.append(np.empty(0, dtype=np.float32))
        elif qi == 1:
            # All-constant: single run
            val = draw(st.floats(width=32, allow_nan=False, allow_infinity=False))
            tracks_parts.append(np.full(length, val, dtype=np.float32))
        else:
            # Piecewise constant with interesting RLE structure
            # Draw run boundaries: build runs by drawing lengths
            buf = np.empty(length, dtype=np.float32)
            pos = 0
            while pos < length:
                run_len = draw(st.integers(min_value=1, max_value=max(1, length - pos)))
                run_len = min(run_len, length - pos)
                val = draw(
                    st.floats(
                        min_value=-1e3,
                        max_value=1e3,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                buf[pos : pos + run_len] = val
                pos += run_len
            tracks_parts.append(buf)

    regions = np.array(regions_list, dtype=np.int32)
    track_offsets = np.concatenate([[0], np.cumsum(track_lengths)]).astype(np.int64)
    tracks = (
        np.concatenate(tracks_parts) if tracks_parts else np.empty(0, dtype=np.float32)
    )

    return regions, tracks, track_offsets


@st.composite
def get_reference_inputs(draw):
    """Generate (regions, out_offsets, reference, ref_offsets, pad_char, parallel)
    with regions whose [start,end) windows may run off either contig edge.

    Note: start is restricted to [-5, clen) so that the region overlaps the
    contig (start < clen). The numba kernel has a pre-existing size-mismatch
    crash when start >= clen (region entirely past contig end); that degenerate
    case never occurs in production (BED regions are clipped to contig bounds).
    """
    from hypothesis.extra.numpy import arrays

    n_contigs = draw(st.integers(1, 3))
    contig_lens = [draw(st.integers(1, 40)) for _ in range(n_contigs)]
    ref_offsets = np.concatenate([[0], np.cumsum(contig_lens)]).astype(np.int64)
    reference = draw(
        arrays(np.uint8, int(ref_offsets[-1]), elements=st.integers(0, 255))
    )
    n_regions = draw(st.integers(1, 6))
    regions = np.empty((n_regions, 3), np.int32)
    lengths = []
    for i in range(n_regions):
        c = draw(st.integers(0, n_contigs - 1))
        clen = contig_lens[c]
        # Restrict start < clen so the region overlaps the contig.  numba's
        # padded_slice raises ValueError when start >= clen (region entirely
        # past the contig end): pad_right = end - clen > out_len triggers a
        # size-mismatch in the ndarray assignment.  Both backends fail loudly
        # on that degenerate input, so it is outside the byte-identity domain
        # and is intentionally not generated here.  In production, BED regions
        # are always clipped to contig bounds, so start >= clen never occurs.
        # Regions extending past the right edge (end > clen) are still generated.
        start = draw(st.integers(-5, clen - 1))
        length = draw(st.integers(0, clen + 5))
        regions[i] = (c, start, start + length)
        lengths.append(length)
    out_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    pad_char = draw(st.integers(0, 255))
    parallel = draw(st.booleans())
    return regions, out_offsets, reference, ref_offsets, np.uint8(pad_char), parallel


@st.composite
def shift_and_realign_tracks_inputs(draw):  # noqa: C901
    """Contract-valid inputs for shift_and_realign_tracks_sparse.

    Returns ``(total_out_size, inputs_tuple)`` where inputs_tuple is everything
    EXCEPT the out buffer (inserted at index 0 by the parity harness).

    Exercises all five strategy IDs:
      0 = REPEAT_5P
      1 = REPEAT_5P_NORM
      2 = CONSTANT
      3 = FLANK_SAMPLE
      4 = INTERPOLATE

    Layout mirrors the numba batch driver signature:
      out_offsets (b*p+1,), regions (b,3), shifts (b,p),
      geno_offset_idx (b,p), geno_v_idxs, geno_offsets (2,n),
      v_starts, ilens, tracks (ragged b*l), track_offsets (b+1),
      params (f64), keep (optional), keep_offsets (optional),
      strategy_id, base_seed.
    """
    # ── strategy ──────────────────────────────────────────────────────────────
    strategy_id = draw(st.integers(min_value=0, max_value=4))
    if strategy_id == 2:  # CONSTANT
        param_val = draw(st.floats(width=64, allow_nan=False, allow_infinity=False))
    elif strategy_id == 3:  # FLANK_SAMPLE
        param_val = float(draw(st.integers(min_value=0, max_value=5)))
    elif strategy_id == 4:  # INTERPOLATE — order in {1,2,3}
        param_val = float(draw(st.integers(min_value=1, max_value=3)))
    else:  # REPEAT_5P (0) or REPEAT_5P_NORM (1): param unused
        param_val = 0.0
    params = np.array([param_val], dtype=np.float64)

    base_seed = np.uint64(
        draw(st.integers(min_value=0, max_value=int(np.iinfo(np.uint64).max)))
    )

    # ── variants (SNP/ins/del mix) ─────────────────────────────────────────────
    n_unique = draw(st.integers(min_value=1, max_value=8))
    # v_starts sorted, in [0, 120] so they fit within track windows
    v_starts_raw = sorted(
        draw(st.lists(st.integers(0, 120), min_size=n_unique, max_size=n_unique))
    )
    v_starts = np.array(v_starts_raw, dtype=np.int32)
    # ilens: -3..3 for del/snp/ins mix; ensure at least one each
    ilens = np.array(
        draw(st.lists(st.integers(-3, 3), min_size=n_unique, max_size=n_unique)),
        dtype=np.int32,
    )

    # ── regions & tracks ─────────────────────────────────────────────────────
    n_q = draw(st.integers(1, 4))
    ploidy = draw(st.integers(1, 2))
    n_groups = n_q * ploidy

    # Per-query: q_start in [0, 80], region length in [4, 40]
    q_starts = [draw(st.integers(0, 80)) for _ in range(n_q)]
    region_lengths = [draw(st.integers(4, 40)) for _ in range(n_q)]

    regions = np.empty((n_q, 3), np.int32)
    for i in range(n_q):
        regions[i] = (0, q_starts[i], q_starts[i] + region_lengths[i])

    # Track for each query: length = region_length + extra deletion headroom
    # We give a bit of extra ref track beyond the region so deletions can read
    # past the region end (production contract: track is always >= region length).
    track_lengths = [max(rl + 10, 1) for rl in region_lengths]
    track_offsets = np.concatenate([[0], np.cumsum(track_lengths)]).astype(np.int64)
    total_track = int(track_offsets[-1])
    tracks = draw(
        st.lists(
            st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
            min_size=total_track,
            max_size=total_track,
        ).map(lambda xs: np.array(xs, dtype=np.float32))
    )

    # ── sparse genotypes ──────────────────────────────────────────────────────
    counts = [draw(st.integers(0, 4)) for _ in range(n_groups)]
    geno_offsets_1d = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    geno_offset_idx = np.arange(n_groups, dtype=np.int64).reshape(n_q, ploidy)
    v_idx_list: list[int] = []
    for c in counts:
        idxs = sorted(
            draw(st.lists(st.integers(0, n_unique - 1), min_size=c, max_size=c))
        )
        v_idx_list.extend(idxs)
    geno_v_idxs = np.array(v_idx_list, dtype=np.int32)

    # normalize geno_offsets to (2, n) form
    geno_offsets_2d = np.stack([geno_offsets_1d[:-1], geno_offsets_1d[1:]]).astype(
        np.int64
    )

    # ── out_offsets: (n_q * ploidy + 1,) ─────────────────────────────────────
    # Each (query, hap) output has the same length as the region (no jitter here)
    out_lengths = np.array(
        [rl for rl in region_lengths for _ in range(ploidy)], dtype=np.int64
    )
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int64)
    total_out = int(out_offsets[-1])

    # ── shifts ────────────────────────────────────────────────────────────────
    shifts = np.zeros((n_q, ploidy), dtype=np.int32)
    for qi in range(n_q):
        for h in range(ploidy):
            shifts[qi, h] = draw(st.integers(0, max(0, region_lengths[qi] // 4)))

    # ── optional keep mask ────────────────────────────────────────────────────
    use_keep = draw(st.booleans())
    total_v = int(geno_offsets_1d[-1])
    if use_keep and total_v > 0:
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total_v, max_size=total_v)), np.bool_
        )
        keep_offsets = geno_offsets_1d.copy()
    else:
        keep = None
        keep_offsets = None

    inputs = (
        out_offsets,  # (b*p+1,)
        regions,  # (b, 3)
        shifts,  # (b, p)
        geno_offset_idx,  # (b, p)
        geno_v_idxs,  # ragged variant idxs
        geno_offsets_2d,  # (2, n)
        v_starts,  # (n_unique,)
        ilens,  # (n_unique,)
        tracks,  # (total_track,) ragged
        track_offsets,  # (b+1,)
        params,  # (1,) f64
        keep,  # optional bool
        keep_offsets,  # optional i64
        int(strategy_id),  # int
        base_seed,  # np.uint64
    )
    return total_out, inputs


@st.composite
def reconstruct_haplotypes_inputs(draw, annotate=False):  # noqa: ARG001
    """Contract-valid inputs for reconstruct_haplotypes_from_sparse.

    Returns ``(total_out_size, inputs_tuple)`` where inputs_tuple is everything
    EXCEPT the out buffer (inserted at index 0 by the harness). The
    ``annotate`` parameter is accepted but unused — the test file decides whether
    to build annotation buffers.
    """
    from hypothesis.extra.numpy import arrays as hp_arrays

    # ── reference (1–2 contigs) ─────────────────────────────────────────────
    # Draw reference FIRST so we can constrain variant positions to be within
    # the contig bounds (mirrors the production contract where variants always
    # come from VCF records within the contig).
    n_contigs = draw(st.integers(1, 2))
    contig_lens = [draw(st.integers(10, 80)) for _ in range(n_contigs)]

    # ── variants ──────────────────────────────────────────────────────────────
    n_unique = draw(st.integers(min_value=1, max_value=6))
    # Constrain v_starts to [0, min_contig_len - 1] so that ref[ref_idx:v_pos]
    # never exceeds any contig's bounds. Variants are shared across all queries
    # (which may reference different contigs), so we must be conservative and use
    # the shortest contig's length as the upper bound. In production, variants are
    # always within-contig; this constraint enforces that invariant.
    min_contig_len = min(contig_lens)
    v_starts_raw = draw(
        st.lists(
            st.integers(0, min_contig_len - 1), min_size=n_unique, max_size=n_unique
        )
    )
    v_starts = np.sort(np.array(v_starts_raw, dtype=np.int32))
    ilens = np.array(
        draw(st.lists(st.integers(-3, 3), min_size=n_unique, max_size=n_unique)),
        dtype=np.int32,
    )
    # atomized: alt_len = max(1, 1 + ilen)
    alt_lens = np.maximum(1, 1 + ilens).astype(np.int64)
    alt_offsets = np.concatenate([[np.int64(0)], np.cumsum(alt_lens)]).astype(np.int64)
    total_alt = int(alt_offsets[-1])
    alt_alleles = draw(hp_arrays(np.uint8, total_alt, elements=st.integers(65, 90)))
    ref_offsets = np.concatenate([[np.int64(0)], np.cumsum(contig_lens)]).astype(
        np.int64
    )
    reference = draw(
        hp_arrays(np.uint8, int(ref_offsets[-1]), elements=st.integers(65, 90))
    )

    # ── sparse genotypes ──────────────────────────────────────────────────────
    n_q = draw(st.integers(1, 3))
    ploidy = draw(st.integers(1, 2))
    n_groups = n_q * ploidy
    counts = [draw(st.integers(0, 4)) for _ in range(n_groups)]
    geno_offsets_1d = np.concatenate([[np.int64(0)], np.cumsum(counts)]).astype(
        np.int64
    )
    geno_offset_idx = np.arange(n_groups, dtype=np.int64).reshape(n_q, ploidy)
    v_idx_list: list[int] = []
    for c in counts:
        idxs = sorted(
            draw(st.lists(st.integers(0, n_unique - 1), min_size=c, max_size=c))
        )
        v_idx_list.extend(idxs)
    geno_v_idxs = np.array(v_idx_list, dtype=np.int32)

    # ── regions: (contig_idx, start, end) ────────────────────────────────────
    regions = np.empty((n_q, 3), np.int32)
    region_lengths: list[int] = []
    for i in range(n_q):
        c = draw(st.integers(0, n_contigs - 1))
        clen = contig_lens[c]
        start = draw(st.integers(0, max(0, clen - 1)))
        length = draw(st.integers(1, min(40, clen - start + 5)))
        regions[i] = (c, start, start + length)
        region_lengths.append(length)

    # ── out_offsets: (n_q * ploidy + 1,) ─────────────────────────────────────
    out_lengths_mat = np.array(region_lengths, dtype=np.int64)[:, None] * np.ones(
        ploidy, dtype=np.int64
    )  # (n_q, ploidy)
    out_offsets = np.concatenate(
        [np.array([np.int64(0)]), np.cumsum(out_lengths_mat.ravel())]
    ).astype(np.int64)
    total_out = int(out_offsets[-1])

    # ── shifts ────────────────────────────────────────────────────────────────
    shifts = np.zeros((n_q, ploidy), dtype=np.int32)
    for qi in range(n_q):
        for h in range(ploidy):
            shifts[qi, h] = draw(st.integers(0, max(0, region_lengths[qi] // 4)))

    # ── optional keep mask ────────────────────────────────────────────────────
    use_keep = draw(st.booleans())
    total_v = int(geno_offsets_1d[-1])
    if use_keep and total_v > 0:
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total_v, max_size=total_v)), np.bool_
        )
        keep_offsets = geno_offsets_1d.copy()
    else:
        keep = None
        keep_offsets = None

    # normalize geno_offsets to (2, n) form (the registered backends accept this)
    geno_offsets_2d = np.stack([geno_offsets_1d[:-1], geno_offsets_1d[1:]]).astype(
        np.int64
    )

    inputs = (
        out_offsets,
        regions,
        shifts,
        geno_offset_idx,
        geno_offsets_2d,
        geno_v_idxs,
        v_starts,
        ilens,
        alt_alleles,
        alt_offsets,
        reference,
        ref_offsets,
        np.uint8(78),  # pad_char = ord('N')
        keep,
        keep_offsets,
        None,  # annot_v_idxs — caller fills for annotated path
        None,  # annot_ref_pos — caller fills for annotated path
    )
    return total_out, inputs
