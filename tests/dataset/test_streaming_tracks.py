"""Smoke test for the interval-streaming parity fixture (issue #279 Task 2).

Confirms `streaming_tracks_fixture` (see `conftest.py`) actually constructs and
that its two tracks -- passed to `gvl.write` in non-alphabetical order
(``[zeta, alpha]``) -- land on a name-sorted track axis, since every later
parity test (Tasks 5-8) depends on that ordering.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._track_stream import _TrackBackend


def test_fixture_builds(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    ds = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
    assert list(ds.available_tracks) == ["alpha", "zeta"], (
        "written track axis must be name-sorted, not tracks= argument order"
    )


def _backend(f, tracks):
    """Build a `_TrackBackend` over the fixture's regions.

    Take `_regions` off a constructed `StreamingDataset` rather than calling
    `bed_to_regions` directly: that helper is
    `bed_to_regions(bed: pl.DataFrame, contig_norm: ContigNormalizer)` and
    returns ONE `(n_regions, 4)` array, not a `(regions, sort_order)` pair,
    so hand-rolling the call means also hand-rolling a `ContigNormalizer`.
    Reading the attribute keeps the test honest about what production builds.

    Uses the VARIANTS-only constructor, which already works today. Do not use
    `tracks=` here: that argument does not exist until Task 5, and this task
    must be testable on its own.
    """
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    return _TrackBackend(tracks, sds._regions, list(sds.contigs), list(sds.samples))


def test_names_are_sorted_not_argument_order(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    b = _backend(f, [f.table, f.bigwigs])  # zeta, alpha
    assert b.names == ["alpha", "zeta"]


def test_duplicate_track_names_raise(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _backend(f, [f.bigwigs, f.bigwigs])


def test_missing_sample_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    with pytest.raises(ValueError, match="not present"):
        _TrackBackend(
            [f.bigwigs],
            sds._regions,
            list(sds.contigs),
            [*f.samples, "no_such_sample"],
        )


def test_read_window_shape(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    b = _backend(f, [f.bigwigs])
    r_idx = np.array([0, 1], dtype=np.intp)
    s_idx = np.arange(len(f.samples), dtype=np.intp)
    (itvs,) = b.read_window(r_idx, s_idx)
    assert itvs.shape[:2] == (len(r_idx), len(s_idx))


def test_iteration_order_rejects_bad_value(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    with pytest.raises(ValueError, match="iteration_order"):
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order="sideways",
        )


def test_both_orders_visit_the_same_windows(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    made = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        # Force the sample axis to chunk; otherwise both orders are trivially
        # identical (spec §4.4).
        object.__setattr__(sds, "_window_samples", 1)
        made[order] = sorted(
            (tuple(r.tolist()), tuple(s.tolist())) for r, s in sds._plan()
        )
    assert made["regions"] == made["samples"]


def test_orders_differ_in_sequence_when_samples_chunk(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    seqs = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        object.__setattr__(sds, "_window_samples", 1)
        object.__setattr__(sds, "_window_regions", 1)
        seqs[order] = [(tuple(r.tolist()), tuple(s.tolist())) for r, s in sds._plan()]
    assert seqs["regions"] != seqs["samples"], (
        "with both axes chunked the visit ORDER must differ"
    )


def test_auto_resolves_to_regions_for_variants_only(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    assert sds._iteration_order == "regions"


def test_n_batches_is_order_invariant(svar1_multicontig_fixture):
    """Spec §7.3: the batch-span multiset is identical between orders."""
    f = svar1_multicontig_fixture
    counts = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        object.__setattr__(sds, "_window_samples", 1)
        object.__setattr__(sds, "_window_regions", 1)
        counts[order] = sds.n_batches(3)
    assert counts["regions"] == counts["samples"]


def test_return_indices_are_original_bed_rows(svar1_multicontig_fixture):
    """Spec §7.2: returned region indices are BED-row order, not sweep order."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    n_regions, n_samples = sds.shape
    seen = set()
    for _data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        seen.update(zip(map(int, r_idx), map(int, s_idx)))
    assert seen == {(r, s) for r in range(n_regions) for s in range(n_samples)}


def test_tracks_only_constructs(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    assert sds.shape == (len(f.bed), len(f.samples))
    assert sds.samples == sorted(f.samples)


def test_tracks_only_auto_is_sample_major(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    assert sds._iteration_order == "samples"


def test_mixed_auto_is_region_major(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    )
    assert sds._iteration_order == "regions"


def test_no_sources_still_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    with pytest.raises(ValueError, match="variants|tracks"):
        gvl.StreamingDataset(f.bed)


def test_with_seqs_on_tracks_only_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    with pytest.raises(ValueError, match="no variant source"):
        sds.with_seqs("haplotypes")


def test_tracks_only_samples_are_track_intersection(streaming_tracks_fixture):
    """Section 8 "strict superset" row: sample identity is the INTERSECTION

    across every track, not the union or either track's raw list.
    `bigwigs_superset` deliberately carries one extra sample
    (`zz_extra_sample`) that `table` does not; with two tracks passed, only
    a real intersection (not a union, and not "just the first track's
    samples") produces the right answer, so this is the coverage a
    single-track test cannot provide.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=[f.bigwigs_superset, f.table])
    assert sds.samples == sorted(f.samples)
    assert "zz_extra_sample" not in sds.samples


def _plan_seq(sds) -> list[tuple[int, int]]:
    """Flatten `_plan()` into the same per-cell `(bed_row, sample_idx)`

    sequence `to_iter()` emits (see `_iter_batches`'s `flat_r`/`flat_s`
    construction: `np.repeat(sort_order[r_idx], n_s)` /
    `np.tile(s_idx, len(r_idx))`, i.e. r outer, s inner).
    """
    seq = []
    for r_idx, s_idx in sds._plan():
        for r in r_idx:
            for s in s_idx:
                seq.append((int(sds._sort_order[r]), int(s)))
    return seq


def _to_iter_seq(sds) -> list[tuple[int, int]]:
    seq = []
    for _data, r_idx, s_idx in sds.to_iter(batch_size=1, return_indices=True):
        seq.extend(zip(map(int, r_idx), map(int, s_idx)))
    return seq


def test_mixed_iteration_order_samples_drives_to_iter_end_to_end(
    streaming_tracks_fixture,
):
    """Close the Task 4 carry-forward gap: `iteration_order="samples"` must be

    exercised through the real drive (`_iter_batches`/`to_iter`), not only
    through `_plan()`. A cell-SET check alone cannot distinguish "samples"
    from "regions" -- both orders visit the same set by construction
    (`test_both_orders_visit_the_same_windows`, Task 4) -- so this asserts
    on the emitted SEQUENCE: each order's real `to_iter()` output must match
    what `_plan()` alone predicts for that SAME order (proving the real
    engine drive genuinely honors `_iteration_order`, not just `_plan()` in
    isolation or a drive that silently ignores it), and the two orders'
    sequences must differ from each other while visiting the identical cell
    set. Both axes are forced to chunk (`_window_samples`=`_window_regions`=1`)
    so the two orders are guaranteed to diverge in sequence (mirrors
    `test_orders_differ_in_sequence_when_samples_chunk`, Task 4).
    """
    f = streaming_tracks_fixture

    def _make(order):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            tracks=f.bigwigs,
            iteration_order=order,
        )
        object.__setattr__(sds, "_window_samples", 1)
        object.__setattr__(sds, "_window_regions", 1)
        return sds

    samples_sds = _make("samples")
    regions_sds = _make("regions")
    assert samples_sds._iteration_order == "samples"
    assert regions_sds._iteration_order == "regions"

    samples_actual = _to_iter_seq(samples_sds)
    regions_actual = _to_iter_seq(regions_sds)

    # The real to_iter() drive matches what _plan() predicts for its OWN
    # order -- the drive isn't silently reordering or ignoring the plan.
    assert samples_actual == _plan_seq(samples_sds)
    assert regions_actual == _plan_seq(regions_sds)

    # Both orders visit the same SET of cells...
    n_regions, n_samples = samples_sds.shape
    full = {(r, s) for r in range(n_regions) for s in range(n_samples)}
    assert set(samples_actual) == set(regions_actual) == full

    # ...but the real drive's emitted SEQUENCE genuinely differs by order --
    # the assertion a cell-set-only check (or an ignored `_iteration_order`)
    # would fail to catch.
    assert samples_actual != regions_actual


def test_tracks_only_parity(streaming_tracks_fixture):
    """Tracks WITHOUT variants: intervals_to_tracks, no realignment."""
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=[f.table, f.bigwigs])
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs(None)
        .with_settings(realign_tracks=False)
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            streamed = data[i]
            expected = written[r, s]
            assert streamed.shape[0] == 2, "track axis is never squeezed"
            np.testing.assert_array_equal(np.asarray(streamed), np.asarray(expected))
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_single_track_keeps_its_axis(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    data, _r, _s = next(iter(sds.to_iter(batch_size=1, return_indices=True)))
    assert data[0].shape[0] == 1, "one track must still be (1, ...), not squeezed"


def test_tracks_only_superset_matches_by_name(streaming_tracks_fixture):
    """Arms the positional-sample-matching trap.

    `bigwigs_superset` has an extra sample (`zz_extra_sample`) inserted
    BEFORE the dataset's real samples in dict-insertion order. If the
    tracks-only drive ever matched samples by position instead of by name,
    every real sample's values would silently shift by one. Restrict the
    written side to just `alpha` (`bigwigs_superset` only carries that
    track) so the comparison is apples-to-apples.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs_superset)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs(None)
        .with_settings(realign_tracks=False)
        .with_tracks("alpha")
    )

    # The streaming side legitimately carries the superset's OWN samples --
    # a single track's "intersection" is itself -- so it has one more sample
    # than the written dataset, and at a different position. Translate through
    # names, never positions: that is precisely the property under test.
    written_pos = {name: i for i, name in enumerate(written.samples)}
    assert "zz_extra_sample" in sds.samples, (
        "the superset's extra sample must survive tracks-only construction"
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            name = sds.samples[s]
            if name not in written_pos:
                continue
            streamed = data[i]
            expected = written[r, written_pos[name]]
            np.testing.assert_array_equal(np.asarray(streamed), np.asarray(expected))
            seen.add((r, written_pos[name]))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("length", [5, 20])
def test_tracks_only_with_len_matches_written(streaming_tracks_fixture, length):
    """`with_len(L)` must agree with the written path cell-for-cell.

    Covers the fixed-`output_length` branch of the tracks-only drive, which the
    ragged-default tests never reach. Two lengths on purpose: one SHORTER than
    the regions (exercising truncation) and one EQUAL to them (the boundary).

    A length LONGER than the region is deliberately not tested: the written
    path refuses it outright (`ValueError: Jitter-expanded output length ...`)
    because this fixture writes with `extend_to_length=False, max_jitter=None`
    and so holds no padding to serve it from. There is therefore no oracle to
    compare against, and no divergence to detect.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=[f.table, f.bigwigs]).with_len(length)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs(None)
        .with_settings(realign_tracks=False)
        .with_len(length)
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            streamed = np.asarray(data[i])
            assert streamed.shape[-1] == length, "with_len must fix the last axis"
            np.testing.assert_array_equal(streamed, np.asarray(written[r, s]))
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_read_window_rejects_empty_regions(streaming_tracks_fixture):
    """The empty-`r_idx` guard raises a clear error, not an opaque IndexError.

    `_plan()` never yields an empty window, so this is only reachable by calling
    `read_window` directly -- which is exactly why it needs its own test rather
    than riding along on the parity tests.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    tb = sds._track_backend
    assert tb is not None
    with pytest.raises(ValueError, match="r_idx is empty"):
        tb.read_window(np.empty(0, np.intp), np.array([0], np.intp))


def test_tracks_only_rejects_jitter(streaming_tracks_fixture):
    """Jitter on a tracks-only dataset fails fast instead of reading unjittered.

    There is no variant engine to derive translated bounds from, and
    `read_window`'s contract requires the caller to supply them, so silently
    reading the un-translated bounds would emit wrong data.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs).with_settings(jitter=1)
    with pytest.raises(NotImplementedError, match="jitter"):
        next(iter(sds.to_iter(batch_size=1)))


# --- Issue #279 Task 7: mixed SVAR1 variants + re-aligned tracks -----------


def _assert_cell_equal(streamed, expected, ctx=""):
    """Assert one ``(region, sample)`` cell is byte-identical to the oracle.

    Compares a streamed cell against ``Dataset[r, s]``'s tracks half through
    the Ragged's own ``shape`` / ``lengths`` / packed values, NOT by indexing
    down to leaves. Two accessor traps make the obvious spelling wrong:

    1. Chained integer indexing (``cell[t][h]``) does NOT return a sub-Ragged.
       seqpro CONCATENATES the indexed group, so ``cell[0]`` on a
       ``(t, p, None)`` Ragged yields one flat array holding BOTH haplotypes
       and ``cell[0][0]`` collapses to a 0-d scalar -- comparing a scalar
       against an 18-element array and reporting a value diff for what is
       really a bad accessor.
    2. ``.data`` on a cell sliced out of a batch is the WHOLE batch's backing
       buffer, not the cell's slice of it, so its length is the batch total
       (80) rather than the cell's (38) even when ``.lengths`` already agrees.
       ``.to_packed()`` trims it to just this cell's values.

    ``with_len(L)`` returns a dense ndarray rather than a Ragged, so handle
    that shape-only case separately instead of demanding a ``.lengths``.

    The ragged assertions are ordered so the most diagnostic one fails first,
    matching the spec's three required parity axes:

    1. ``shape[:-1]`` -- rank plus the track and ploidy sizes. A spurious
       squeeze or a missing ploidy axis fails HERE, as a shape error, rather
       than silently broadcasting into a value diff.
    2. ``lengths`` -- per-(track, hap) output length. A mismatch is the
       signature of the extend_to_length span bug and must fail loudly and
       separately from a value diff.
    3. packed values -- the flat values in ``(track, ploidy)`` order. This is
       the byte oracle, and comparing the FLAT buffer is what makes the
       assertion sensitive to the ``(t, b, p)`` vs ``(b, t, p)`` assembly
       ordering: a mis-ordered buffer has identical shape and identical
       lengths, and differs only here.
    """
    if not hasattr(streamed, "lengths") or not hasattr(expected, "lengths"):
        # `with_len(L)` yields dense arrays on both sides; shape carries the
        # track and ploidy axes directly, so one comparison covers everything.
        s, e = np.asarray(streamed), np.asarray(expected)
        assert s.shape == e.shape, f"{ctx}shape {s.shape} != oracle {e.shape}"
        np.testing.assert_array_equal(s, e, err_msg=f"{ctx}track values differ")
        return

    s_shape, e_shape = streamed.shape[:-1], expected.shape[:-1]
    assert s_shape == e_shape, f"{ctx}shape {s_shape} != oracle {e_shape}"
    np.testing.assert_array_equal(
        np.asarray(streamed.lengths),
        np.asarray(expected.lengths),
        err_msg=f"{ctx}per-(track, hap) lengths differ",
    )
    np.testing.assert_array_equal(
        np.asarray(streamed.to_packed().data),
        np.asarray(expected.to_packed().data),
        err_msg=f"{ctx}track values differ",
    )


def test_mixed_parity_with_indels(streaming_tracks_fixture):
    """Tracks re-aligned to haplotype coordinates -- the case #279 calls out.

    Two tracks, passed NON-alphabetically (``[table, bigwigs]`` = ``[zeta,
    alpha]``) so the name-sorted track-axis rule is actually under test: an
    alphabetical fixture cannot distinguish "sorted by name" from "argument
    order". Compared as a SET of cells, since streaming's iteration order
    differs from the written dataset's index order by design.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar_path,
        tracks=[f.table, f.bigwigs],
    )
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        # The track axis must never be squeezed, and must carry BOTH tracks.
        assert data.shape[1] == 2, f"track axis {data.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            # `written[r, s]` is a `(haps, tracks)` 2-tuple whenever seqs are
            # active; streaming's mixed convention yields bare tracks, so only
            # the tracks half of the oracle is comparable.
            _assert_cell_equal(data[i], written[r, s][1], ctx=f"cell (r={r}, s={s}): ")
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_fixed_output_length_parity(streaming_tracks_fixture):
    """``with_len(L)`` parity as well as ``with_len("ragged")``.

    Deviation from the brief (which used ``L = 64``): every region in
    ``streaming_tracks_fixture`` is exactly 20bp and the fixture is written
    with ``extend_to_length=False, max_jitter=None``, so the WRITTEN oracle's
    own ``with_len`` guard rejects any ``L > 20`` before streaming is even
    involved -- ``L = 64`` raises on ``written.with_len(L)`` itself, so no
    oracle exists for it. Use a value comfortably under 20 instead. Task 6 hit
    and settled the same limit for the tracks-only ``with_len`` test.
    """
    f = streaming_tracks_fixture
    L = 12
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_len(L)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_tracks("alpha")
        .with_len(L)
    )
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        r, s = int(r_idx[i]), int(s_idx[i])
        _assert_cell_equal(data[i], written[r, s][1], ctx=f"cell (r={r}, s={s}): ")


def test_non_default_insertion_fill_parity(streaming_tracks_fixture):
    """A non-default insertion fill, not just the default ``Repeat5p()``.

    ``insertion_fill`` is a byte-level input to the fused kernel -- it selects
    what a realigned track emits across an insertion -- so parity must hold for
    a strategy other than the default.
    """
    f = streaming_tracks_fixture
    # The exported strategies are InsertionFill, Repeat5p (the default) and
    # Repeat5pNormalized; pick a non-default one.
    fill = gvl.Repeat5pNormalized()
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_insertion_fill(fill)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_tracks("alpha")
        .with_insertion_fill(fill)
    )
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        r, s = int(r_idx[i]), int(s_idx[i])
        _assert_cell_equal(data[i], written[r, s][1], ctx=f"cell (r={r}, s={s}): ")


def test_track_with_superset_samples_parity(streaming_tracks_fixture):
    """A track whose samples strictly contain the dataset's.

    Extra track samples must be ignored, and the ones that remain must still be
    matched by NAME -- a positional index would silently shift every sample.
    ``written`` is narrowed to ``alpha`` because ``bigwigs_superset`` carries
    only that track while the written dataset holds both.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar_path,
        tracks=f.bigwigs_superset,
    )
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_tracks(
        "alpha"
    )
    # Translate through sample NAMES: `sds` has the superset's samples while
    # `written` has fewer, so a positional index would run off the end (and,
    # worse, silently compare the wrong sample where it did not).
    written_pos = {name: i for i, name in enumerate(written.samples)}
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        name = sds.samples[int(s_idx[i])]
        if name not in written_pos:
            continue
        r, w_s = int(r_idx[i]), written_pos[name]
        _assert_cell_equal(
            data[i], written[r, w_s][1], ctx=f"cell (r={r}, sample={name!r}): "
        )


def test_realign_false_drops_ploidy_axis(streaming_tracks_fixture):
    """``realign_tracks=False`` keeps reference coordinates, dropping ploidy.

    Deviation from the brief: the written oracle keeps BOTH ``alpha`` and
    ``zeta`` active (it is opened with no ``.with_tracks(...)`` narrowing)
    while ``sds`` only requests ``tracks=f.bigwigs`` (``alpha`` alone), so the
    raw arrays would be a 1-track streamed output against a 2-track written
    one. Narrow ``written`` to ``alpha`` to match.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=False)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_tracks("alpha")
        .with_settings(realign_tracks=False)
    )
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=1, return_indices=True)))
    r, s = int(r_idx[0]), int(s_idx[0])
    expected = written[r, s][1]
    # The point of the test: no ploidy axis on either side.
    assert data[0].shape[:-1] == expected.shape[:-1]
    _assert_cell_equal(data[0], expected, ctx=f"cell (r={r}, s={s}): ")


@pytest.mark.parametrize("src", ["vcf", "pgen"])
def test_mixed_tracks_non_svar1_raises(streaming_case, src):
    """Mixed variants+tracks is SVAR1-ONLY in v1.

    Combining ``tracks=`` with a VCF or PGEN variant source must raise
    ``NotImplementedError`` at ``to_iter`` time, not silently ignore the tracks
    or produce wrong output. Parametrized across both non-SVAR1 backends
    ``streaming_case`` supports (SVAR2 has its own fixture shape -- see
    ``test_mixed_tracks_svar2_raises`` -- and SVAR1 is the one backend this
    guard must NOT fire for, exercised by the parity tests above).

    Sample ids come from the written oracle's own ``samples`` (not a
    hand-typed guess) so the ``Table`` construction here can't drift out of
    sync with whichever fixture ``src`` selects.
    """
    bed, reference, variants, written = streaming_case(src)
    samples = list(written.samples)
    table = gvl.Table(
        "t",
        pl.DataFrame(
            {
                "sample_id": samples,
                "chrom": ["chr1"] * len(samples),
                "start": [0] * len(samples),
                "end": [10] * len(samples),
                "value": [float(i) for i in range(len(samples))],
            }
        ),
    )
    sds = gvl.StreamingDataset(
        bed, reference=reference, variants=variants, tracks=table
    )
    with pytest.raises(NotImplementedError, match="SVAR1|\\.svar"):
        next(iter(sds.to_iter(batch_size=1)))


def test_mixed_tracks_svar2_raises(streaming_svar2_case):
    """Same guard, SVAR2 source.

    ``streaming_svar2_case`` returns ``(bed, reference, variants)`` -- three
    values, not ``streaming_case``'s four -- because it has no plain
    ``gvl.Dataset.open(...)`` oracle wired up (see its docstring in
    ``conftest.py``), so sample ids are hand-typed from the shared
    ``_SVAR1_MC_VCF`` fixture text (``S0``/``S1``/``S2``) that
    ``svar2_multicontig_fixture`` converts, rather than read off a `written`
    dataset.
    """
    bed, reference, variants = streaming_svar2_case
    # `svar2_multicontig_fixture`'s bed spans BOTH chr1 and chr2 (unlike the
    # vcf/pgen fixtures above, which are single-contig) -- `_TrackBackend`
    # validates contig coverage against every contig the bed references, so
    # the table must cover chr2 too or construction fails before the guard
    # under test ever runs.
    samples = ["S0", "S1", "S2"]
    table = gvl.Table(
        "t",
        pl.DataFrame(
            {
                "sample_id": samples * 2,
                "chrom": ["chr1"] * len(samples) + ["chr2"] * len(samples),
                "start": [0] * len(samples) * 2,
                "end": [10] * len(samples) * 2,
                "value": [float(i) for i in range(len(samples) * 2)],
            }
        ),
    )
    sds = gvl.StreamingDataset(
        bed, reference=reference, variants=variants, tracks=table
    )
    with pytest.raises(NotImplementedError, match="SVAR1|\\.svar"):
        next(iter(sds.to_iter(batch_size=1)))


def test_insertion_fill_without_tracks_raises(streaming_tracks_fixture):
    """Spec §3.1: `with_insertion_fill` on a track-less dataset is an error.

    The written path (`_impl.py:872-889`) rejects this; streaming must too.
    Silently returning a dataset with an empty fill map is the "accepts a
    setting that cannot change any byte" failure mode the rest of this class
    guards against.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    with pytest.raises(ValueError, match="requires tracks"):
        sds.with_insertion_fill(gvl.Repeat5pNormalized())


def test_insertion_fill_without_realign_raises(streaming_tracks_fixture):
    """Spec §3.1: insertion fill has no effect when `realign_tracks=False`.

    Insertion fill only applies while re-aligning tracks to haplotype
    coordinates, so accepting it with re-alignment off would silently no-op.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=False)
    with pytest.raises(ValueError, match="no effect when realign_tracks=False"):
        sds.with_insertion_fill(gvl.Repeat5pNormalized())


# --- Issue #279 Task 8: guards for unsupported track combinations ---------


def test_variant_windows_with_realigned_tracks_raises(streaming_tracks_fixture):
    """Spec §3.3: a REAL written-path `ValueError`, not a streaming gap.

    `with_seqs("variant-windows")` windows are reference-oriented; the written
    path's own `_build_reconstructor` (`_reconstruct.py:537-543`) refuses to
    re-align them for ANY backend, so streaming must raise the exact
    `ValueError` (not `NotImplementedError`) -- a caller mirroring the written
    path's error handling needs to catch the same exception type here.
    """
    f = streaming_tracks_fixture
    opt = gvl.VarWindowOpt(flank_length=2, token_alphabet=b"ACGT", unknown_token=4)
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    )
    with pytest.raises(ValueError, match="realign_tracks"):
        next(iter(sds.with_seqs("variant-windows", opt).to_iter(batch_size=1)))


def test_variant_windows_tracks_realign_false_raises_streaming_gap(
    streaming_tracks_fixture,
):
    """`with_seqs("variant-windows")` + tracks + `realign_tracks=False`.

    The written path SUPPORTS this exact combination (`_build_reconstructor`
    returns `SeqsTracks` when `realign_tracks` is `False`, per the same
    `_reconstruct.py:537-543` branch the previous test exercises the other
    side of); streaming just hasn't wired the fused kernel for anything but
    bare haplotype output, so this is `NotImplementedError` -- a genuinely
    different exception from the `ValueError` above, and it must stay that
    way (asserting `NotImplementedError` here catches a regression that
    widens the `ValueError` guard to also swallow this supported case).
    """
    f = streaming_tracks_fixture
    opt = gvl.VarWindowOpt(flank_length=2, token_alphabet=b"ACGT", unknown_token=4)
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=False)
    with pytest.raises(NotImplementedError, match="variant-windows"):
        next(iter(sds.with_seqs("variant-windows", opt).to_iter(batch_size=1)))


@pytest.mark.parametrize("realign_tracks", [True, False])
def test_variants_with_tracks_raises_streaming_gap(
    streaming_tracks_fixture, realign_tracks
):
    """`with_seqs("variants")` + tracks: NOT a written-path `ValueError`.

    Deviation from the Task 8 brief / design spec §3.3 table's row 2, recorded
    at the guard site in `_streaming.py` and here: verified empirically
    against the actual written `Dataset` (`Dataset.open(...).with_seqs(
    "variants")[r, s]` with tracks active and `realign_tracks=True` indexes
    successfully -- no `ValueError` exists in `_build_reconstructor` for this
    combination, for either `realign_tracks` value; `with_insertion_fill`'s
    own allow-list at `_impl.py:872` including `"variants"` is independent
    confirmation). So there is no written-path `ValueError` to mirror here;
    streaming just hasn't wired the fused kernel for anything but bare
    haplotype output, hence `NotImplementedError` for BOTH `realign_tracks`
    values, unlike the "variant-windows" pair of tests above where one side
    really is a written-path `ValueError`.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=realign_tracks)
    with pytest.raises(NotImplementedError, match="variants"):
        next(iter(sds.with_seqs("variants").to_iter(batch_size=1)))


@pytest.mark.parametrize("realign_tracks", [True, False])
def test_annotated_with_tracks_raises_streaming_gap(
    streaming_tracks_fixture, realign_tracks
):
    """`with_seqs("annotated")` + tracks: also unwired, also no written-path
    `ValueError` to mirror.

    The SVAR1 `HapsTracks.__call__` this streaming path fuses into has no
    annotated-specific guard at all (only the separate SVAR2 `_call_svar2`
    path rejects `RaggedAnnotatedHaps`, and that combination is unreachable
    here -- mixed tracks are SVAR1-only). So, like `"variants"` above, this is
    a pure streaming wiring gap: `NotImplementedError` for both
    `realign_tracks` values.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=realign_tracks)
    with pytest.raises(NotImplementedError, match="annotated"):
        next(iter(sds.with_seqs("annotated").to_iter(batch_size=1)))


def test_jitter_with_realigned_tracks_raises(streaming_tracks_fixture):
    """`jitter>0` + `realign_tracks=True` (the default) has no oracle.

    The deletion-extension track query always sizes itself off
    `_Svar1Backend.read_window`'s RAW, un-jittered region bounds, while the
    haplotype engine and the track query itself would use jitter-translated
    bounds -- silently under-extending the buffer under jitter. The guard for
    this already exists (`_streaming.py`'s realign_tracks branch just below
    the SVAR1-only check); this closes the Task 7 review's "currently
    untested" gap, not a new code change.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(jitter=1)
    with pytest.raises(NotImplementedError, match="jitter"):
        next(iter(sds.to_iter(batch_size=1)))


def test_jitter_with_unrealigned_tracks_produces_output(streaming_tracks_fixture):
    """`jitter>0` + `realign_tracks=False` + variants is a LIVE branch.

    Un-realigned tracks stay in reference coordinates, so they need no
    deletion-extension read-ahead and can safely reuse the SAME
    jitter-translated bounds the haplotype engine gets (`_streaming.py`'s
    `region_offsets is not None` branch feeding `t_starts`/`t_ends` when
    `realign_tracks` is `False`). This is deliberately NOT a byte-parity test
    -- jitter has no written oracle by design (see `to_iter`'s docstring) --
    it only asserts the combination actually produces output for every cell
    instead of raising, catching a regression that widens either jitter guard
    to also cover this supported branch.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(jitter=1, realign_tracks=False)
    n_cells = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        assert data.shape[0] == len(r_idx)
        n_cells += len(r_idx)
    n_regions, n_samples = sds.shape
    assert n_cells == n_regions * n_samples


def test_realign_false_with_len_matches_written(streaming_tracks_fixture):
    """`realign_tracks=False` + `with_len(L)` mixed-path parity.

    The one branch that substitutes a FIXED length for the region length
    (`_streaming.py`'s `isinstance(self._output_length, int)` check inside the
    `track_w.realign is None` arm) rather than the deletion-extended
    `out_lengths` the `realign_tracks=True` branch above it uses. Neither
    `test_realign_false_drops_ploidy_axis` (ragged, no `with_len`) nor
    `test_fixed_output_length_parity` (`realign_tracks=True`, the default)
    exercises this combination. `L=12` for the same reason as
    `test_fixed_output_length_parity`: the fixture's 20bp regions and
    `extend_to_length=False, max_jitter=None` writing mean the written
    oracle's own `with_len` guard has no room to serve anything bigger.
    """
    f = streaming_tracks_fixture
    L = 12
    sds = (
        gvl.StreamingDataset(
            f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
        )
        .with_settings(realign_tracks=False)
        .with_len(L)
    )
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_tracks("alpha")
        .with_settings(realign_tracks=False)
        .with_len(L)
    )
    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            _assert_cell_equal(data[i], written[r, s][1], ctx=f"cell (r={r}, s={s}): ")
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }
