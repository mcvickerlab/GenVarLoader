"""Smoke test for the interval-streaming parity fixture (issue #279 Task 2).

Confirms `streaming_tracks_fixture` (see `conftest.py`) actually constructs and
that its two tracks -- passed to `gvl.write` in non-alphabetical order
(``[zeta, alpha]``) -- land on a name-sorted track axis, since every later
parity test (Tasks 5-8) depends on that ordering.
"""

from __future__ import annotations

import numpy as np
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
