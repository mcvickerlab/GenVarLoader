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
