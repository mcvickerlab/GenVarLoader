import awkward as ak
import genvarloader as gvl
import numpy as np
import pytest

from genvarloader._flat import _Flat
from genvarloader._ragged import FlatIntervals, RaggedIntervals

_REASON_242 = (
    "mcvickerlab/GenVarLoader#242 — intervals_to_tracks itv.start<query_start "
    "contract violation; both backends; fix deferred to separate PR"
)


def _flat(data, offsets, dtype):
    return _Flat.from_offsets(
        np.asarray(data, dtype), (2, None), np.asarray(offsets, np.int64)
    )


def test_flat_intervals_to_ragged_roundtrip():
    # Two groups: group 0 has 1 interval, group 1 has 2 intervals.
    offsets = [0, 1, 3]
    fi = FlatIntervals(
        starts=_flat([10, 20, 30], offsets, np.int32),
        ends=_flat([15, 25, 35], offsets, np.int32),
        values=_flat([1.0, 2.0, 3.0], offsets, np.float32),
    )
    assert fi.shape == (2, None)
    ri = fi.to_ragged()
    assert isinstance(ri, RaggedIntervals)
    assert ak.to_list(ri.starts.to_ak()) == [[10], [20, 30]]
    assert ak.to_list(ri.ends.to_ak()) == [[15], [25, 35]]
    assert ak.to_list(ri.values.to_ak()) == [[1.0], [2.0, 3.0]]


def test_flat_intervals_public_export():
    assert gvl.FlatIntervals is FlatIntervals
    assert "FlatIntervals" in gvl.__all__


def test_flat_intervals_end_to_end_matches_ragged():
    ds = gvl.get_dummy_dataset()
    idx = ([0, 1], [0, 1])

    rag = ds.with_seqs(None).with_tracks(["read-depth"], kind="intervals")
    flat = (
        ds.with_seqs(None)
        .with_tracks(["read-depth"], kind="intervals")
        .with_output_format("flat")
    )

    ri = rag[idx]
    fi = flat[idx]

    assert type(fi).__name__ == "FlatIntervals"
    assert isinstance(ri, gvl.RaggedIntervals)

    back = fi.to_ragged()
    assert ak.to_list(back.starts.to_ak()) == ak.to_list(ri.starts.to_ak())
    assert ak.to_list(back.ends.to_ak()) == ak.to_list(ri.ends.to_ak())
    assert ak.to_list(back.values.to_ak()) == ak.to_list(ri.values.to_ak())


def test_flat_intervals_multi_track_matches_ragged():
    ds = gvl.get_dummy_dataset()
    idx = ([0, 1, 2], [0, 1, 2])
    names = ["read-depth", "annot"]

    ri = ds.with_seqs(None).with_tracks(names, kind="intervals")[idx]
    fi = (
        ds.with_seqs(None)
        .with_tracks(names, kind="intervals")
        .with_output_format("flat")[idx]
    )
    back = fi.to_ragged()
    # (batch, track, ~itv) C-order must match awkward concat order
    assert ak.to_list(back.starts.to_ak()) == ak.to_list(ri.starts.to_ak())
    assert ak.to_list(back.values.to_ak()) == ak.to_list(ri.values.to_ak())


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_flat_float_tracks_only_returns_flatragged():
    ds = gvl.get_dummy_dataset()
    flat = ds.with_seqs(None).with_tracks(["read-depth"]).with_output_format("flat")
    out = flat[[0, 1], [0, 1]]
    assert type(out).__name__ == "_Flat"  # FlatRagged
    # round-trips to the ragged track values
    rag = ds.with_seqs(None).with_tracks(["read-depth"])[[0, 1], [0, 1]]
    assert ak.to_list(out.to_ragged().to_ak()) == ak.to_list(rag.to_ak())


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_flat_haps_plus_tracks_returns_flat_pair():
    ds = gvl.get_dummy_dataset()
    flat = (
        ds.with_seqs("haplotypes")
        .with_tracks(["read-depth"])
        .with_output_format("flat")
    )
    seqs, tracks = flat[[0, 1], [0, 1]]
    assert type(seqs).__name__ == "_Flat"
    assert type(tracks).__name__ == "_Flat"
