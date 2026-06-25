import pytest

import genvarloader as gvl
from genvarloader._dataset._reconstruct import SeqsTracks
from genvarloader._flat import _Flat

_REASON_242 = (
    "mcvickerlab/GenVarLoader#242 — intervals_to_tracks itv.start<query_start "
    "contract violation; both backends; fix deferred to separate PR"
)


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_reference_plus_tracks_uses_seqstracks():
    ds = gvl.get_dummy_dataset()
    rt = ds.with_seqs("reference").with_tracks(["read-depth"])
    assert type(rt._recon) is SeqsTracks
    seqs, tracks = rt[[0, 1], [0, 1]]
    assert seqs.shape[0] == 2
    assert tracks.shape[0] == 2


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_reference_plus_tracks_flat_returns_flat_seqs():
    """with_output_format('flat') on reference+tracks yields FlatRagged seqs."""
    ds = gvl.get_dummy_dataset()
    rt = (
        ds.with_seqs("reference").with_tracks(["read-depth"]).with_output_format("flat")
    )
    seqs, tracks = rt[[0, 1], [0, 1]]
    assert isinstance(seqs, _Flat)
    assert isinstance(tracks, _Flat)
