import genvarloader as gvl
from genvarloader._dataset._reconstruct import SeqsTracks


def test_reference_plus_tracks_uses_seqstracks():
    ds = gvl.get_dummy_dataset()
    rt = ds.with_seqs("reference").with_tracks(["read-depth"])
    assert type(rt._recon) is SeqsTracks
    seqs, tracks = rt[[0, 1], [0, 1]]
    assert seqs.shape[0] == 2
    assert tracks.shape[0] == 2
