"""flat-mode output, re-wrapped via .to_ragged(), must be byte-identical to ragged mode."""
from __future__ import annotations

import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader._ragged import RaggedAnnotatedHaps

IDX = [0, (np.array([0, 1, 2]),)]  # scalar-ish and a list index


def _to_plain(obj):
    """Normalize a ragged/annot/flat object into dict of ndarrays for comparison."""
    if isinstance(obj, RaggedAnnotatedHaps):
        return {
            "haps": np.asarray(obj.haps.data), "haps_off": np.asarray(obj.haps.offsets),
            "vidx": np.asarray(obj.var_idxs.data), "pos": np.asarray(obj.ref_coords.data),
        }
    if isinstance(obj, Ragged):
        return {"data": np.asarray(obj.data), "off": np.asarray(obj.offsets)}
    raise TypeError(type(obj))


@pytest.mark.parametrize("seqs", ["haplotypes", "reference", "annotated"])
@pytest.mark.parametrize("idx", IDX)
def test_a0_flat_to_ragged_matches_ragged(snap_dataset, seqs, idx):
    ds = snap_dataset.with_seqs(seqs).with_tracks(False)
    ragged = ds[idx]
    flat = ds.with_output_format("flat")[idx]
    rewrapped = flat.to_ragged()
    r, f = _to_plain(ragged), _to_plain(rewrapped)
    assert r.keys() == f.keys()
    for k in r:
        np.testing.assert_array_equal(r[k], f[k], err_msg=f"field {k}")
