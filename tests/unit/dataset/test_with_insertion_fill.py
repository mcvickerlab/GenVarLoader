"""Dataset-level ``with_insertion_fill`` API tests.

Single API rejection check extracted from
``tests/integration/dataset/test_dummy_dataset_insertion_fill.py``.
Uses ``gvl.get_dummy_dataset()`` (a gvl-shipped helper) rather than a
project toy-dataset path fixture; the two remaining tests in the
integration file exercise the full reconstruction call path and stay
in the integration tier.
"""

import genvarloader as gvl
import pytest
from genvarloader._dataset._insertion_fill import Repeat5p


def test_with_insertion_fill_rejects_when_no_tracks_active():
    """A dataset with tracks disabled should reject with_insertion_fill."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Disable tracks: view-state no longer has active tracks.
    ds_no_tracks = ds.with_tracks(False)
    with pytest.raises(ValueError, match="with_tracks"):
        ds_no_tracks.with_insertion_fill(Repeat5p())
