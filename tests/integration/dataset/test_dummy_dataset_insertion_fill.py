"""End-to-end insertion-fill tests requiring a real Dataset.

The unit-tier insertion-fill tests live in
``tests/unit/tracks/test_insertion_fill.py``. The tests here use
``gvl.get_dummy_dataset()`` and exercise the full
``with_insertion_fill`` → reconstruction call path.
"""

import math

import genvarloader as gvl
import pytest
from genvarloader._dataset._insertion_fill import Constant

_REASON_242 = (
    "mcvickerlab/GenVarLoader#242 — intervals_to_tracks itv.start<query_start "
    "contract violation; both backends; fix deferred to separate PR"
)


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_end_to_end_set_insertion_fill():
    """Use the dummy dataset to confirm with_insertion_fill plumbing works end-to-end."""
    ds = gvl.get_dummy_dataset()
    # Only haps+tracks datasets support insertion fill.
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    first_track = next(iter(ds._tracks.active_tracks))
    ds_nan = ds.with_insertion_fill({first_track: Constant(float("nan"))})
    assert isinstance(ds_nan._tracks.insertion_fill[first_track], Constant)
    assert math.isnan(ds_nan._tracks.insertion_fill[first_track].value)
    # Immutability: original dataset's insertion_fill is not mutated by the new dataset.
    assert first_track not in ds._tracks.insertion_fill
    # Trigger actual reconstruction to verify the full call path executes without error.
    _ = ds_nan[0, 0]


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_dummy_dataset_with_default_insertion_fill_does_not_crash():
    """Tracks created outside from_path may have empty insertion_fill — must not KeyError."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Just trigger reconstruction; the call must not raise KeyError.
    _ = ds[0, 0]
