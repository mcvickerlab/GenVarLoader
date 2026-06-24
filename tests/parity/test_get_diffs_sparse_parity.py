import pytest
from hypothesis import given, settings

from genvarloader._dataset import _genotypes  # noqa: F401  (import triggers register())
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import get_diffs_sparse_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None)
@given(get_diffs_sparse_inputs())
def test_get_diffs_sparse_parity(inputs):
    # The public wrapper normalizes offsets; here we call the registered
    # backends directly through the wrapper's dispatch name with the wrapper's
    # already-normalized (2, n) form, so feed normalized inputs.
    from genvarloader._dataset._genotypes import _as_starts_stops
    import numpy as np

    goi, gvi, offsets, ilens, keep, keep_off, qs, qe, vs = inputs
    norm = (
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(ilens, np.int32),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_off is None else np.ascontiguousarray(keep_off, np.int64),
        None if qs is None else np.ascontiguousarray(qs, np.int32),
        None if qe is None else np.ascontiguousarray(qe, np.int32),
        None if vs is None else np.ascontiguousarray(vs, np.int32),
    )
    assert_kernel_parity_tuple("get_diffs_sparse", *norm)
