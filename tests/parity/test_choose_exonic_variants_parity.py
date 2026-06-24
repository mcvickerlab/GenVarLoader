import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _genotypes  # noqa: F401
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import choose_exonic_variants_inputs

pytestmark = pytest.mark.parity


@given(choose_exonic_variants_inputs())
@settings(deadline=None)
def test_choose_exonic_variants_parity(inputs):
    qs, qe, goi, gvi, offsets, vs, ilens = inputs
    norm = (
        np.ascontiguousarray(qs, np.int32),
        np.ascontiguousarray(qe, np.int32),
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(vs, np.int32),
        np.ascontiguousarray(ilens, np.int32),
    )
    assert_kernel_parity_tuple("choose_exonic_variants", *norm)
