import numpy as np
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _flat_variants  # noqa: F401  (triggers register())
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import gather_rows_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None)
@given(gather_rows_inputs())
def test_gather_rows_parity(inputs):
    goi, offsets, data = inputs
    assert_kernel_parity_tuple(
        "gather_rows",
        np.ascontiguousarray(goi, np.int64),
        _as_starts_stops(offsets),
        np.ascontiguousarray(data, np.int32),
    )
