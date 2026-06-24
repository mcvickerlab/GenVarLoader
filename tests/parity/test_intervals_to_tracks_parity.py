import numpy as np
import pytest
from hypothesis import given

from genvarloader._dataset import _intervals  # noqa: F401  (import triggers register())
from tests.parity._harness import assert_inplace_kernel_parity
from tests.parity.strategies import intervals_to_tracks_inputs

pytestmark = pytest.mark.parity


@given(intervals_to_tracks_inputs())
def test_intervals_to_tracks_parity(inputs):
    out_offsets = inputs[6]
    total = int(out_offsets[-1])
    # NaN sentinel: any position the kernel fails to zero/paint stays NaN and is caught.
    assert_inplace_kernel_parity(
        "intervals_to_tracks",
        inputs,
        out_factory=lambda: np.full(total, np.nan, np.float32),
        out_index=6,
    )
