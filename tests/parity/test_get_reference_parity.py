import pytest
from hypothesis import given, settings

from genvarloader._dataset import _reference  # noqa: F401  (triggers register())
from tests.parity._harness import assert_kernel_parity
from tests.parity.strategies import get_reference_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None)
@given(get_reference_inputs())
def test_get_reference_parity(inputs):
    regions, out_offsets, reference, ref_offsets, pad_char, parallel = inputs
    assert_kernel_parity(
        "get_reference", regions, out_offsets, reference, ref_offsets, pad_char, parallel
    )
