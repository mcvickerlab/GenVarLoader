import pytest
from hypothesis import given

from genvarloader._dataset import _utils  # noqa: F401  (import triggers register())
from tests.parity._harness import assert_kernel_parity
from tests.parity.strategies import splits_inputs

pytestmark = pytest.mark.parity


@given(splits_inputs())
def test_splits_sum_le_value_parity(inputs):
    arr, max_value = inputs
    assert_kernel_parity("splits_sum_le_value", arr, max_value)
