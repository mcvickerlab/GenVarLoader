import numpy as np
import pytest

from genvarloader import _dispatch
from tests.parity._harness import assert_kernel_parity_tuple

pytestmark = pytest.mark.parity


def test_tuple_helper_detects_match(monkeypatch):
    def impl(x):
        return x * 2, x + 1

    _dispatch.register("_tuple_smoke", numba=impl, rust=impl, default="rust")
    assert_kernel_parity_tuple("_tuple_smoke", np.arange(4, dtype=np.int32))


def test_tuple_helper_detects_mismatch():
    def a(x):
        return x, x

    def b(x):
        return x, x + 1

    _dispatch.register("_tuple_smoke_bad", numba=a, rust=b, default="rust")
    with pytest.raises(AssertionError):
        assert_kernel_parity_tuple("_tuple_smoke_bad", np.arange(4, dtype=np.int32))
