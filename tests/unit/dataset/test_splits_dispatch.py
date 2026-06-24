import numpy as np
import pytest
from genvarloader._dataset._utils import splits_sum_le_value


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_wrapper_matches_known_result(backend, monkeypatch):
    monkeypatch.setenv("GVL_BACKEND", backend)
    out = splits_sum_le_value(np.array([5, 5, 11, 9, 2, 7]), 10)
    np.testing.assert_array_equal(out, np.array([0, 2, 3, 4, 6]))
    assert out.dtype == np.intp


def test_wrapper_is_registered():
    from genvarloader import _dispatch

    assert "splits_sum_le_value" in _dispatch.registered_names()
