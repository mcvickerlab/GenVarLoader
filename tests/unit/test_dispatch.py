import pytest
from genvarloader import _dispatch


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    # Isolate each test: fresh registry + no inherited GVL_BACKEND.
    monkeypatch.setattr(_dispatch, "_REGISTRY", {})
    monkeypatch.delenv("GVL_BACKEND", raising=False)
    yield


def _reg():
    _dispatch.register("k", numba=lambda: "numba", rust=lambda: "rust", default="numba")


def test_get_returns_default_backend():
    _reg()
    assert _dispatch.get("k")() == "numba"


def test_get_respects_per_kernel_rust_default():
    _dispatch.register("k", numba=lambda: "n", rust=lambda: "r", default="rust")
    assert _dispatch.get("k")() == "r"


def test_env_override_forces_all_kernels(monkeypatch):
    _reg()
    monkeypatch.setenv("GVL_BACKEND", "rust")
    assert _dispatch.get("k")() == "rust"


def test_backends_returns_both_regardless_of_default():
    _reg()
    numba_fn, rust_fn = _dispatch.backends("k")
    assert numba_fn() == "numba" and rust_fn() == "rust"


def test_unknown_name_raises_keyerror_listing_names():
    _reg()
    with pytest.raises(KeyError, match="k"):
        _dispatch.get("missing")


def test_invalid_env_backend_raises(monkeypatch):
    _reg()
    monkeypatch.setenv("GVL_BACKEND", "julia")
    with pytest.raises(ValueError, match="GVL_BACKEND"):
        _dispatch.get("k")
