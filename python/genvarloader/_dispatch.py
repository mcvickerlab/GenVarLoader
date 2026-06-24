"""Backend dispatch registry for the Rust migration strangler window.

Each migratable Python-entry kernel registers a numba and a rust implementation.
Production code calls ``get(name)(...)``; ``GVL_BACKEND=numba|rust`` force-overrides
all kernels (used by CI parity sweeps). Deleted wholesale in migration Phase 5.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

_Backend = Literal["numba", "rust"]
_REGISTRY: dict[str, dict[str, object]] = {}


def register(
    name: str,
    *,
    numba: Callable,
    rust: Callable,
    default: _Backend = "numba",
) -> None:
    if default not in ("numba", "rust"):
        raise ValueError(f"default must be 'numba' or 'rust', got {default!r}")
    _REGISTRY[name] = {"numba": numba, "rust": rust, "default": default}


def _entry(name: str) -> dict[str, object]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no kernel registered as {name!r}; registered: {registered_names()}"
        ) from None


def get(name: str) -> Callable:
    entry = _entry(name)
    backend = os.environ.get("GVL_BACKEND")
    if backend is None:
        backend = entry["default"]  # type: ignore[assignment]
    elif backend not in ("numba", "rust"):
        raise ValueError(f"GVL_BACKEND must be 'numba' or 'rust', got {backend!r}")
    return entry[backend]  # type: ignore[return-value]


def backends(name: str) -> tuple[Callable, Callable]:
    entry = _entry(name)
    return entry["numba"], entry["rust"]  # type: ignore[return-value]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
