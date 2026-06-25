"""Direct numba-vs-rust parity test for xorshift64 and hash4 PRNG primitives.

This is the highest-priority parity guard for the FlankSample fill strategy
(Tasks 8/9). If Rust and numba diverge by even one bit here, FlankSample output
will diverge downstream.

The Rust functions are exposed as DEBUG exports (`_debug_xorshift64`,
`_debug_hash4`) in the genvarloader extension module. These may be kept or
removed after Task 8/9 review.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import Rust debug exports from the compiled extension module.
from genvarloader.genvarloader import _debug_hash4 as _hash4_rust
from genvarloader.genvarloader import _debug_xorshift64 as _xorshift64_rust

# Import numba implementations from _tracks.py.  They are @nb.njit functions;
# calling them from Python forces a first-call JIT compile — that is expected.
from genvarloader._dataset._tracks import _hash4 as _hash4_numba
from genvarloader._dataset._tracks import _xorshift64 as _xorshift64_numba

pytestmark = pytest.mark.parity

UINT64_MAX = 2**64 - 1
uint64_strategy = st.integers(0, UINT64_MAX)


# ── xorshift64 ────────────────────────────────────────────────────────────────


@settings(max_examples=500, deadline=None)
@given(uint64_strategy)
def test_xorshift64_parity(x: int) -> None:
    """Rust xorshift64 must equal numba _xorshift64 for every uint64 input."""
    expected = int(_xorshift64_numba(np.uint64(x)))
    got = _xorshift64_rust(x)
    assert got == expected, f"xorshift64({x:#x}): rust={got:#x} numba={expected:#x}"


# ── hash4 ─────────────────────────────────────────────────────────────────────


@settings(max_examples=500, deadline=None)
@given(uint64_strategy, uint64_strategy, uint64_strategy, uint64_strategy)
def test_hash4_parity(a: int, b: int, c: int, d: int) -> None:
    """Rust hash4 must equal numba _hash4 for every (a,b,c,d) uint64 quadruple.

    Passes np.uint64 args to numba so it uses uint64 semantics (wrapping
    arithmetic); compares against Python int() of the result to avoid any
    uint64 vs Python-int comparison issues.
    """
    expected = int(_hash4_numba(np.uint64(a), np.uint64(b), np.uint64(c), np.uint64(d)))
    got = _hash4_rust(a, b, c, d)
    assert got == expected, (
        f"hash4({a:#x}, {b:#x}, {c:#x}, {d:#x}): rust={got:#x} numba={expected:#x}"
    )


# ── smoke: fixed known vectors ─────────────────────────────────────────────────


def test_xorshift64_known_vectors() -> None:
    """Smoke-test a few hand-verified xorshift64 outputs."""
    assert _xorshift64_rust(1) == 1_082_269_761
    assert _xorshift64_rust(2) == 2_164_539_522
    assert _xorshift64_rust(42) == 45_454_805_674
    assert _xorshift64_rust(0xDEADBEEF) == 4_018_790_486_776_397_394
    assert _xorshift64_rust(UINT64_MAX) == 1_065_361_344


def test_hash4_known_vectors() -> None:
    """Smoke-test a few hand-verified hash4 outputs."""
    assert _hash4_rust(1, 2, 3, 4) == 11_323_120_931_611_735_037
    assert _hash4_rust(0, 0, 0, 0) == 0
    assert _hash4_rust(0xDEADBEEF, 0xCAFE, 0xBABE, 1) == 5_244_362_157_944_750_963
