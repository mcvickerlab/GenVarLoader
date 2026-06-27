"""Direct rust parity test for xorshift64 and hash4 PRNG primitives.

Known-vector tests run directly against the Rust debug exports.  The
hypothesis-driven numba-comparison tests have been replaced with frozen-golden
replay (goldens generated in generate_goldens.py, cross-checked against numba at
generation time).

The Rust functions are exposed as DEBUG exports (`_debug_xorshift64`,
`_debug_hash4`) in the genvarloader extension module.
"""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader.genvarloader import _debug_hash4 as _hash4_rust
from genvarloader.genvarloader import _debug_xorshift64 as _xorshift64_rust
from tests.parity import _golden

pytestmark = pytest.mark.parity

UINT64_MAX = 2**64 - 1


# ── frozen-golden replay ───────────────────────────────────────────────────────


def test_xorshift64_golden():
    """Rust xorshift64 must equal the frozen golden (cross-checked vs numba at freeze time)."""
    cases = _golden.load_golden("prng_xorshift64")
    assert cases, "empty golden"
    for ci, (inputs, golden) in enumerate(cases):
        (x,) = inputs
        got = np.uint64(_xorshift64_rust(int(x)))
        exp = np.uint64(golden)
        assert got == exp, (
            f"xorshift64 case {ci}: input={x:#x} got={got:#x} exp={exp:#x}"
        )


def test_hash4_golden():
    """Rust hash4 must equal the frozen golden (cross-checked vs numba at freeze time)."""
    cases = _golden.load_golden("prng_hash4")
    assert cases, "empty golden"
    for ci, (inputs, golden) in enumerate(cases):
        a, b, c, d = inputs
        got = np.uint64(_hash4_rust(int(a), int(b), int(c), int(d)))
        exp = np.uint64(golden)
        assert got == exp, (
            f"hash4 case {ci}: ({a:#x},{b:#x},{c:#x},{d:#x}) got={got:#x} exp={exp:#x}"
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
