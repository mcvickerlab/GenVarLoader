"""Reference-mode dataset-level parity backstop.

Proves that flipping GVL_BACKEND (numba vs rust) produces byte-identical
reference-sequence output through the real Dataset.__getitem__ path — with a
spy guard proving the Rust get_reference kernel is actually invoked (no
vacuous pass).

Kernel exercised end-to-end:
  - get_reference  (reference fetch — dispatched via _dispatch.get in
                    _dataset/_reference.py:get_reference())

Spliced-reference note:
  The parity fixture (phased_svar_gvl) is not opened with splice_info, so the
  splice branch (_fetch_spliced_ref → get_reference) is NOT exercised here.
  However, _fetch_spliced_ref is plain Python that delegates its hot call to
  the dispatched get_reference (see _reference.py:759), so the same kernel
  dispatch entry point is covered.  A dedicated spliced fixture would require
  a GTF / transcript ID column that the current synthetic case does not
  provide; see the "Spliced coverage TODO" comment below.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._reference  # noqa: F401 — triggers register("get_reference")
import genvarloader._dispatch as _dispatch
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _compare_ragged_bytes(
    numba_out: Ragged, rust_out: Ragged, name: str = "reference"
) -> None:
    """Assert that two Ragged[np.bytes_] results are byte-identical.

    Compares both the flat character data buffer (uint8 / S1) and the
    per-row offsets.
    """
    n_data = np.asarray(numba_out.data)
    r_data = np.asarray(rust_out.data)
    assert n_data.dtype == r_data.dtype, (
        f"dtype mismatch for {name}: numba={n_data.dtype}, rust={r_data.dtype}"
    )
    np.testing.assert_array_equal(
        n_data,
        r_data,
        err_msg=f"sequence data differs across backends for '{name}'",
    )
    n_off = np.asarray(numba_out.offsets, dtype=np.int64)
    r_off = np.asarray(rust_out.offsets, dtype=np.int64)
    np.testing.assert_array_equal(
        n_off,
        r_off,
        err_msg=f"offsets differ across backends for '{name}'",
    )


# ---------------------------------------------------------------------------
# Main backstop test
# ---------------------------------------------------------------------------


def test_reference_mode_dataset_parity(phased_svar_gvl, reference, monkeypatch):
    """Flips GVL_BACKEND numba<->rust through the real reference getitem path.

    The spy asserts that the Rust get_reference kernel is actually invoked
    (non-vacuous guard).  The ragged output is compared byte-identically
    between backends, and a non-triviality check ensures the comparison is
    meaningful (output is not all-padding).

    Spliced coverage TODO: the phased_svar_gvl fixture does not carry
    splice_info, so only the unspliced branch (_getitem_unspliced →
    get_reference) is exercised.  The spliced branch routes through
    _fetch_spliced_ref which calls the same dispatched get_reference entry
    point.  Add a spliced fixture here once a GTF / transcript-ID column is
    available in the synthetic test case.
    """
    # --- open dataset in reference mode ---
    # with_tracks is intentionally omitted: the fixture has no tracks, so
    # with_seqs("reference") already returns Ragged[np.bytes_] directly without
    # any with_tracks(False) call.  Calling it would only emit a spurious
    # "Dataset has no tracks" warning and return self unchanged.
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("reference")

    # --- install spy on the Rust get_reference kernel ---
    # Pattern mirrors test_variants_dataset_parity.py (lines 99-109):
    # pull both impls from the registry, wrap the rust one, re-register.
    numba_fn, rust_fn = _dispatch.backends("get_reference")
    calls: dict[str, int] = {"n": 0}

    def _spy_rust(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig_entry = dict(_dispatch._REGISTRY["get_reference"])
    _dispatch.register(
        "get_reference", numba=numba_fn, rust=_spy_rust, default="numba"
    )

    try:
        # --- rust read (spy active) ---
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ds[:, :]

        # Spy-wiring guard: capture count right after rust read.
        # It must be > 0 here (proven below) and must not grow during the
        # numba read (proven after it), confirming the spy is wired ONLY to
        # the rust kernel and not to the numba path.
        rust_call_count = calls["n"]

        # --- numba reference read ---
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]

        # Spy-wiring guard: numba must NOT fire the rust spy.
        assert calls["n"] == rust_call_count, (
            f"get_reference spy fired during the numba read "
            f"(count went from {rust_call_count} to {calls['n']}) — "
            "the spy is wired to the numba path, which is a bug in the test setup."
        )

    finally:
        # Restore the original registry entry unconditionally.
        _dispatch._REGISTRY["get_reference"] = orig_entry

    # --- anti-vacuous guard ---
    # Spy fires only under GVL_BACKEND=rust; if zero calls, the rust path
    # wasn't reached and this backstop proves nothing.
    assert calls["n"] > 0, (
        f"Rust get_reference was NEVER invoked during the rust read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the reference read path to confirm get_reference is still "
        "dispatched via _dispatch.get on the Dataset.__getitem__ → "
        "_getitem_unspliced code path."
    )

    # --- sanity: output must be non-trivial ---
    out_rust_arr = np.asarray(out_rust.data)
    n_bases = out_rust_arr.size
    assert n_bases > 0, (
        "Reference output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    # Reference sequences should not be all-N padding; at least one real base.
    n_pad = np.uint8(ord("N"))
    # data is S1 dtype; compare as uint8 view
    data_u8 = out_rust_arr.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Reference output is entirely 'N' padding — regions may fall outside "
        "the reference contigs.  Non-padding bases are required to prove the "
        "comparison is meaningful."
    )

    # --- byte-identical comparison ---
    _compare_ragged_bytes(out_numba, out_rust, name="reference")
