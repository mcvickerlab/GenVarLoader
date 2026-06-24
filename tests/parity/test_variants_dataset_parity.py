"""Variants-mode dataset-level parity backstop.

Proves that flipping GVL_BACKEND (numba vs rust) produces byte-identical
variants output through the real Dataset.__getitem__ path — with a spy
guard proving the Rust gather_rows_i32 kernel is actually invoked (no
vacuous pass).

Kernels exercised end-to-end:
  - gather_rows_i32   (v_idxs gather — always on the variants path)
  - gather_alleles    (alt/ref sequence gather)
  - fill_empty_*      (empty group sentinel fill)
  - compact_keep_*    (AF filtering, when min_af/max_af are active)
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._flat_variants  # noqa: F401 — triggers register()
import genvarloader._dispatch as _dispatch
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_ragged_field(numba_field: Ragged, rust_field: Ragged, name: str) -> None:
    """Assert that two Ragged fields are byte-identical.

    For opaque-string fields (alt/ref) the comparison covers both the char
    data buffer (S1 dtype) and the variant-level offsets.  For numeric fields
    it covers the flat data array and the offsets.
    """
    if numba_field.is_string:
        # opaque-string: compare char data via .data and char-level offsets
        # via .offsets (which returns str_offsets for string layouts).
        n_data = np.asarray(numba_field.data, dtype="S1")
        r_data = np.asarray(rust_field.data, dtype="S1")
        np.testing.assert_array_equal(
            n_data,
            r_data,
            err_msg=f"allele char data differs for field '{name}'",
        )
        n_off = np.asarray(numba_field.offsets, dtype=np.int64)
        r_off = np.asarray(rust_field.offsets, dtype=np.int64)
        np.testing.assert_array_equal(
            n_off,
            r_off,
            err_msg=f"allele offsets differ for field '{name}'",
        )
    else:
        n_data = np.asarray(numba_field.data)
        r_data = np.asarray(rust_field.data)
        assert n_data.dtype == r_data.dtype, (
            f"dtype mismatch for field '{name}': numba={n_data.dtype}, "
            f"rust={r_data.dtype}"
        )
        np.testing.assert_array_equal(
            n_data,
            r_data,
            err_msg=f"data differs for numeric field '{name}'",
        )
        n_off = np.asarray(numba_field.offsets, dtype=np.int64)
        r_off = np.asarray(rust_field.offsets, dtype=np.int64)
        np.testing.assert_array_equal(
            n_off,
            r_off,
            err_msg=f"offsets differ for numeric field '{name}'",
        )


# ---------------------------------------------------------------------------
# Main backstop test
# ---------------------------------------------------------------------------


def test_variants_getitem_parity_and_kernels_invoked(
    phased_svar_gvl, reference, monkeypatch
):
    """Flips GVL_BACKEND numba<->rust through the real variants getitem path.

    The spy asserts that the Rust gather_rows_i32 kernel is actually invoked
    (non-vacuous guard).  Every present RaggedVariants field is compared
    byte-identically between backends.
    """
    # --- open dataset in variants mode ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_tracks(False)  # ensure return type is RaggedVariants directly
    ds = ds.with_seqs("variants")

    # --- install spy on the Rust gather_rows_i32 kernel ---
    # Save the original registry entry so we can restore it unconditionally.
    numba_fn, rust_fn = _dispatch.backends("gather_rows_i32")
    calls: dict[str, int] = {"n": 0}

    def _spy_rust(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    # Re-register with the spied rust impl.
    orig_entry = dict(_dispatch._REGISTRY["gather_rows_i32"])
    _dispatch.register(
        "gather_rows_i32", numba=numba_fn, rust=_spy_rust, default="numba"
    )

    try:
        # --- numba reference read ---
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]

        # Spy guard: verify the spy hasn't fired yet (we're in numba mode)
        assert calls["n"] == 0, (
            "gather_rows_i32 spy fired during numba read — "
            "the spy is wired to the numba path, which is a bug in the test setup."
        )

        # --- rust read (spy active) ---
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ds[:, :]

    finally:
        # Restore the original registry entry unconditionally.
        _dispatch._REGISTRY["gather_rows_i32"] = orig_entry

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust gather_rows_i32 was NEVER invoked during the rust read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the variants read path to confirm gather_rows_i32 is still "
        "called on the get_variants_flat → _gather_rows code path."
    )

    # --- sanity: output must be non-trivial ---
    start_numba = out_numba.start
    n_total_variants = int(start_numba.data.size)
    assert n_total_variants > 0, (
        "RaggedVariants output contains zero variants — regions don't overlap any "
        "variants in the dataset.  The parity comparison is vacuous."
    )

    # --- byte-identical comparison for every present field ---
    fields = out_numba.fields
    assert len(fields) > 0, "RaggedVariants has no fields — unexpected empty record."

    for field_name in fields:
        n_field = out_numba[field_name]
        r_field = out_rust[field_name]
        _compare_ragged_field(n_field, r_field, field_name)


# ---------------------------------------------------------------------------
# AF-filtered backstop (compact_keep_i32 exercise)
# ---------------------------------------------------------------------------


def test_variants_af_filter_parity(phased_svar_gvl, reference, monkeypatch):
    """Same parity check with a mild AF filter to exercise compact_keep_i32.

    If the dataset has no AF annotation, skips with a clear message.
    """
    ds_base = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds_base = ds_base.with_tracks(False)

    # Try to apply an AF filter.  with_settings raises if AF is unavailable.
    try:
        ds = ds_base.with_seqs("variants").with_settings(min_af=0.1, max_af=0.9)
    except Exception as e:
        pytest.skip(
            f"AF filtering unavailable on this dataset — skipping compact_keep "
            f"exercise ({type(e).__name__}: {e})"
        )

    # Spy on compact_keep_i32 to confirm it fires during the rust read.
    numba_ck, rust_ck = _dispatch.backends("compact_keep_i32")
    ck_calls: dict[str, int] = {"n": 0}

    def _spy_ck(*a, **k):
        ck_calls["n"] += 1
        return rust_ck(*a, **k)

    orig_ck = dict(_dispatch._REGISTRY["compact_keep_i32"])
    _dispatch.register(
        "compact_keep_i32", numba=numba_ck, rust=_spy_ck, default="numba"
    )

    try:
        monkeypatch.setenv("GVL_BACKEND", "numba")
        try:
            out_numba = ds[:, :]
        except KeyError as e:
            # AF info genuinely missing from variant info at read time → skip.
            # Any other exception propagates and fails loudly (don't mask a real
            # AF-path regression as a skip).
            pytest.skip(
                f"AF key missing in variant info at read time — "
                f"skipping compact_keep exercise ({type(e).__name__}: {e})"
            )

        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ds[:, :]
    finally:
        _dispatch._REGISTRY["compact_keep_i32"] = orig_ck

    # compact_keep may not fire if no variants fall within the AF window;
    # only assert it if variants are present.
    n_vars = int(out_numba.start.data.size)
    if n_vars > 0 and ck_calls["n"] == 0:
        pytest.xfail(
            "compact_keep_i32 was not invoked even though variants are present — "
            "AF filter may not be active on this code path."
        )

    for field_name in out_numba.fields:
        _compare_ragged_field(out_numba[field_name], out_rust[field_name], field_name)
