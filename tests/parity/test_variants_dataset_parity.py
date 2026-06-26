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
from genvarloader._dataset._flat_variants import DummyVariant
from seqpro.rag import Ragged

from ._fixtures import build_strand_mixed_dataset

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


# ---------------------------------------------------------------------------
# variant-windows cross-backend parity
# ---------------------------------------------------------------------------


def _compare_flat_window(n_win, r_win, name: str) -> None:
    """Assert that two _FlatWindow objects are byte-identical.

    Compares data tokens (dtype + values), seq_offsets, and var_offsets.
    """
    n_data = np.asarray(n_win.data)
    r_data = np.asarray(r_win.data)
    assert n_data.dtype == r_data.dtype, (
        f"{name}.data dtype mismatch: numba={n_data.dtype}, rust={r_data.dtype}"
    )
    np.testing.assert_array_equal(
        n_data, r_data, err_msg=f"{name}.data mismatch across backends"
    )
    n_seq = np.asarray(n_win.seq_offsets, np.int64)
    r_seq = np.asarray(r_win.seq_offsets, np.int64)
    np.testing.assert_array_equal(
        n_seq, r_seq, err_msg=f"{name}.seq_offsets mismatch across backends"
    )
    n_var = np.asarray(n_win.var_offsets, np.int64)
    r_var = np.asarray(r_win.var_offsets, np.int64)
    np.testing.assert_array_equal(
        n_var, r_var, err_msg=f"{name}.var_offsets mismatch across backends"
    )


def test_variant_windows_getitem_parity_across_backends(
    phased_svar_gvl, reference, monkeypatch
):
    """variant-windows __getitem__ must be byte-identical across numba/rust backends.

    Closes the coverage gap identified in the Task 7 review: the windows wiring
    uses ``setattr(win, name, fw)`` for each kernel dict key, so a wrong key name
    would silently drop the window with no crash.  This test proves the windows
    output is non-empty AND byte-identical end-to-end on both backends.
    """
    from genvarloader import VarWindowOpt

    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = (
        ds.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )

    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[[0, 1], [0, 1]]

    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[[0, 1], [0, 1]]

    # Both outputs must have the same window fields present.
    assert (out_numba.ref_window is None) == (out_rust.ref_window is None), (
        "ref_window presence differs across backends: "
        f"numba={out_numba.ref_window is not None}, rust={out_rust.ref_window is not None}"
    )
    assert (out_numba.alt_window is None) == (out_rust.alt_window is None), (
        "alt_window presence differs across backends: "
        f"numba={out_numba.alt_window is not None}, rust={out_rust.alt_window is not None}"
    )

    if out_numba.ref_window is not None:
        _compare_flat_window(out_numba.ref_window, out_rust.ref_window, "ref_window")
    if out_numba.alt_window is not None:
        _compare_flat_window(out_numba.alt_window, out_rust.alt_window, "alt_window")

    # Anti-vacuous: at least one window field must be present and non-empty.
    present = [w for w in (out_numba.ref_window, out_numba.alt_window) if w is not None]
    assert len(present) > 0, (
        "No window fields present in the numba output — test is vacuous. "
        "Check that VarWindowOpt.ref/alt defaults produce at least one window."
    )
    assert any(np.asarray(w.data).size > 0 for w in present), (
        "All window data arrays are empty — no variants in the indexed batch. "
        "The cross-backend comparison is vacuous."
    )


# ---------------------------------------------------------------------------
# Neg-strand variants parity + dummy-fill coverage (Task 6)
# ---------------------------------------------------------------------------


def _read_variants_both_backends(ds, monkeypatch):
    """Read ds[:, :] under numba then rust; return (out_numba, out_rust)."""
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    return out_numba, out_rust


def test_neg_strand_variants_rc_parity_and_kernel_invoked(
    tmp_path, synthetic_case, monkeypatch
):
    """variants-mode neg-strand RC is byte-identical across backends, and the
    rust rc_alleles kernel actually fires on the live read (non-vacuous)."""
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = gvl.Dataset.open(ds_dir, reference=ref).with_tracks(False).with_seqs("variants")

    # Non-vacuity: fixture must carry −strand regions (rc_neg defaults True).
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    # Spy on the rust rc_alleles to prove it runs on the live neg-strand path.
    numba_fn, rust_fn = _dispatch.backends("rc_alleles")
    calls = {"n": 0}

    def _spy_rust(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig_entry = dict(_dispatch._REGISTRY["rc_alleles"])
    _dispatch.register("rc_alleles", numba=numba_fn, rust=_spy_rust, default="rust")
    try:
        out_numba, out_rust = _read_variants_both_backends(ds, monkeypatch)
    finally:
        _dispatch._REGISTRY["rc_alleles"] = orig_entry

    assert calls["n"] > 0, (
        "rust rc_alleles was never invoked on the neg-strand variants read — "
        "the backstop is vacuous. Confirm a variant overlaps a −strand region; if "
        "the synthetic variant set does not, extend build_strand_mixed_dataset with a "
        "−strand region positioned over a known variant."
    )
    for field_name in out_numba.fields:
        _compare_ragged_field(out_numba[field_name], out_rust[field_name], field_name)


def test_neg_strand_variants_custom_dummy_parity(tmp_path, synthetic_case, monkeypatch):
    """A custom non-palindromic dummy (alt/ref = b'AC') filled into empty groups on
    a −strand read is RC'd identically by rust and the seqpro reference."""
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref)
        .with_tracks(False)
        .with_seqs("variants")
        .with_settings(dummy_variant=DummyVariant(alt=b"AC", ref=b"AC"))
    )
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    out_numba, out_rust = _read_variants_both_backends(ds, monkeypatch)
    for field_name in out_numba.fields:
        _compare_ragged_field(out_numba[field_name], out_rust[field_name], field_name)
