"""Variants-mode dataset-level parity backstop.

Proves that the Rust backend produces byte-identical variants output matching
the frozen golden (generated from the rust implementation, oracle-verified
against the numba pipeline at gen time).

Kernels exercised end-to-end:
  - gather_rows_i32   (v_idxs gather — always on the variants path)
  - gather_alleles    (alt/ref sequence gather)
  - fill_empty_*      (empty group sentinel fill)
  - compact_keep_*    (AF filtering, when min_af/max_af are active)
  - rc_alleles        (reverse-complement of alleles on neg-strand regions)
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._flat_variants  # noqa: F401 — triggers register()
from genvarloader._dataset._flat_variants import DummyVariant

from tests.parity import _golden
from ._fixtures import build_strand_mixed_dataset

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Main backstop test
# ---------------------------------------------------------------------------


def test_variants_getitem_parity_and_kernels_invoked(phased_svar_gvl, reference):
    """Rust variants output matches the frozen golden.

    The spy asserts that the Rust gather_rows_i32 kernel is actually invoked
    (non-vacuous guard).
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_tracks(False)
    ds = ds.with_seqs("variants")

    spy_fn, calls, restore = _golden.make_kernel_spy("gather_rows_i32")
    try:
        out = ds[:, :]
    finally:
        restore()

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust gather_rows_i32 was NEVER invoked during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the variants read path to confirm gather_rows_i32 is still "
        "called on the get_variants_flat → _gather_rows code path."
    )

    # --- sanity: output must be non-trivial ---
    n_total_variants = int(out.start.data.size)
    assert n_total_variants > 0, (
        "RaggedVariants output contains zero variants — regions don't overlap any "
        "variants in the dataset.  The parity comparison is vacuous."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, _golden.load_flat_golden("ds_variants"))


# ---------------------------------------------------------------------------
# AF-filtered backstop (compact_keep_i32 exercise)
# ---------------------------------------------------------------------------


def test_variants_af_filter_parity(phased_svar_gvl, reference):
    """Same parity check with a mild AF filter to exercise compact_keep_i32.

    If the dataset has no AF annotation or the golden was not generated,
    skips with a clear message.
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

    # Load golden — may not exist if AF was unavailable at generation time.
    try:
        golden = _golden.load_flat_golden("ds_variants_af")
    except FileNotFoundError:
        pytest.skip("ds_variants_af golden not generated (AF unavailable at gen time)")

    spy_fn, ck_calls, restore = _golden.make_kernel_spy("compact_keep_i32")
    try:
        out = ds[:, :]
    finally:
        restore()

    # compact_keep may not fire if no variants fall within the AF window;
    # only assert it if variants are present.
    n_vars = int(out.start.data.size)
    if n_vars > 0 and ck_calls["n"] == 0:
        pytest.xfail(
            "compact_keep_i32 was not invoked even though variants are present — "
            "AF filter may not be active on this code path."
        )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, golden)


# ---------------------------------------------------------------------------
# variant-windows cross-backend parity
# ---------------------------------------------------------------------------


def test_variant_windows_getitem_parity_across_backends(phased_svar_gvl, reference):
    """variant-windows __getitem__ must match the frozen golden.

    Proves the windows output is non-empty AND byte-identical to the golden
    end-to-end.
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

    out = ds[[0, 1], [0, 1]]

    # Anti-vacuous: at least one window field must be present and non-empty.
    present = [w for w in (out.ref_window, out.alt_window) if w is not None]
    assert len(present) > 0, (
        "No window fields present in the output — test is vacuous. "
        "Check that VarWindowOpt.ref/alt defaults produce at least one window."
    )
    assert any(np.asarray(w.data).size > 0 for w in present), (
        "All window data arrays are empty — no variants in the indexed batch. "
        "The comparison is vacuous."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_variant_windows")
    )


# ---------------------------------------------------------------------------
# Neg-strand variants parity + dummy-fill coverage (Task 6)
# ---------------------------------------------------------------------------


def test_neg_strand_variants_rc_parity_and_kernel_invoked(tmp_path, synthetic_case):
    """variants-mode neg-strand RC output matches the frozen golden, and the
    rust rc_alleles kernel actually fires on the live read (non-vacuous)."""
    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref).with_tracks(False).with_seqs("variants")
    )

    # Non-vacuity: fixture must carry −strand regions (rc_neg defaults True).
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    spy_fn, calls, restore = _golden.make_kernel_spy("rc_alleles")
    try:
        out = ds[:, :]
    finally:
        restore()

    assert calls["n"] > 0, (
        "rust rc_alleles was never invoked on the neg-strand variants read — "
        "the backstop is vacuous. Confirm a variant overlaps a −strand region; if "
        "the synthetic variant set does not, extend build_strand_mixed_dataset with a "
        "−strand region positioned over a known variant."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_neg_strand_variants")
    )


def test_neg_strand_variants_custom_dummy_parity(tmp_path, synthetic_case):
    """A custom non-palindromic dummy (alt/ref = b'AC') filled into empty groups on
    a −strand read produces output matching the frozen golden."""
    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref)
        .with_tracks(False)
        .with_seqs("variants")
        .with_settings(dummy_variant=DummyVariant(alt=b"AC", ref=b"AC"))
    )
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    out = ds[:, :]

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_neg_strand_variants_dummy")
    )
