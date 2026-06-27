"""Dataset-level parity backstop for the fused haplotypes __getitem__ kernel.

Proves that the fused Rust entry ``reconstruct_haplotypes_fused`` (Task 13)
produces byte-identical haplotype output to the frozen golden (generated from
the rust implementation, oracle-verified against numba at generation time).

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The Rust output is byte-identical to the frozen golden.
  3. The output is non-trivial (contains non-N bases).

Scope:
  - Only the NON-SPLICE plain haplotypes path is fused (per task spec and
    audit section 5d).  The splice path continues to use the existing
    per-kernel dispatched entries.
  - The annotated path is NOT fused in Task 13.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod

from tests.parity import _golden

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Main parity gate — fused Rust path vs. frozen golden
# ---------------------------------------------------------------------------


def test_fused_haps_dataset_parity(phased_svar_gvl, reference, monkeypatch):
    """Fused reconstruct_haplotypes_fused output matches the frozen golden.

    Spy guard: we monkeypatch ``_haps_mod.reconstruct_haplotypes_fused`` to
    count calls.  The spy must fire at least once (anti-vacuous guard).
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes")

    orig_fused = getattr(_haps_mod, "reconstruct_haplotypes_fused", None)
    assert orig_fused is not None, (
        "reconstruct_haplotypes_fused not found on _haps_mod — "
        "ensure it is imported at module level in _haps.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_fused", _spy_fused)

    # --- read (default rust backend, spy active) ---
    out = ds[:, :]

    # Anti-vacuous guard: fused entry must have been invoked
    assert calls["n"] > 0, (
        f"reconstruct_haplotypes_fused was NEVER invoked during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_fused "
        "on the non-splice path."
    )

    # --- sanity: non-trivial output ---
    out_data = np.asarray(out.data)
    assert out_data.size > 0, (
        "Haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Haplotypes output is entirely 'N' padding — non-padding bases are "
        "required to prove the comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_haplotypes_mode")
    )


# ---------------------------------------------------------------------------
# Fixed-length parity gate — exercises the output_length >= 0 fused branch
# ---------------------------------------------------------------------------


def test_fused_haps_dataset_parity_fixed_length(
    phased_svar_gvl, reference, monkeypatch
):
    """Fused reconstruct_haplotypes_fused (fixed-length arm) matches the frozen golden.

    Requests a fixed output_length via ``Dataset.with_len(N)``.  The fused entry
    then receives ``output_length=N`` (>= 0) rather than -1 (ragged mode).

    Spy guard and non-vacuity check mirror the ragged test above.
    The golden stores the fixed-length ndarray output.
    """
    FIXED_LEN = 15
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes").with_len(FIXED_LEN)

    orig_fused = getattr(_haps_mod, "reconstruct_haplotypes_fused", None)
    assert orig_fused is not None, (
        "reconstruct_haplotypes_fused not found on _haps_mod — "
        "ensure it is imported at module level in _haps.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_fused", _spy_fused)

    # --- read (default rust backend, fixed-length fused path) ---
    out = ds[:, :]

    # Anti-vacuous guard
    assert calls["n"] > 0, (
        f"reconstruct_haplotypes_fused was NEVER invoked during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_fused "
        "on the non-splice path."
    )

    # --- type + shape sanity ---
    assert isinstance(out, np.ndarray), (
        f"Expected ndarray from fixed-length haplotypes mode, got {type(out)}"
    )
    assert out.shape[-1] == FIXED_LEN, (
        f"Expected last axis == {FIXED_LEN}, got shape {out.shape}"
    )

    # --- sanity: non-trivial output ---
    data_u8 = out.view(np.uint8)
    assert data_u8.size > 0, (
        "Fixed-length haplotypes output has zero bytes — the comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    assert np.any(data_u8 != n_pad), (
        "Fixed-length haplotypes output is entirely 'N' padding — non-padding "
        "bases are required to prove the comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_haps_fixed_len")
    )
