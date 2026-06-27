"""Haplotypes-mode dataset-level parity backstop.

Proves that the Rust reconstruct_haplotypes_fused / reconstruct_annotated_haplotypes_fused
kernels produce byte-identical output to the frozen goldens generated from the numba-verified
rust output.

Kernels exercised end-to-end:
  - reconstruct_haplotypes_fused         (haplotypes mode, non-splice, Task 13)
  - reconstruct_annotated_haplotypes_fused (annotated mode, non-splice, Task 4)

Two output modes are covered:
  - "haplotypes"  → Ragged[np.bytes_]
  - "annotated"   → RaggedAnnotatedHaps (.haps, .var_idxs, .ref_coords)
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._genotypes  # noqa: F401 — triggers register("reconstruct_haplotypes_from_sparse")
import genvarloader._dataset._haps as _haps_mod
from genvarloader._ragged import RaggedAnnotatedHaps

from tests.parity import _golden

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Main backstop — "haplotypes" mode
# ---------------------------------------------------------------------------


def test_haplotypes_mode_dataset_parity(phased_svar_gvl, reference, monkeypatch):
    """Rust reconstruct_haplotypes_fused output matches the frozen golden.

    Spy guard proves the fused entry is actually invoked (non-vacuous).
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes")

    # --- install spy on the fused Rust reconstruct_haplotypes_fused entry ---
    orig_fused = _haps_mod.reconstruct_haplotypes_fused
    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_fused", _spy_fused)

    # --- read (default rust backend, spy active) ---
    out = ds[:, :]

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust reconstruct_haplotypes_fused was NEVER invoked during the "
        f"read (calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the haplotypes read path to confirm "
        "reconstruct_haplotypes_fused is called on the non-splice rust path "
        "in _haps._reconstruct_haplotypes."
    )

    # --- sanity: output must be non-trivial ---
    out_data = np.asarray(out.data)
    n_bases = out_data.size
    assert n_bases > 0, (
        "Haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Haplotypes output is entirely 'N' padding — regions may fall outside "
        "the reference contigs.  Non-padding bases are required to prove the "
        "comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, _golden.load_flat_golden("ds_haplotypes_mode"))


# ---------------------------------------------------------------------------
# Annotated backstop — "annotated" mode
# ---------------------------------------------------------------------------


def test_annotated_haplotypes_mode_dataset_parity(
    phased_svar_gvl, reference, monkeypatch
):
    """Rust reconstruct_annotated_haplotypes_fused output matches the frozen golden.

    Covers the annotated path (with_seqs("annotated")).  All three arrays —
    haps, var_idxs, and ref_coords — are compared byte-identically against the golden.
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("annotated")

    # --- install spy on the fused Rust reconstruct_annotated_haplotypes_fused entry ---
    orig_fused = _haps_mod.reconstruct_annotated_haplotypes_fused
    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_annotated_haplotypes_fused", _spy_fused)

    # --- read (default rust backend, spy active) ---
    out = ds[:, :]

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust reconstruct_annotated_haplotypes_fused was NEVER invoked during the "
        f"read (calls={calls['n']}) — the annotated backstop is vacuous. "
        "Inspect the annotated read path to confirm "
        "reconstruct_annotated_haplotypes_fused is called on the non-splice rust path "
        "in _haps._reconstruct_annotated_haplotypes."
    )

    # --- type sanity ---
    assert isinstance(out, RaggedAnnotatedHaps), (
        f"Expected RaggedAnnotatedHaps from annotated mode, got {type(out)}"
    )

    # --- sanity: output must be non-trivial ---
    haps_data = np.asarray(out.haps.data)
    n_bases = haps_data.size
    assert n_bases > 0, (
        "Annotated haplotypes output contains zero bytes — regions don't overlap "
        "any reference sequence.  The parity comparison is vacuous."
    )
    data_u8 = haps_data.view(np.uint8)
    n_pad = np.uint8(ord("N"))
    assert np.any(data_u8 != n_pad), (
        "Annotated haplotypes output is entirely 'N' padding — regions may fall "
        "outside the reference contigs.  Non-padding bases are required to prove "
        "the comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, _golden.load_flat_golden("ds_annotated_mode"))
