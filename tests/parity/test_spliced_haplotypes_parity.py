"""Spliced-haplotypes dataset parity backstop (fused rust splice entry).

Proves that the fused Rust entry ``reconstruct_haplotypes_spliced_fused`` (Task 5)
produces byte-identical haplotype output to the frozen golden (generated from
the rust implementation, oracle-verified against the composed numba pipeline).

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The Rust output is byte-identical to the frozen golden.
  3. The output is non-trivial (contains non-N bases).

Dataset construction:
  - Opens the existing phased_svar_gvl fixture in haplotypes mode.
  - Adds a synthetic transcript_id column grouping regions 0+1 → T1, 2+3 → T2.
  - Activates splice mode via with_settings(splice_info="transcript_id").
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod

from tests.parity import _golden

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Main parity gate — fused Rust splice path vs. frozen golden
# ---------------------------------------------------------------------------


def test_spliced_haplotypes_parity(phased_svar_gvl, reference, monkeypatch):
    """Fused reconstruct_haplotypes_spliced_fused output matches the frozen golden.

    Spy guard: we monkeypatch ``_haps_mod.reconstruct_haplotypes_spliced_fused``
    to count calls.  The spy must fire at least once (anti-vacuous guard).
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes").with_tracks(False)

    n = 4
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"])
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")

    assert ds.is_spliced, "Dataset should be in spliced mode"

    orig_fused = getattr(_haps_mod, "reconstruct_haplotypes_spliced_fused", None)
    assert orig_fused is not None, (
        "reconstruct_haplotypes_spliced_fused not found on _haps_mod — "
        "ensure it is imported at module level in _haps.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_spliced_fused", _spy_fused)

    # --- read (default rust backend, spy active) ---
    out = ds[:, :]

    # Anti-vacuous guard
    assert calls["n"] > 0, (
        f"reconstruct_haplotypes_spliced_fused was NEVER invoked during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_spliced_fused "
        "on the splice path."
    )

    # --- sanity: non-trivial output ---
    out_data = np.asarray(out.data)
    assert out_data.size > 0, (
        "Spliced haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Spliced haplotypes output is entirely 'N' padding — non-padding bases are "
        "required to prove the comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_spliced_haps")
    )
