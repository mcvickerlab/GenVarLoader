"""Annotated+spliced haplotypes dataset parity backstop (fused rust entry, Phase 5 W3).

Proves the fused Rust entry ``reconstruct_annotated_haplotypes_spliced_fused`` produces
byte-identical (haps, var_idxs, ref_coords) output to the frozen golden (generated from
the rust implementation, oracle-verified against the composed numba pipeline at gen time),
including a negative-strand transcript that exercises the in-kernel RC triple.

Asserts:
  1. The fused entry actually fires on the rust path (spy).
  2. All three arrays are byte-identical to the frozen golden.
  3. RC actually changes the output (rc_neg=True vs rc_neg=False differ) — proves the
     negative-strand transcript exercises the in-kernel RC path (non-vacuous RC coverage).
  4. Output is non-trivial (contains non-N bases).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod
from genvarloader._ragged import RaggedAnnotatedHaps

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_annotated_spliced_haplotypes_parity(phased_svar_gvl, reference, monkeypatch):
    # --- open in annotated mode, build a spliced dataset with mixed strands inline ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("annotated").with_tracks(False)

    n = 4
    # Group regions 0+1 -> T1 (+ strand), 2+3 -> T2 (- strand). The '-' transcript
    # exercises the in-kernel RC triple (rc bytes + reverse var_idxs/ref_coords).
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"]),
        pl.Series("strand", ["+", "+", "-", "-"]),
    )
    assert (sub_bed["strand"] == "-").any(), "need a '-' transcript to cover RC"
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced, "Dataset should be in spliced mode"

    # --- spy on the fused annotated-spliced entry ---
    orig = getattr(_haps_mod, "reconstruct_annotated_haplotypes_spliced_fused", None)
    assert orig is not None, (
        "reconstruct_annotated_haplotypes_spliced_fused not found on _haps_mod — "
        "ensure it is imported at module level in _haps.py"
    )
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(
        _haps_mod, "reconstruct_annotated_haplotypes_spliced_fused", _spy
    )

    # --- read (default rust backend, spy active) ---
    out = ds[:, :]
    rust_calls = calls["n"]

    assert rust_calls > 0, (
        "reconstruct_annotated_haplotypes_spliced_fused was NEVER invoked on the "
        "read — the backstop is vacuous. Ensure _haps._reconstruct_annotated_haplotypes "
        "calls it on the splice path."
    )

    assert isinstance(out, RaggedAnnotatedHaps), type(out)

    # --- non-trivial output ---
    data_u8 = np.asarray(out.haps.data).view(np.uint8)
    assert data_u8.size > 0 and np.any(data_u8 != np.uint8(ord("N"))), (
        "annotated-spliced output is empty or all-N padding — comparison is vacuous."
    )

    # --- RC non-vacuity: rc_neg flips the '-' transcript output (rust backend) ---
    out_norc = ds.with_settings(rc_neg=False)[:, :]
    assert not np.array_equal(
        np.asarray(out.haps.data), np.asarray(out_norc.haps.data)
    ), (
        "RC made no difference — the negative-strand transcript is not exercising the "
        "in-kernel RC path (check strand propagation / rc_neg default)."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, _golden.load_flat_golden("ds_annotated_spliced"))
