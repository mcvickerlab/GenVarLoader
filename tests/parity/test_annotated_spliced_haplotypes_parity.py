"""Annotated+spliced haplotypes dataset parity backstop (fused rust entry, Phase 5 W3).

Proves the fused Rust entry ``reconstruct_annotated_haplotypes_spliced_fused`` produces
byte-identical (haps, var_idxs, ref_coords) output to the composed numba oracle for the
annotated AND spliced path — including a negative-strand transcript, which exercises the
in-kernel RC triple (reverse-complement of the sequence bytes + reverse of the two
annotation arrays, no complement).

Asserts:
  1. The fused entry actually fires on the rust path and NOT on the numba path (spy).
  2. All three arrays are byte-identical across backends (haps + var_idxs + ref_coords + offsets).
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
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


def _compare_ragged(numba_out: Ragged, rust_out: Ragged, name: str) -> None:
    n_data = np.asarray(numba_out.data)
    r_data = np.asarray(rust_out.data)
    assert n_data.dtype == r_data.dtype, (
        f"dtype mismatch for {name}: numba={n_data.dtype}, rust={r_data.dtype}"
    )
    np.testing.assert_array_equal(
        n_data, r_data, err_msg=f"data differs across backends for '{name}'"
    )
    np.testing.assert_array_equal(
        np.asarray(numba_out.offsets, np.int64),
        np.asarray(rust_out.offsets, np.int64),
        err_msg=f"offsets differ across backends for '{name}'",
    )


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

    # --- rust read (fused path) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    rust_calls = calls["n"]

    # --- numba read (composed oracle; spy must NOT fire) ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    assert calls["n"] == rust_calls, (
        "fused annotated-spliced spy fired during the numba read — "
        "the fused entry is being called on the numba path."
    )
    assert rust_calls > 0, (
        "reconstruct_annotated_haplotypes_spliced_fused was NEVER invoked on the rust "
        "read — the backstop is vacuous. Ensure _haps._reconstruct_annotated_haplotypes "
        "calls it on the splice path when GVL_BACKEND=rust."
    )

    assert isinstance(out_rust, RaggedAnnotatedHaps), type(out_rust)
    assert isinstance(out_numba, RaggedAnnotatedHaps), type(out_numba)

    # --- non-trivial output ---
    data_u8 = np.asarray(out_rust.haps.data).view(np.uint8)
    assert data_u8.size > 0 and np.any(data_u8 != np.uint8(ord("N"))), (
        "annotated-spliced output is empty or all-N padding — comparison is vacuous."
    )

    # --- RC non-vacuity: rc_neg flips the '-' transcript output (rust backend) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_norc = ds.with_settings(rc_neg=False)[:, :]
    assert not np.array_equal(
        np.asarray(out_rust.haps.data), np.asarray(out_norc.haps.data)
    ), (
        "RC made no difference — the negative-strand transcript is not exercising the "
        "in-kernel RC path (check strand propagation / rc_neg default)."
    )

    # --- byte-identity across backends on all three arrays ---
    _compare_ragged(out_numba.haps, out_rust.haps, "annotated-spliced.haps")
    _compare_ragged(out_numba.var_idxs, out_rust.var_idxs, "annotated-spliced.var_idxs")
    _compare_ragged(
        out_numba.ref_coords, out_rust.ref_coords, "annotated-spliced.ref_coords"
    )
