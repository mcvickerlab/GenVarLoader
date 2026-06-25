"""Spliced-haplotypes dataset parity backstop (fused rust splice entry).

Proves that the fused Rust entry ``reconstruct_haplotypes_spliced_fused`` (Task 5)
produces byte-identical haplotype output to the composed numba pipeline
(reconstruct_haplotypes_from_sparse numba), which is the oracle.

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The fused Rust output is byte-identical to the composed numba output.
  3. The output is non-trivial (contains non-N bases).

Dataset construction:
  - Opens the existing phased_svar_gvl fixture in haplotypes mode.
  - Adds a synthetic transcript_id column grouping regions 0+1 → T1, 2+3 → T2.
  - Activates splice mode via with_settings(splice_info="transcript_id").

Spy mechanism:
  - Monkeypatches ``_haps_mod.reconstruct_haplotypes_spliced_fused`` to count calls.
  - The numba read uses ``GVL_BACKEND=numba``, the spy must NOT fire during it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _compare_ragged_bytes(
    numba_out: Ragged, rust_out: Ragged, name: str = "spliced haplotypes"
) -> None:
    """Assert two Ragged[np.bytes_] results are byte-identical."""
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
# Main parity gate — fused Rust splice path vs. composed numba oracle
# ---------------------------------------------------------------------------


def test_spliced_haplotypes_parity(phased_svar_gvl, reference, monkeypatch):
    """Fused reconstruct_haplotypes_spliced_fused is byte-identical to composed numba oracle.

    The fused splice entry (called directly from _haps._reconstruct_haplotypes on the
    splice path) must produce the same bytes as the composed numba pipeline for every
    (transcript, sample, hap) triple.

    Spy guard: we monkeypatch ``_haps_mod.reconstruct_haplotypes_spliced_fused`` to
    count calls.  The spy must fire at least once during the rust read and must
    NOT fire during the numba read (the numba path uses the composed dispatch).
    """
    # --- open dataset in haplotypes mode and build a spliced dataset inline ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes").with_tracks(False)

    # Group regions 0+1 → T1, 2+3 → T2 (4 regions total).
    n = 4
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"])
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")

    assert ds.is_spliced, "Dataset should be in spliced mode"

    # --- install spy on reconstruct_haplotypes_spliced_fused ---
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

    # --- rust read (spy active, fused splice path) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    rust_call_count = calls["n"]

    # --- numba read (composed path — spy must NOT fire) ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # Wiring guard: numba must NOT fire the fused splice spy
    assert calls["n"] == rust_call_count, (
        f"reconstruct_haplotypes_spliced_fused spy fired during the numba read "
        f"(count went from {rust_call_count} to {calls['n']}) — "
        "the fused splice entry is being called on the numba path, which is a bug."
    )

    # Anti-vacuous guard: fused splice entry must have been invoked
    assert rust_call_count > 0, (
        f"reconstruct_haplotypes_spliced_fused was NEVER invoked during the rust read "
        f"(calls={rust_call_count}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_spliced_fused "
        "on the splice path when GVL_BACKEND=rust."
    )

    # --- sanity: non-trivial output ---
    out_rust_data = np.asarray(out_rust.data)
    assert out_rust_data.size > 0, (
        "Spliced haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_rust_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Spliced haplotypes output is entirely 'N' padding — non-padding bases are "
        "required to prove the comparison is meaningful."
    )

    # --- byte-identical comparison (fused Rust vs. composed numba) ---
    _compare_ragged_bytes(out_numba, out_rust, name="spliced haplotypes (fused)")
