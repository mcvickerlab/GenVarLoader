"""Dataset-level parity backstop for the fused haplotypes __getitem__ kernel.

Proves that the fused Rust entry ``reconstruct_haplotypes_fused`` (Task 13)
produces byte-identical haplotype output to the composed numba pipeline
(get_diffs_sparse → reconstruct_haplotypes_from_sparse), which is the oracle.

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The fused Rust output is byte-identical to the composed numba output.
  3. The output is non-trivial (contains non-N bases).

Scope:
  - Only the NON-SPLICE plain haplotypes path is fused (per task spec and
    audit section 5d).  The splice path continues to use the existing
    per-kernel dispatched entries.
  - The annotated path is NOT fused in Task 13 (annotation buffers must be
    sized from out_offsets[-1] which Rust computes internally; leaving it on
    the unfused dispatch path keeps the annotation path correct while the plain
    path gains the single-FFI benefit).

Spy mechanism:
  - Unlike the existing haplotypes backstop (which spies on the _dispatch
    registry for ``reconstruct_haplotypes_from_sparse``), this test spies on
    the genvarloader extension module attribute ``reconstruct_haplotypes_fused``
    directly (monkeypatched on the Haps module that calls it), since the fused
    entry is a direct call — not registered in the dispatch table.
  - The numba read uses ``GVL_BACKEND=numba``, which forces the composed path
    (get_diffs_sparse numba → reconstruct_haplotypes_from_sparse numba).  The
    fused spy must NOT fire during the numba read — its count is checked before
    and after.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _compare_ragged_bytes(
    numba_out: Ragged, rust_out: Ragged, name: str = "haplotypes"
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
# Main parity gate — fused Rust path vs. composed numba oracle
# ---------------------------------------------------------------------------


def test_fused_haps_dataset_parity(phased_svar_gvl, reference, monkeypatch):
    """Fused reconstruct_haplotypes_fused is byte-identical to composed numba oracle.

    The fused entry (called directly from _haps._reconstruct_haplotypes on the
    non-splice default path) must produce the same bytes as the composed numba
    pipeline for every (region, sample, hap) triple.

    Spy guard: we monkeypatch ``_haps_mod.reconstruct_haplotypes_fused`` to
    count calls.  The spy must fire at least once during the rust read and must
    NOT fire during the numba read (the numba path uses the composed dispatch).
    """
    # --- open dataset in haplotypes mode ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes")

    # --- install spy on reconstruct_haplotypes_fused ---
    # The fused entry is called as ``_haps_mod.reconstruct_haplotypes_fused(...)``
    # on the non-splice Rust path.
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

    # --- rust read (spy active, fused path) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    rust_call_count = calls["n"]

    # --- numba read (composed path — spy must NOT fire) ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # Wiring guard: numba must NOT fire the fused spy
    assert calls["n"] == rust_call_count, (
        f"reconstruct_haplotypes_fused spy fired during the numba read "
        f"(count went from {rust_call_count} to {calls['n']}) — "
        "the fused entry is being called on the numba path, which is a bug."
    )

    # Anti-vacuous guard: fused entry must have been invoked
    assert rust_call_count > 0, (
        f"reconstruct_haplotypes_fused was NEVER invoked during the rust read "
        f"(calls={rust_call_count}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_fused "
        "on the non-splice path when GVL_BACKEND=rust."
    )

    # --- sanity: non-trivial output ---
    out_rust_data = np.asarray(out_rust.data)
    assert out_rust_data.size > 0, (
        "Haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_rust_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Haplotypes output is entirely 'N' padding — non-padding bases are "
        "required to prove the comparison is meaningful."
    )

    # --- byte-identical comparison (fused Rust vs. composed numba) ---
    _compare_ragged_bytes(out_numba, out_rust, name="haplotypes (fused)")


# ---------------------------------------------------------------------------
# Fixed-length parity gate — exercises the output_length >= 0 fused branch
# ---------------------------------------------------------------------------


def test_fused_haps_dataset_parity_fixed_length(
    phased_svar_gvl, reference, monkeypatch
):
    """Fused reconstruct_haplotypes_fused (fixed-length arm) is byte-identical to
    composed numba oracle.

    Requests a fixed output_length via ``Dataset.with_len(N)``, which causes
    ``_prepare_request`` to emit equally-spaced ``out_offsets`` so that
    ``out_offsets[1] - out_offsets[0] == N``.  The fused entry then receives
    ``output_length=N`` (>= 0) rather than -1 (ragged mode), exercising the
    fixed-length prefix-sum arm of ``reconstruct_haplotypes_fused``.

    The dataset regions are 20 bp wide (SEQ_LEN=20 in the synthetic fixture)
    with max_jitter=2.  A fixed output_length of 15 is safely below the
    minimum region length, so no jitter expansion is needed and the
    ``with_len`` call succeeds without raising.

    Spy guard and non-vacuity check mirror the ragged test above.
    The comparison is on numpy arrays (fixed-length path returns an ndarray,
    not a Ragged, because the query layer calls ``_Flat.to_fixed``).
    """
    # --- open dataset in fixed-length haplotypes mode ---
    # SEQ_LEN=20, so output_length=15 is safely below the minimum region length.
    FIXED_LEN = 15
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes").with_len(FIXED_LEN)

    # --- install spy on reconstruct_haplotypes_fused ---
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

    # --- rust read (spy active, fixed-length fused path) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    rust_call_count = calls["n"]

    # --- numba read (composed path — spy must NOT fire) ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # Wiring guard: numba must NOT fire the fused spy
    assert calls["n"] == rust_call_count, (
        f"reconstruct_haplotypes_fused spy fired during the numba read "
        f"(count went from {rust_call_count} to {calls['n']}) — "
        "the fused entry is being called on the numba path, which is a bug."
    )

    # Anti-vacuous guard: fused entry must have been invoked at least once
    assert rust_call_count > 0, (
        f"reconstruct_haplotypes_fused was NEVER invoked during the rust read "
        f"(calls={rust_call_count}) — the backstop is vacuous. "
        "Ensure _haps._reconstruct_haplotypes calls reconstruct_haplotypes_fused "
        "on the non-splice path when GVL_BACKEND=rust."
    )

    # --- type + shape sanity ---
    # Fixed-length output returns a numpy ndarray, not a Ragged.
    assert isinstance(out_rust, np.ndarray), (
        f"Expected ndarray from fixed-length haplotypes mode, got {type(out_rust)}"
    )
    assert isinstance(out_numba, np.ndarray), (
        f"Expected ndarray from fixed-length haplotypes mode, got {type(out_numba)}"
    )
    # Last axis must be the fixed output length.
    assert out_rust.shape[-1] == FIXED_LEN, (
        f"Expected last axis == {FIXED_LEN}, got shape {out_rust.shape}"
    )

    # --- sanity: non-trivial output (contains real bases, not all 'N') ---
    data_u8 = out_rust.view(np.uint8)
    assert data_u8.size > 0, (
        "Fixed-length haplotypes output has zero bytes — the comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    assert np.any(data_u8 != n_pad), (
        "Fixed-length haplotypes output is entirely 'N' padding — non-padding "
        "bases are required to prove the comparison is meaningful."
    )

    # --- byte-identical comparison (fused fixed-length Rust vs. composed numba) ---
    np.testing.assert_array_equal(
        out_numba,
        out_rust,
        err_msg="fixed-length haplotype data differs across backends",
    )
