"""Haplotypes-mode dataset-level parity backstop.

Proves that flipping GVL_BACKEND (numba vs rust) produces byte-identical
haplotype output through the real Dataset.__getitem__ path — with a spy
guard proving the Rust reconstruct_haplotypes_from_sparse kernel is actually
invoked (no vacuous pass).

Kernels exercised end-to-end:
  - reconstruct_haplotypes_from_sparse  (haplotype reconstruction — dispatched
    via _dispatch.get in
    _dataset/_genotypes.py:reconstruct_haplotypes_from_sparse())

Two output modes are covered:
  - "haplotypes"  → Ragged[np.bytes_]
  - "annotated"   → RaggedAnnotatedHaps (.haps, .var_idxs, .ref_coords)

Spliced-haplotypes note:
  The parity fixture (phased_svar_gvl) is not opened with splice_info, so the
  splice branch (_reconstruct_haplotypes splice path) is NOT exercised here.
  The rust non-splice unspliced haps path now uses ``reconstruct_haplotypes_fused``
  (a direct fused Rust entry — Task 13) rather than the composed dispatched
  ``reconstruct_haplotypes_from_sparse`` pair.  The annotated non-splice rust path
  now uses ``reconstruct_annotated_haplotypes_fused`` (Task 4).  The splice paths
  still use the composed dispatched ``reconstruct_haplotypes_from_sparse`` wrapper.
  A dedicated spliced fixture would require a GTF / transcript-ID
  column that the current synthetic case does not provide; see the "Spliced
  coverage TODO" comment below.

Numba SystemError note:
  The numba parallel=True reconstruct driver is known to raise SystemError on
  certain deletion-heavy inputs (negative slice index inside prange).  The
  existing unit-level parity test (test_reconstruct_haplotypes_parity.py) uses
  assume(False) to discard those inputs.  The synthetic fixture dataset used
  here contains a mix of SNPs, insertions, and deletions.  If the numba read
  raises SystemError below, that is a real pre-existing numba bug — the test
  will fail with a clear error rather than silently pass.  This is intentional:
  we want the dataset-level backstop to fail loudly if the fixture happens to
  trigger the bug so it can be investigated.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._genotypes  # noqa: F401 — triggers register("reconstruct_haplotypes_from_sparse")
import genvarloader._dataset._haps as _haps_mod
from genvarloader._ragged import RaggedAnnotatedHaps
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_ragged_bytes(
    numba_out: Ragged, rust_out: Ragged, name: str = "haplotypes"
) -> None:
    """Assert that two Ragged[np.bytes_] results are byte-identical.

    Compares both the flat character data buffer (uint8 / S1) and the
    per-row offsets.
    """
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


def _compare_ragged_int(numba_out: Ragged, rust_out: Ragged, name: str) -> None:
    """Assert that two Ragged integer arrays are identical."""
    n_data = np.asarray(numba_out.data)
    r_data = np.asarray(rust_out.data)
    assert n_data.dtype == r_data.dtype, (
        f"dtype mismatch for '{name}': numba={n_data.dtype}, rust={r_data.dtype}"
    )
    np.testing.assert_array_equal(
        n_data,
        r_data,
        err_msg=f"annotation data differs across backends for '{name}'",
    )
    n_off = np.asarray(numba_out.offsets, dtype=np.int64)
    r_off = np.asarray(rust_out.offsets, dtype=np.int64)
    np.testing.assert_array_equal(
        n_off,
        r_off,
        err_msg=f"annotation offsets differ across backends for '{name}'",
    )


# ---------------------------------------------------------------------------
# Main backstop — "haplotypes" mode
# ---------------------------------------------------------------------------


def test_haplotypes_mode_dataset_parity(phased_svar_gvl, reference, monkeypatch):
    """Flips GVL_BACKEND numba<->rust through the real haplotypes getitem path.

    After Task 13 fusion, the rust non-splice default path calls
    ``reconstruct_haplotypes_fused`` (a direct Rust entry, one FFI crossing)
    instead of the composed ``get_diffs_sparse`` + ``reconstruct_haplotypes_from_sparse``
    pair.  The spy therefore tracks ``_haps_mod.reconstruct_haplotypes_fused``
    for the rust read.  The numba path still uses the composed dispatch
    (``reconstruct_haplotypes_from_sparse``), so the fused spy must NOT fire
    during the numba read — confirmed by the wiring guard.

    The ragged output is compared byte-identically between backends, and a
    non-triviality check ensures the comparison is meaningful.

    Spliced coverage TODO: the phased_svar_gvl fixture does not carry
    splice_info, so only the unspliced branch (_reconstruct_haplotypes without
    splice_plan) is exercised here.  The splice path still calls the composed
    (unfused) dispatched reconstruct_haplotypes_from_sparse entry point
    (see _haps.py splice-plan branch).  Add a spliced fixture once a GTF /
    transcript-ID column is available in the synthetic test case.
    """
    # --- open dataset in haplotypes mode ---
    # with_tracks is intentionally omitted: the fixture has no tracks, so
    # with_seqs("haplotypes") returns Ragged[np.bytes_] directly.
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("haplotypes")

    # --- install spy on the fused Rust reconstruct_haplotypes_fused entry ---
    # After Task 13, the non-splice rust path calls reconstruct_haplotypes_fused
    # (module-level name in _haps_mod) rather than the dispatched
    # reconstruct_haplotypes_from_sparse.  The numba path goes through the
    # composed dispatch and never calls reconstruct_haplotypes_fused.
    orig_fused = _haps_mod.reconstruct_haplotypes_fused
    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_fused", _spy_fused)

    # --- rust read (spy active) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    # Spy-wiring guard: capture count right after rust read.
    rust_call_count = calls["n"]

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # Spy-wiring guard: numba must NOT fire the fused spy.
    assert calls["n"] == rust_call_count, (
        f"reconstruct_haplotypes_fused spy fired during the numba read "
        f"(count went from {rust_call_count} to {calls['n']}) — "
        "the fused spy is being triggered by the numba path, which is a bug."
    )

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust reconstruct_haplotypes_fused was NEVER invoked during the "
        f"rust read (calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the haplotypes read path to confirm "
        "reconstruct_haplotypes_fused is called on the non-splice rust path "
        "in _haps._reconstruct_haplotypes."
    )

    # --- sanity: output must be non-trivial ---
    # out_rust is Ragged[np.bytes_] (ragged haplotype sequences)
    out_rust_data = np.asarray(out_rust.data)
    n_bases = out_rust_data.size
    assert n_bases > 0, (
        "Haplotypes output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    # Haplotypes should contain real bases, not just 'N' padding.
    n_pad = np.uint8(ord("N"))
    data_u8 = out_rust_data.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Haplotypes output is entirely 'N' padding — regions may fall outside "
        "the reference contigs.  Non-padding bases are required to prove the "
        "comparison is meaningful."
    )

    # --- byte-identical comparison ---
    _compare_ragged_bytes(out_numba, out_rust, name="haplotypes")


# ---------------------------------------------------------------------------
# Annotated backstop — "annotated" mode
# ---------------------------------------------------------------------------


def test_annotated_haplotypes_mode_dataset_parity(
    phased_svar_gvl, reference, monkeypatch
):
    """Flips GVL_BACKEND numba<->rust through the real annotated getitem path.

    Covers the annotated path (with_seqs("annotated")), which routes through
    _reconstruct_annotated_haplotypes and passes non-None annot_v_idxs and
    annot_ref_pos to reconstruct_haplotypes_from_sparse.  The spy asserts that
    the Rust kernel is actually invoked.  All three arrays — haps, var_idxs,
    and ref_coords — are compared byte-identically between backends.

    The return type is RaggedAnnotatedHaps with fields:
      .haps       — Ragged[np.bytes_]
      .var_idxs   — Ragged[np.int32]
      .ref_coords — Ragged[np.int32]
    """
    # --- open dataset in annotated mode ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("annotated")

    # --- install spy on the fused Rust reconstruct_annotated_haplotypes_fused entry ---
    # After Task 4, the non-splice rust path calls reconstruct_annotated_haplotypes_fused
    # (module-level name in _haps_mod) rather than the composed dispatched
    # reconstruct_haplotypes_from_sparse.  The numba path goes through the
    # composed dispatch and never calls reconstruct_annotated_haplotypes_fused.
    orig_fused = _haps_mod.reconstruct_annotated_haplotypes_fused
    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_annotated_haplotypes_fused", _spy_fused)

    # --- rust read (spy active) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    rust_call_count = calls["n"]

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # Spy-wiring guard: numba must NOT fire the fused spy.
    assert calls["n"] == rust_call_count, (
        f"reconstruct_annotated_haplotypes_fused spy fired during the numba read "
        f"(count went from {rust_call_count} to {calls['n']}) — "
        "the fused spy is being triggered by the numba path, which is a bug."
    )

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust reconstruct_annotated_haplotypes_fused was NEVER invoked during the "
        f"rust read (calls={calls['n']}) — the annotated backstop is vacuous. "
        "Inspect the annotated read path to confirm "
        "reconstruct_annotated_haplotypes_fused is called on the non-splice rust path "
        "in _haps._reconstruct_annotated_haplotypes."
    )

    # --- type sanity ---
    assert isinstance(out_rust, RaggedAnnotatedHaps), (
        f"Expected RaggedAnnotatedHaps from annotated mode, got {type(out_rust)}"
    )
    assert isinstance(out_numba, RaggedAnnotatedHaps), (
        f"Expected RaggedAnnotatedHaps from annotated mode, got {type(out_numba)}"
    )

    # --- sanity: output must be non-trivial ---
    rust_haps_data = np.asarray(out_rust.haps.data)
    n_bases = rust_haps_data.size
    assert n_bases > 0, (
        "Annotated haplotypes output contains zero bytes — regions don't overlap "
        "any reference sequence.  The parity comparison is vacuous."
    )
    data_u8 = rust_haps_data.view(np.uint8)
    n_pad = np.uint8(ord("N"))
    assert np.any(data_u8 != n_pad), (
        "Annotated haplotypes output is entirely 'N' padding — regions may fall "
        "outside the reference contigs.  Non-padding bases are required to prove "
        "the comparison is meaningful."
    )

    # --- byte-identical comparison of all three arrays ---
    _compare_ragged_bytes(out_numba.haps, out_rust.haps, name="annotated.haps")
    _compare_ragged_int(
        out_numba.var_idxs, out_rust.var_idxs, name="annotated.var_idxs"
    )
    _compare_ragged_int(
        out_numba.ref_coords, out_rust.ref_coords, name="annotated.ref_coords"
    )
