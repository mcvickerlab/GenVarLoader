"""Reference-mode dataset-level parity backstop.

Proves that the Rust get_reference kernel produces byte-identical output
matching the frozen golden (generated from the rust implementation,
oracle-verified against the composed numba pipeline at gen time).

Kernel exercised end-to-end:
  - get_reference  (reference fetch, via make_kernel_spy)
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
import genvarloader._dataset._reference  # noqa: F401 — triggers register("get_reference")

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_reference_mode_dataset_parity(phased_svar_gvl, reference):
    """Rust get_reference output matches the frozen golden.

    The spy asserts that the Rust get_reference kernel is actually invoked
    (non-vacuous guard).  The ragged output is compared byte-identically
    against the golden, and a non-triviality check ensures the comparison is
    meaningful (output is not all-padding).
    """
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("reference")

    # --- install counting spy via make_kernel_spy ---
    spy_fn, calls, restore = _golden.make_kernel_spy("get_reference")
    try:
        # --- read (default rust backend, spy active) ---
        out = ds[:, :]
    finally:
        restore()

    # --- anti-vacuous guard ---
    assert calls["n"] > 0, (
        f"Rust get_reference was NEVER invoked during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the reference read path to confirm get_reference is still "
        "dispatched via _dispatch.get on the Dataset.__getitem__ → "
        "_getitem_unspliced code path."
    )

    # --- sanity: output must be non-trivial ---
    out_arr = np.asarray(out.data)
    n_bases = out_arr.size
    assert n_bases > 0, (
        "Reference output contains zero bytes — regions don't overlap any "
        "reference sequence.  The parity comparison is vacuous."
    )
    n_pad = np.uint8(ord("N"))
    data_u8 = out_arr.view(np.uint8)
    assert np.any(data_u8 != n_pad), (
        "Reference output is entirely 'N' padding — regions may fall outside "
        "the reference contigs.  Non-padding bases are required to prove the "
        "comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(out, _golden.load_flat_golden("ds_reference_mode"))
