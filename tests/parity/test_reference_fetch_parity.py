"""Parity backstop for Reference.fetch (rerouted through dispatched get_reference).

fetch builds regions=(contig_idx, start, end) and out_offsets, then calls the
same get_reference core used by the main reference read path. This test asserts
that the rust get_reference kernel is actually invoked (spy guard) and that the
output matches the frozen golden.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader._dataset._reference  # noqa: F401 — triggers register("get_reference")

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_reference_fetch_parity(reference):
    ref = reference
    contigs = ref.contigs[:1]
    starts = np.array([0], dtype=np.int64)
    ends = np.array([50], dtype=np.int64)

    spy_fn, calls, restore = _golden.make_kernel_spy("get_reference")
    try:
        out = ref.fetch(contigs, starts, ends)
    finally:
        restore()

    assert calls["n"] > 0, "rust get_reference never invoked via fetch — vacuous"

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden("ds_reference_fetch")
    )
