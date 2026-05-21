"""Regression test for the choose_exonic_variants 2-D geno_offsets bug.

The function used to JIT-fail with
``TypingError: slice(array(int64, 1d, C), array(int64, 1d, C))`` when
``geno_offsets`` was 2-D, because the second prange loop indexed
``geno_offsets[o_idx]`` (returning a length-2 row, not scalars) and
then sliced ``geno_v_idxs[o_s:o_e]`` with those rows.

Mirror the fix in the first loop + the sibling ``filter_af`` kernel
which both branch on ``geno_offsets.ndim == 1``.
"""

from __future__ import annotations

import numpy as np

from genvarloader._dataset._genotypes import choose_exonic_variants


def _common_inputs() -> dict[str, np.ndarray]:
    """Tiny shared fixture: 1 region, ploidy 2, 2 variants, both exonic."""
    return {
        "starts": np.asarray([0], dtype=np.int32),
        "ends": np.asarray([100], dtype=np.int32),
        "geno_offset_idxs": np.asarray([[0, 1]], dtype=np.intp),
        # Two variants, indices 0 and 1, both inside [0, 100):
        "geno_v_idxs": np.asarray([0, 1], dtype=np.int32),
        "v_starts": np.asarray([10, 50], dtype=np.int32),
        "ilens": np.asarray([0, 0], dtype=np.int32),  # SNVs, length-0
    }


def test_choose_exonic_variants_1d_geno_offsets() -> None:
    """1-D geno_offsets always worked; lock the baseline output."""
    inputs = _common_inputs()
    # Shape (total_variants + 1,) -- the canonical 1-D layout.
    inputs["geno_offsets"] = np.asarray([0, 1, 2], dtype=np.int64)
    keep, keep_offsets = choose_exonic_variants(**inputs)
    assert keep.dtype == np.bool_
    assert keep_offsets.shape == (3,)  # n_regions * ploidy + 1 = 1 * 2 + 1
    # Both variants are inside [0, 100) so both kept.
    assert keep.tolist() == [True, True]


def test_choose_exonic_variants_2d_geno_offsets() -> None:
    """2-D geno_offsets used to JIT-fail at numba compile time."""
    inputs = _common_inputs()
    # Shape (total_variants, 2) -- each row is [o_s, o_e] for one variant.
    # Equivalent logical content to the 1-D fixture above.
    inputs["geno_offsets"] = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    keep, keep_offsets = choose_exonic_variants(**inputs)
    # Same output as the 1-D path -- the 2-D layout is just a different
    # internal storage shape; logical content is identical.
    assert keep_offsets.shape == (3,)
    assert keep.tolist() == [True, True]
