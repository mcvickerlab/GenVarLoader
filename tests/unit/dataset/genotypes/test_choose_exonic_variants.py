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
import pytest
from genvarloader._dataset._genotypes import choose_exonic_variants


def _common_inputs() -> dict[str, np.ndarray]:
    """Tiny shared fixture: 1 region, ploidy 2, 2 variants, both exonic."""
    return {
        "starts": np.asarray([0], dtype=np.int32),
        "ends": np.asarray([100], dtype=np.int32),
        "geno_offset_idx": np.asarray([[0, 1]], dtype=np.intp),
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
    """SVAR offsets are shape (2, n_slices). Wrong indexing returns a
    length-n_slices row and produces garbage o_s/o_e. Use n_slices > 2
    so wrong indexing cannot accidentally yield 2 elements.
    """
    inputs = _common_inputs()
    # Real SVAR layout: shape (2, n_slices). Row 0 is starts, row 1 is ends.
    # 3 slices total; region 0, haplotype 0 -> slice 0 = [0, 1).
    # region 0, haplotype 1 -> slice 1 = [1, 2). Slice 2 is unused padding.
    inputs["geno_offsets"] = np.asarray(
        [
            [0, 1, 2],  # starts
            [1, 2, 2],  # ends
        ],
        dtype=np.int64,
    )
    keep, keep_offsets = choose_exonic_variants(**inputs)
    # Logical content identical to the 1-D case: 2 variants kept.
    assert keep_offsets.shape == (3,)  # n_regions * ploidy + 1
    assert keep_offsets.tolist() == [0, 1, 2]
    assert keep.tolist() == [True, True]


# ---------------------------------------------------------------------------
# Boundary / containment scenarios for the inner filter rule:
#   keep iff v_pos >= query_start AND v_ref_end <= query_end
#   where v_ref_end = v_pos - min(0, ilen) + 1
#
# Region used below: start=10, end=50 (single haplotype, 1-D offsets).
# ---------------------------------------------------------------------------


def _single_variant_inputs(
    v_pos: int,
    ilen: int,
    query_start: int = 10,
    query_end: int = 50,
) -> dict[str, np.ndarray]:
    """Build inputs for a single-variant, single-haplotype, single-region query."""
    return {
        "starts": np.asarray([query_start], dtype=np.int32),
        "ends": np.asarray([query_end], dtype=np.int32),
        # 1 region, ploidy 1
        "geno_offset_idx": np.asarray([[0]], dtype=np.intp),
        "geno_v_idxs": np.asarray([0], dtype=np.int32),
        "geno_offsets": np.asarray([0, 1], dtype=np.int64),
        "v_starts": np.asarray([v_pos], dtype=np.int32),
        "ilens": np.asarray([ilen], dtype=np.int32),
    }


@pytest.mark.parametrize(
    "v_pos, ilen, expected_keep, scenario",
    [
        # (a) SNV fully inside region: v_pos=20, v_ref_end=21, region [10,50)
        #     20 >= 10 AND 21 <= 50 → True
        (20, 0, True, "snv_fully_inside"),
        # (b) Variant spans region start: deletion v_pos=5, ilen=-10
        #     v_ref_end = 5 - (-10) + 1 = 16; 5 >= 10 is False → False
        (5, -10, False, "variant_spans_region_start"),
        # (c) Variant spans region end: deletion v_pos=45, ilen=-10
        #     v_ref_end = 45 - (-10) + 1 = 56; 56 <= 50 is False → False
        (45, -10, False, "variant_spans_region_end"),
        # (d) Variant entirely before region: SNV v_pos=3, v_ref_end=4
        #     3 >= 10 is False → False
        (3, 0, False, "variant_entirely_before_region"),
        # (e) Variant entirely after region: SNV v_pos=60, v_ref_end=61
        #     61 <= 50 is False → False
        (60, 0, False, "variant_entirely_after_region"),
    ],
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_choose_exonic_variants_containment(
    v_pos: int, ilen: int, expected_keep: bool, scenario: str
) -> None:
    """Boundary containment: each scenario exercises one edge of the keep rule."""
    inputs = _single_variant_inputs(v_pos=v_pos, ilen=ilen)
    keep, _ = choose_exonic_variants(**inputs)
    assert keep.tolist() == [expected_keep], (
        f"scenario={scenario!r}: v_pos={v_pos}, ilen={ilen}, "
        f"v_ref_end={v_pos - min(0, ilen) + 1}, expected keep={expected_keep}"
    )
