"""Correctness of the trailing-fill clause when a deletion exhausts the contig.

The overshoot sub-domain (ref_idx past contig end with output unfilled) was
historically excluded from parity because numba and rust diverged AND both were
wrong. Correct behavior: pad the entire unfilled tail (no reference left).
"""

import numpy as np

from genvarloader._dataset._genotypes import reconstruct_haplotype_from_sparse


def test_overshoot_pads_full_tail():
    # ref=[1,2,3,4], deletion at pos 2 (ilen=-5) -> ref_idx advances to 8 (>4).
    # out_len=8: [1,2] ref + [50] allele, then ref exhausted -> pad rest with 0.
    out = np.full(8, 255, dtype=np.uint8)  # 0xFF sentinel: catches unwritten positions
    reconstruct_haplotype_from_sparse(
        np.array([0], dtype=np.int32),  # v_idxs
        np.array([2], dtype=np.int32),  # v_starts
        np.array([-5], dtype=np.int32),  # ilens
        0,  # shift
        np.array([50], dtype=np.uint8),  # alt_alleles
        np.array([0, 1], dtype=np.int64),  # alt_offsets
        np.array([1, 2, 3, 4], dtype=np.uint8),  # ref
        0,  # ref_start
        out,  # out
        0,  # pad_char
    )
    np.testing.assert_array_equal(
        out, np.array([1, 2, 50, 0, 0, 0, 0, 0], dtype=np.uint8)
    )
