"""Unit tests for `_window_to_sparse`: the shared dense->sparse conversion that
dispatches between plain `dense2sparse` (no extension) and genoray's
`_dense2sparse_with_length` (per-haplotype-minimal extension)."""

import awkward as ak
import numpy as np
from genoray._types import V_IDX_TYPE

from genvarloader._dataset._write import _window_to_sparse


def _window():
    """A 1-sample, 2-haplotype window over query [0, 4).

    Two variants both starting inside the query:
      - v0 @ start=1, ILEN=-3 (3bp deletion)
      - v1 @ start=5, ILEN=0  (SNP) -- starts AFTER q_end=4; it is an
        extension variant only needed by a haplotype shortened by the deletion.
    hap0 carries both v0 and v1; hap1 carries only v1. Note hap1 carries v1 but
    has no deletion, so its length walk reaches q_end=4 without ever reaching
    v1's start=5 -- v1 is dropped for hap1 (expected []).
    """
    # (samples, ploidy, variants)
    genos = np.array([[[1, 1], [0, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=V_IDX_TYPE)
    v_starts = np.array([1, 5], dtype=np.int32)
    ilens = np.array([-3, 0], dtype=np.int32)
    q_start, q_end = 0, 4
    return genos, var_idxs, q_start, q_end, v_starts, ilens


def test_no_extend_keeps_all_carried_variants():
    genos, var_idxs, q_start, q_end, v_starts, ilens = _window()
    rag = _window_to_sparse(
        genos, var_idxs, q_start, q_end, v_starts, ilens, extend_to_length=False
    )
    # plain dense2sparse: every haplotype keeps exactly the variants it carries.
    # hap0 carries v0,v1 -> [0, 1]; hap1 carries v1 -> [1]
    assert ak.to_list(rag) == [[[0, 1], [1]]]


def test_extend_trims_per_haplotype_to_length():
    genos, var_idxs, q_start, q_end, v_starts, ilens = _window()
    rag = _window_to_sparse(
        genos, var_idxs, q_start, q_end, v_starts, ilens, extend_to_length=True
    )
    # hap0 was shortened by the 3bp deletion at v0, so it needs the extension
    # variant v1 to reach length 4 -> [0, 1].
    # hap1 carries v1 (a SNP at pos=5, past q_end=4) but has no deletion;
    # the length walk stops at q_end=4 without pulling in v1 (which starts
    # outside the query window). genoray returns [] for hap1.
    assert ak.to_list(rag) == [[[0, 1], []]]


def test_extend_drops_unneeded_extension_for_full_length_haplotype():
    """A haplotype with no deletion must not absorb extension variants it
    doesn't need (this is the over-extension bug the change fixes)."""
    # hap0: SNP only (no deletion); hap1: SNP only. Query [0,4).
    # v0 @ start=1 ILEN=0 (in-query SNP), v1 @ start=5 ILEN=0 (past q_end).
    genos = np.array([[[1, 1], [1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=V_IDX_TYPE)
    v_starts = np.array([1, 5], dtype=np.int32)
    ilens = np.array([0, 0], dtype=np.int32)
    rag = _window_to_sparse(
        genos, var_idxs, 0, 4, v_starts, ilens, extend_to_length=True
    )
    # Neither haplotype is shortened, so neither needs v1 (which starts at 5,
    # outside [0,4)). Both keep only v0.
    assert ak.to_list(rag) == [[[0], [0]]]
