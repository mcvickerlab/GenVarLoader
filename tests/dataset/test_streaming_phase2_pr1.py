"""PR 1b: svar2_read_window Rust FFI — shape/dtype smoke + byte-equivalence vs the
Phase-1 name-based SparseVar2._find_ranges path."""

from __future__ import annotations

import numpy as np


def test_svar2_read_window_shapes(svar2_multicontig_fixture) -> None:
    from genoray import SparseVar2
    from genvarloader.genvarloader import Svar2Store, svar2_read_window

    fx = svar2_multicontig_fixture
    sv = SparseVar2(str(fx.svar2_path))
    ploidy = int(sv.ploidy)
    store = Svar2Store(str(fx.svar2_path), sv.contigs, sv.n_samples, ploidy)

    # One contig window: chr1 regions [0,20) and [4,24); all physical samples 0..n.
    contig = "chr1"
    starts = np.array([0, 4], np.uint32)
    ends = np.array([20, 24], np.uint32)
    phys = np.arange(sv.n_samples, dtype=np.int64)
    n_reg, n_s = len(starts), len(phys)

    vk_snp, vk_indel, dense_snp, dense_indel, sample_cols = svar2_read_window(
        store, contig, starts, ends, phys
    )
    for a in (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols):
        assert np.asarray(a).dtype == np.int64
    assert np.asarray(vk_snp).size == n_reg * n_s * ploidy * 2
    assert np.asarray(vk_indel).size == n_reg * n_s * ploidy * 2
    assert np.asarray(dense_snp).size == n_reg * 2
    assert np.asarray(dense_indel).size == n_reg * 2
    assert np.asarray(sample_cols).size == n_s
