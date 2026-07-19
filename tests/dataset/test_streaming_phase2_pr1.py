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


def test_svar2_read_window_matches_find_ranges(svar2_multicontig_fixture) -> None:
    """The rewired Rust read_window is byte-identical to the Phase-1 name-based
    SparseVar2._find_ranges path for the same window."""
    import genvarloader as gvl

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None

    # Reference (old) implementation: name-based _find_ranges, reshaped as Phase 1 did.
    def old_read_window(r_idx, s_idx):
        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0])
        ends = np.ascontiguousarray(rb[:, 1])
        names = [backend._sample_names[i] for i in s_idx]
        d = backend._sv._find_ranges(contig, starts, ends, samples=names)
        n_reg, n_s, P = len(r_idx), len(s_idx), backend.ploidy
        return {
            "orig_samples": np.ascontiguousarray(d["sample_cols"], np.int64),
            "vk_snp": np.asarray(d["vk_snp_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(d["vk_indel_range"], np.int64).reshape(
                n_reg, n_s, P, 2
            ),
            "dense_snp": np.asarray(d["dense_snp_range"], np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(d["dense_indel_range"], np.int64).reshape(
                n_reg, 2
            ),
        }

    for r_idx, s_idx in sds._plan():
        new = backend.read_window(r_idx, s_idx)  # rewired (Rust) path
        old = old_read_window(r_idx, s_idx)
        for k in ("orig_samples", "vk_snp", "vk_indel", "dense_snp", "dense_indel"):
            np.testing.assert_array_equal(
                np.asarray(new[k]), old[k], err_msg=f"mismatch in {k}"
            )
