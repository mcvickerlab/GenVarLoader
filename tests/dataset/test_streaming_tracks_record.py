"""Rust seam for VCF/PGEN mixed variants+tracks (issue #375, Track B).

`window_realign_inputs` is the pymethod the Python mixed path uses to size a
window's deletion-extended track query before pulling the window's first
batch. Task 6 consumes it; this file pins its contract.
"""

from __future__ import annotations

import numpy as np

from genvarloader.genvarloader import RecordStreamEngine


def test_window_realign_inputs_matches_decode_and_csr_is_well_formed(
    vcf_snp_ins_del_multi,
):
    f = vcf_snp_ins_del_multi
    contig_len = int(f.regions["chromEnd"][0])
    ref_seq = "".join(f.fasta.read_text().splitlines()[1:])

    eng = RecordStreamEngine(
        "vcf",
        str(f.vcf),
        f.sample_names,
        f.ploidy,
        [f.contig],
        [ref_seq.encode()],
        [],
        [],
        [],
        [],
        [],
        None,  # fasta_path=None -- matches gvl.write's VCF parity (no read-time left-align)
        ord("N"),
        False,
        32,
        -1,  # output_length: ragged (unused by window_realign_inputs/debug_decode_window)
    )

    v_starts, ilens, geno_v_idxs, geno_offsets = eng.window_realign_inputs(
        0, [0], [contig_len], 0, f.n_samples
    )
    d_v_starts, d_ilens, _alt, _alt_off = eng.debug_decode_window(
        0, [0], [contig_len], 0, f.n_samples
    )

    # Same decode path -- the variant table must agree exactly.
    np.testing.assert_array_equal(v_starts, np.asarray(d_v_starts, np.int32))
    np.testing.assert_array_equal(ilens, np.asarray(d_ilens, np.int32))

    # Dtypes are part of the contract Task 6 relies on.
    assert v_starts.dtype == np.int32
    assert ilens.dtype == np.int32
    assert geno_v_idxs.dtype == np.int32
    assert geno_offsets.dtype == np.int64

    # CSR is per-hap over the sample sub-range, not per (region, sample) row.
    assert geno_offsets.shape == (f.n_samples * f.ploidy + 1,)
    assert geno_offsets[0] == 0
    assert geno_offsets[-1] == geno_v_idxs.size
    assert np.all(np.diff(geno_offsets) >= 0), "CSR offsets must be non-decreasing"

    # Every CSR value indexes a real column of this window's variant table.
    if geno_v_idxs.size:
        assert geno_v_idxs.min() >= 0
        assert geno_v_idxs.max() < v_starts.size

    # The fixture carries indels, so the window must actually contain variants --
    # otherwise every assertion above is vacuously true.
    assert v_starts.size > 0
    assert geno_v_idxs.size > 0
    assert np.any(ilens != 0), "fixture must contain indels for this seam to matter"
