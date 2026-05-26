"""Cross-mode equivalence invariants.

Asserts that the same logical query returns byte-identical content across:

- variant sources (VCF / PGEN / SVAR),
- output containers (ragged vs fixed-length array, when the fixed length
  matches the ragged length), and
- sample addressing modes (name vs integer index).
"""

from itertools import product

import genvarloader as gvl
import numpy as np
import seqpro as sp


def _open_haps(path, ref):
    return (
        gvl.Dataset.open(path, ref, rc_neg=False)
        .with_tracks(False)
        .with_seqs("haplotypes")
    )


def test_vcf_pgen_svar_yield_identical_haplotypes(
    phased_vcf_gvl, phased_pgen_gvl, phased_svar_gvl, ref_fasta
):
    """The same underlying genotypes loaded via VCF, PGEN, and SVAR backends
    must produce byte-identical haplotypes for every (region, sample)."""
    vcf_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")
    pgen_ds = _open_haps(phased_pgen_gvl, ref_fasta).with_len("ragged")
    svar_ds = _open_haps(phased_svar_gvl, ref_fasta).with_len("ragged")

    assert vcf_ds.n_regions == pgen_ds.n_regions == svar_ds.n_regions
    assert list(vcf_ds.samples) == list(pgen_ds.samples) == list(svar_ds.samples)

    for region, sample in product(range(vcf_ds.n_regions), vcf_ds.samples):
        v = vcf_ds[region, sample]
        p = pgen_ds[region, sample]
        s = svar_ds[region, sample]

        np.testing.assert_array_equal(
            np.asarray(v.lengths),
            np.asarray(p.lengths),
            err_msg=f"VCF vs PGEN lengths mismatch region={region} sample={sample}",
        )
        np.testing.assert_array_equal(
            np.asarray(v.lengths),
            np.asarray(s.lengths),
            err_msg=f"VCF vs SVAR lengths mismatch region={region} sample={sample}",
        )

        v_seq = sp.cast_seqs(v.data)
        p_seq = sp.cast_seqs(p.data)
        s_seq = sp.cast_seqs(s.data)
        np.testing.assert_equal(
            v_seq, p_seq, err_msg=f"VCF vs PGEN region={region} sample={sample}"
        )
        np.testing.assert_equal(
            v_seq, s_seq, err_msg=f"VCF vs SVAR region={region} sample={sample}"
        )


def test_ragged_and_array_agree_on_ragged_length(phased_vcf_gvl, ref_fasta):
    """When every ploidy length at a (region, sample) equals a single value L,
    ``with_len(L)`` must return content identical to the ragged output reshaped
    to (ploidy, L)."""
    rag_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")

    # Find (region, sample) pairs whose two ploidy lengths agree and group by
    # that common length L. Then open one ArrayDataset per L and compare on the
    # restricted set of indices.
    by_length: dict[int, list[tuple[int, str]]] = {}
    for region, sample in product(range(rag_ds.n_regions), rag_ds.samples):
        lens = np.asarray(rag_ds[region, sample].lengths)
        if lens.min() == lens.max():
            by_length.setdefault(int(lens[0]), []).append((region, sample))

    assert by_length, "no (region, sample) pair has uniform ploidy lengths"

    compared = 0
    for length, pairs in by_length.items():
        arr_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len(length)
        for region, sample in pairs:
            rag = rag_ds[region, sample]
            arr = arr_ds[region, sample]
            ploidy = rag.shape[0]
            rag_dense = sp.cast_seqs(rag.data).reshape(ploidy, length)
            arr_seq = sp.cast_seqs(arr)
            np.testing.assert_equal(
                arr_seq,
                rag_dense,
                err_msg=(
                    f"ragged vs array(L={length}) mismatch "
                    f"region={region} sample={sample}"
                ),
            )
            compared += 1

    assert compared > 0


def test_sample_name_and_integer_index_agree(phased_vcf_gvl, ref_fasta):
    """Indexing a (region, sample) by sample name and by integer position must
    yield identical haplotypes."""
    ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")
    for region in range(ds.n_regions):
        for s_int, s_name in enumerate(ds.samples):
            by_name = ds[region, s_name]
            by_int = ds[region, s_int]
            np.testing.assert_array_equal(
                np.asarray(by_name.lengths),
                np.asarray(by_int.lengths),
                err_msg=(
                    f"lengths mismatch region={region} sample={s_name} "
                    f"(int={s_int})"
                ),
            )
            np.testing.assert_equal(
                sp.cast_seqs(by_name.data),
                sp.cast_seqs(by_int.data),
                err_msg=(
                    f"haplotype mismatch region={region} sample={s_name} "
                    f"(int={s_int})"
                ),
            )
