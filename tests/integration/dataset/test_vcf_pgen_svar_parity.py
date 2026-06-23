"""Acceptance test: VCF and PGEN datasets must produce sparse variant output
identical to the SVAR dataset built from the same variants + BED.

This is the guarantee behind routing VCF/PGEN writes through genoray's
`_dense2sparse_with_length` (per-haplotype-minimal, matching SVAR)."""

import numpy as np
import pytest

import genvarloader as gvl


def _all_variants(path, reference):
    ds = gvl.Dataset.open(path, reference=reference).with_seqs("variants")
    r = np.arange(ds.n_regions)
    s = np.arange(ds.n_samples)
    # full cartesian index: every region x every sample
    rv = ds[np.repeat(r, len(s)), np.tile(s, len(r))]
    return rv


def _assert_nonempty(rv, label: str):
    """Guard against a vacuous pass: if the ground-truth ever regenerated to
    empty sparse output, an all-empty-vs-all-empty comparison would pass
    meaninglessly. Require real, multi-variant data so the trimming/extension
    in `_dense2sparse_with_length` is genuinely exercised."""
    # rv["start"] is a _core.Ragged of shape (b, p, ~v); .lengths is (b, p)
    counts_matrix = rv["start"].lengths  # shape (b, p) of int64
    flat = counts_matrix.ravel().tolist()
    total = sum(flat)
    assert total > 0, f"{label}: expected non-empty variant data, got 0 total variants"
    assert max(flat) > 1, (
        f"{label}: expected at least one haplotype with >1 variant (got max={max(flat)})"
    )


@pytest.mark.parametrize("field", ["start", "ilen", "alt"])
def test_vcf_matches_svar(phased_vcf_gvl, phased_svar_gvl, reference, field):
    vcf_rv = _all_variants(phased_vcf_gvl, reference)
    svar_rv = _all_variants(phased_svar_gvl, reference)
    _assert_nonempty(svar_rv, "svar")
    assert vcf_rv[field].to_ak().to_list() == svar_rv[field].to_ak().to_list()


@pytest.mark.parametrize("field", ["start", "ilen", "alt"])
def test_pgen_matches_svar(phased_pgen_gvl, phased_svar_gvl, reference, field):
    pgen_rv = _all_variants(phased_pgen_gvl, reference)
    svar_rv = _all_variants(phased_svar_gvl, reference)
    _assert_nonempty(svar_rv, "svar")
    assert pgen_rv[field].to_ak().to_list() == svar_rv[field].to_ak().to_list()
