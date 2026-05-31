"""Acceptance test: VCF and PGEN datasets must produce sparse variant output
identical to the SVAR dataset built from the same variants + BED.

This is the guarantee behind routing VCF/PGEN writes through genoray's
`_dense2sparse_with_length` (per-haplotype-minimal, matching SVAR)."""

import awkward as ak
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
    counts = ak.to_list(ak.num(rv["start"], axis=-1))
    flat = [c for hap in counts for c in (hap if isinstance(hap, list) else [hap])]
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
    assert ak.to_list(vcf_rv[field]) == ak.to_list(svar_rv[field])


@pytest.mark.parametrize("field", ["start", "ilen", "alt"])
def test_pgen_matches_svar(phased_pgen_gvl, phased_svar_gvl, reference, field):
    pgen_rv = _all_variants(phased_pgen_gvl, reference)
    svar_rv = _all_variants(phased_svar_gvl, reference)
    _assert_nonempty(svar_rv, "svar")
    assert ak.to_list(pgen_rv[field]) == ak.to_list(svar_rv[field])
