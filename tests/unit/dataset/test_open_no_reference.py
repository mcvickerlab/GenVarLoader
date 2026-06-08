"""Regression: opening a genotypes-only dataset without a reference must
default to the 'variants' view (RaggedVariants), not crash trying to build
haplotypes. See docs/superpowers/specs/2026-06-07-open-variants-no-reference-design.md.
"""

from __future__ import annotations

import pytest

import genvarloader as gvl


@pytest.mark.parametrize(
    "fixture_name", ["phased_vcf_gvl", "phased_pgen_gvl", "phased_svar_gvl"]
)
def test_open_without_reference_defaults_to_variants(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    ds = gvl.Dataset.open(path)
    assert ds.sequence_type == "variants"


def test_open_without_reference_indexing_yields_variants(phased_vcf_gvl):
    ds = gvl.Dataset.open(phased_vcf_gvl)
    # List indices (no squeeze) return the batched sequence object directly;
    # scalar indices would squeeze to an awkward Record, so use lists here.
    out = ds[[0], [0]]
    assert isinstance(out, gvl.RaggedVariants)
