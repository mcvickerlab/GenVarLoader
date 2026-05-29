"""Direct probe for #176 Bug 1: with_settings(var_filter=...) must update _recon, not just _seqs.

After .with_seqs("haplotypes"), _recon is a separate Haps instance
(haps.to_kind(RaggedSeqs) returns a fresh instance via evolve). The
old with_settings only evolved self._seqs, silently dropping the
filter from the code path that __getitem__ actually exercises.
"""

import genvarloader as gvl
import pytest


@pytest.fixture
def svar_gvl_path(tmp_path, filtered_svar, source_bed):
    assert filtered_svar.is_dir(), (
        f"missing fixture {filtered_svar}; run pixi run -e dev gen"
    )
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=source_bed, variants=filtered_svar, overwrite=True)
    return out


def test_with_settings_var_filter_propagates_to_recon(svar_gvl_path, ref_fasta):
    ds = (
        gvl.Dataset
        .open(svar_gvl_path, reference=ref_fasta)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
    )
    assert ds._seqs.filter == "exonic"
    assert ds._recon.filter == "exonic", (
        "with_settings(var_filter=...) failed to propagate to _recon; "
        "__getitem__ uses _recon, so the filter would be silently dropped."
    )


def test_with_settings_var_filter_false_clears_recon(svar_gvl_path, ref_fasta):
    ds = (
        gvl.Dataset
        .open(svar_gvl_path, reference=ref_fasta)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
        .with_settings(var_filter=False)
    )
    assert ds._seqs.filter is None
    assert ds._recon.filter is None
