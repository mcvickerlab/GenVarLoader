"""Direct probe for #176 Bug 1: with_settings(var_filter=...) must update _recon, not just _seqs.

After .with_seqs("haplotypes"), _recon is a separate Haps instance
(haps.to_kind(RaggedSeqs) returns a fresh instance via evolve). The
old with_settings only evolved self._seqs, silently dropping the
filter from the code path that __getitem__ actually exercises.
"""

from pathlib import Path

import genvarloader as gvl
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"


@pytest.fixture
def svar_gvl_path(tmp_path):
    svar_path = _DATA_DIR / "filtered.svar"
    bed_path = _DATA_DIR / "source.bed"
    assert svar_path.is_dir(), f"missing fixture {svar_path}; run pixi run -e dev gen"
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed_path, variants=svar_path, overwrite=True)
    return out


def test_with_settings_var_filter_propagates_to_recon(svar_gvl_path):
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset.open(svar_gvl_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
    )
    assert ds._seqs.filter == "exonic"
    assert ds._recon.filter == "exonic", (
        "with_settings(var_filter=...) failed to propagate to _recon; "
        "__getitem__ uses _recon, so the filter would be silently dropped."
    )


def test_with_settings_var_filter_false_clears_recon(svar_gvl_path):
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset.open(svar_gvl_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
        .with_settings(var_filter=False)
    )
    assert ds._seqs.filter is None
    assert ds._recon.filter is None
