from pathlib import Path

import genvarloader as gvl
import seqpro as sp
from genoray import PGEN, VCF


def test_write_errors_when_post_index_budget_too_small(
    tmp_path, monkeypatch, vcf_dir: Path, source_bed: Path
):
    """If max_mem minus the variant index leaves no room for even one
    variant chunk, gvl.write raises ValueError instead of silently
    blowing the budget."""
    import pytest

    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()

    bed = sp.bed.read(source_bed)

    # Force nbytes large enough that effective_max_mem < bytes_per_var.
    # bytes_per_var = n_samples * ploidy (VCF, Genos8 = 1 byte).
    # Set nbytes = max_mem so effective_max_mem == 0.
    max_mem = 4 * 1024 * 1024
    monkeypatch.setattr(type(vcf), "nbytes", property(lambda self: max_mem))

    out = tmp_path / "test.gvl"
    with pytest.raises(ValueError, match="max_mem"):
        gvl.write(out, bed, vcf, max_mem=max_mem)


def test_write_loads_lazy_vcf_index(tmp_path, vcf_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a VCF constructed
    with with_gvi_index=False, and produce a valid dataset."""
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz", with_gvi_index=False)
    assert vcf._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, vcf)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()


def test_write_loads_lazy_pgen_index(tmp_path, pgen_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a PGEN constructed
    with load_index=False, and produce a valid dataset."""
    pgen = PGEN(pgen_dir / "filtered_source.pgen", load_index=False)
    assert pgen._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, pgen)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()
