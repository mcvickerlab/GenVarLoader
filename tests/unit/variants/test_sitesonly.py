"""Unit tests for ``_variants/_sitesonly.py``.

Targets ``sites_vcf_to_table`` (the public helper that converts a sites-only
VCF into a table of bi-allelic SNP records) and the empty-overlap guard in
``DatasetWithSites.__init__``. The numba kernel ``apply_site_only_variants``
is exercised end-to-end by ``tests/integration/variants/test_sites.py``.
"""

from pathlib import Path

import genvarloader as gvl
import polars as pl
import pytest


@pytest.fixture(scope="module")
def sites_vcf(vcf_dir: Path) -> Path:
    """A pre-normalized, bi-allelic VCF suitable for sites-only ingestion."""
    return vcf_dir / "filtered_source.vcf.gz"


def test_sites_vcf_to_table_default_no_info_fields(sites_vcf: Path):
    """Default path: info_fields=None, attributes=None — only CHROM/POS/REF/ALT.

    Covers the ``attributes is None`` branch (line 43-44) and the
    ``info_fields=None`` branch of ``vcf.get_record_info``.
    """
    df = gvl.sites_vcf_to_table(sites_vcf)
    assert isinstance(df, pl.DataFrame)
    # Mandatory columns always present.
    for col in ("CHROM", "POS", "REF", "ALT"):
        assert col in df.columns
    # ALT was flattened from list[str] to str (line 53).
    assert df.schema["ALT"] == pl.String
    assert df.height > 0


def test_sites_vcf_to_table_with_info_fields(sites_vcf: Path):
    """Loading with explicit INFO fields requested.

    Covers the ``info_fields`` non-None branch (line 48 with info=info_fields).
    ``filtered_source.vcf.gz`` defines NS, DP, AF in its INFO header. Note that
    genoray currently passes ``info`` to oxbow but does not surface the INFO
    fields as top-level columns; we just verify the call succeeds and returns
    the mandatory columns.
    """
    df = gvl.sites_vcf_to_table(sites_vcf, info_fields=["AF"])
    assert isinstance(df, pl.DataFrame)
    # Mandatory columns still present.
    for col in ("CHROM", "POS", "REF", "ALT"):
        assert col in df.columns
    assert df.height > 0


def test_sites_vcf_to_table_extra_attributes_deduplicated(sites_vcf: Path):
    """Custom attributes list is merged with the mandatory four without dupes
    (line 46: ``attr for attr in attributes if attr not in min_attrs``)."""
    df = gvl.sites_vcf_to_table(sites_vcf, attributes=["CHROM", "POS"])
    # No duplicate columns from passing already-required attrs.
    assert len(df.columns) == len(set(df.columns))
    for col in ("CHROM", "POS", "REF", "ALT"):
        assert col in df.columns


def test_dataset_with_sites_empty_overlap_raises():
    """``DatasetWithSites`` raises when sites and dataset regions don't overlap.

    Covers ``_sitesonly.py`` line 183-184 (``if rows.height == 0: raise``).
    """
    ds = gvl.get_dummy_dataset().with_len(4).with_tracks(False)
    # Build a sites table on a contig that exists but at a coordinate far from
    # the dummy dataset's regions (which sit at positions < 20 on chrs 2/3/6/8).
    sites = pl.DataFrame(
        {
            "CHROM": ["8"],
            "POS": [10_000_000],
            "REF": ["A"],
            "ALT": ["T"],
        }
    )
    with pytest.raises(RuntimeError, match="No overlap"):
        gvl.DatasetWithSites(ds, sites)
