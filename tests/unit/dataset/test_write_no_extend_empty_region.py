"""Regression test: writing with extend_to_length=False must not crash when a
region contains no variants (the no-extend VCF path previously yielded an empty
chunk list that broke downstream ak.concatenate)."""

import polars as pl
from genoray import VCF

import genvarloader as gvl


def test_no_extend_write_handles_empty_region(vcf_dir, tmp_path):
    vcf_path = vcf_dir / "filtered_source.vcf.gz"
    v = VCF(vcf_path)
    if v._index is None:
        if v._valid_index():
            v._load_index()
        else:
            v._write_gvi_index()
            v._load_index()
    contig = v._index["CHROM"].unique().to_list()[0]
    first_pos = int(v._index.filter(pl.col("CHROM") == contig)["POS"].min())

    # region 0 has variants; region 1 is far away with none
    bed = pl.DataFrame(
        {
            "chrom": [contig, contig],
            "chromStart": [first_pos - 10, first_pos + 10_000_000],
            "chromEnd": [first_pos + 50, first_pos + 10_000_050],
        }
    )

    out = tmp_path / "noext.gvl"

    # must not raise
    gvl.write(
        path=out,
        bed=bed,
        variants=VCF(vcf_path),
        overwrite=True,
        extend_to_length=False,
    )

    # verify the dataset was written with both regions (read metadata directly
    # because Dataset.open without a reference cannot build sequence output)
    import json

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["n_regions"] == 2
