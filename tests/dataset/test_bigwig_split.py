import genvarloader as gvl
import polars as pl
from pathlib import Path


def test_bigwig_sample_split(tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "bigwig"
    bws = gvl.BigWigs(
        "signal",
        {"s0": str(data_dir / "sample_0.bw"), "s1": str(data_dir / "sample_1.bw")},
    )
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [150]})

    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, bigwigs=bws, max_mem=32)

    ds = gvl.Dataset.open(out)
    assert ds.shape == (1, 2)
