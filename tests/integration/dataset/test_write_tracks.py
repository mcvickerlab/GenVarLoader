from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader._table import Table

ddir = Path(__file__).parents[2] / "data"


def _make_bed(tmp_path: Path) -> pl.DataFrame:
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 100],
            "chromEnd": [50, 200],
        }
    )
    return bed


def _make_table_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": ["s0", "s0", "s1", "s1"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [10, 110, 5, 150],
            "end": [20, 130, 15, 160],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_write_with_table_only_roundtrip(tmp_path):
    bed = _make_bed(tmp_path)
    table = Table("signal", _make_table_df())

    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, tracks=table)

    # Sanity: the dataset directory has the expected per-track folder.
    assert (out / "intervals" / "signal" / "intervals.npy").exists()
    assert (out / "intervals" / "signal" / "offsets.npy").exists()

    # Read intervals back and confirm values round-trip.
    INTERVAL_DTYPE = np.dtype(
        [("start", np.int32), ("end", np.int32), ("value", np.float32)],
        align=True,
    )
    arr = np.memmap(
        out / "intervals" / "signal" / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r"
    )
    # Both samples + both regions should produce 4 intervals total.
    assert arr.shape[0] == 4
    values = sorted(float(v) for v in arr["value"])
    assert values == [1.0, 2.0, 3.0, 4.0]


def test_write_with_mixed_bigwigs_and_table(tmp_path):
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [200],
        }
    )
    bw_dir = ddir / "bigwig"
    bw = gvl.BigWigs(
        "bw_signal",
        {
            "sample_0": str(bw_dir / "sample_0.bw"),
            "sample_1": str(bw_dir / "sample_1.bw"),
        },
    )
    # Table sample IDs match the BigWigs sample IDs so the intersection is non-empty.
    table = gvl.Table(
        "tab_signal",
        pl.DataFrame(
            {
                "sample_id": ["sample_0", "sample_1"],
                "chrom": ["chr1", "chr1"],
                "start": [0, 50],
                "end": [10, 60],
                "value": [9.0, 8.0],
            }
        ),
    )

    out = tmp_path / "mixed.gvl"
    gvl.write(path=out, bed=bed, tracks=[bw, table])

    assert (out / "intervals" / "bw_signal" / "intervals.npy").exists()
    assert (out / "intervals" / "tab_signal" / "intervals.npy").exists()


def test_write_with_variants_and_tracks(tmp_path):
    """gvl.write() should succeed when both variants and tracks are provided."""
    from genoray import VCF

    vcf = VCF(ddir / "vcf" / "filtered_source.vcf.gz")
    # VCF samples are NA00001, NA00002, NA00003 — Table must share at least one.
    table = gvl.Table(
        "signal",
        pl.DataFrame(
            {
                "sample_id": ["NA00001", "NA00002", "NA00003"],
                "chrom": ["chr19", "chr19", "chr19"],
                "start": [1010686, 1010686, 1010686],
                "end": [1010706, 1010706, 1010706],
                "value": [1.0, 2.0, 3.0],
            }
        ),
    )
    bed = pl.DataFrame(
        {
            "chrom": ["chr19"],
            "chromStart": [1010686],
            "chromEnd": [1010706],
        }
    )

    out = tmp_path / "variants_and_tracks.gvl"
    gvl.write(path=out, bed=bed, variants=vcf, tracks=table)

    assert (out / "genotypes").is_dir()
    assert (out / "intervals" / "signal" / "intervals.npy").exists()
    assert (out / "intervals" / "signal" / "offsets.npy").exists()

    import json

    meta = json.loads((out / "metadata.json").read_text())
    assert set(meta["samples"]) == {"NA00001", "NA00002", "NA00003"}


def test_write_duplicate_track_names_rejected(tmp_path):
    import pytest

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    t1 = gvl.Table(
        "dup",
        pl.DataFrame(
            {
                "sample_id": ["s0"],
                "chrom": ["chr1"],
                "start": [0],
                "end": [10],
                "value": [1.0],
            }
        ),
    )
    t2 = gvl.Table(
        "dup",
        pl.DataFrame(
            {
                "sample_id": ["s0"],
                "chrom": ["chr1"],
                "start": [50],
                "end": [60],
                "value": [2.0],
            }
        ),
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        gvl.write(path=tmp_path / "x.gvl", bed=bed, tracks=[t1, t2])
