from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader import Table


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

    # Sanity: the dataset directory has the expected per-track SoA files.
    sig_dir = out / "intervals" / "signal"
    for name in ("starts.npy", "ends.npy", "values.npy", "offsets.npy"):
        assert (sig_dir / name).exists()

    # Read intervals back and confirm values round-trip.
    starts = np.memmap(sig_dir / "starts.npy", dtype=np.int32, mode="r")
    ends = np.memmap(sig_dir / "ends.npy", dtype=np.int32, mode="r")
    values = np.memmap(sig_dir / "values.npy", dtype=np.float32, mode="r")
    # Both samples + both regions should produce 4 intervals total.
    assert len(starts) == 4
    assert len(ends) == 4
    assert len(values) == 4
    assert sorted(float(v) for v in values) == [1.0, 2.0, 3.0, 4.0]


def test_write_with_mixed_bigwigs_and_table(tmp_path, bigwig_dir: Path):
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [200],
        }
    )
    bw_dir = bigwig_dir
    bw = gvl.BigWigs(
        "bw_signal",
        {
            "sample_0": str(bw_dir / "sample_0.bw"),
            "sample_1": str(bw_dir / "sample_1.bw"),
        },
    )
    # Table sample IDs match the BigWigs sample IDs so the intersection is non-empty.
    table = Table(
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

    for track_name in ("bw_signal", "tab_signal"):
        track_dir = out / "intervals" / track_name
        for name in ("starts.npy", "ends.npy", "values.npy", "offsets.npy"):
            assert (track_dir / name).exists()


def test_write_with_variants_and_tracks(tmp_path, vcf_dir: Path):
    """gvl.write() should succeed when both variants and tracks are provided."""
    from genoray import VCF

    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    # VCF samples are s0, s1, s2 — Table must share at least one.
    table = Table(
        "signal",
        pl.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2"],
                "chrom": ["chr1", "chr1", "chr1"],
                "start": [1010686, 1010686, 1010686],
                "end": [1010706, 1010706, 1010706],
                "value": [1.0, 2.0, 3.0],
            }
        ),
    )
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [1010686],
            "chromEnd": [1010706],
        }
    )

    out = tmp_path / "variants_and_tracks.gvl"
    gvl.write(path=out, bed=bed, variants=vcf, tracks=table)

    assert (out / "genotypes").is_dir()
    sig_dir = out / "intervals" / "signal"
    for name in ("starts.npy", "ends.npy", "values.npy", "offsets.npy"):
        assert (sig_dir / name).exists()

    import json

    meta = json.loads((out / "metadata.json").read_text())
    assert set(meta["samples"]) == {"s0", "s1", "s2"}
