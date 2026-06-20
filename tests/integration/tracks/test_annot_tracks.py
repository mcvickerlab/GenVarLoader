import awkward as ak
import genvarloader as gvl
import polars as pl
from genoray import VCF


def test_write_with_annot_tracks(vcf_dir, bigwig_dir, ref_fasta, tmp_path):
    out = tmp_path / "ds"
    # chr1:100-200 overlaps the first variant at POS 111
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(
        out, bed, variants=vcf, annot_tracks={"sig": str(bigwig_dir / "sample_0.bw")}
    )
    ds = (
        gvl.Dataset.open(out, ref_fasta)
        .with_seqs("annotated")
        .with_tracks("sig", "tracks")
    )
    assert "sig" in ds.available_tracks


def test_annot_tracks(vcf_dir, ref_fasta, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(out, bed, variants=vcf)
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated")
    annots = ds.regions.with_columns(
        chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0)
    )
    gvl.update(out, annot_tracks={"5ss": annots})
    annot_ds = (
        gvl.Dataset.open(out, ref_fasta)
        .with_seqs("annotated")
        .with_tracks("5ss", "tracks")
    )
    haps, tracks = annot_ds[:]
    mask = haps.ref_coords == ak.Array(
        annot_ds.regions["chromStart"].to_numpy()[:, None, None]
    )
    assert ak.all(tracks[:, :, 0][mask] == 1)


def test_annot_bigwig_wide_intervals_full_width(tmp_path):
    """#233: annot bigWig intervals wider than the query region must expand to
    their full width on readback, not collapse to value/span_length."""
    import numpy as np
    import pyBigWig
    from genvarloader._bigwig import BigWigs

    bw_path = tmp_path / "binned.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr22", 50_000_000)])
    # 1 kb bins: each region below falls inside one bin (interval wider than region)
    starts = list(range(16_770_000, 16_832_000, 1000))
    ends = [s + 1000 for s in starts]
    vals = [float((s // 1000) % 7) * 0.1 + 0.2 for s in starts]
    bw.addEntries(["chr22"] * len(starts), starts, ends=ends, values=vals)
    bw.close()

    spans = [(16_774_654, 16_775_004), (16_829_376, 16_829_599)]
    bed = pl.DataFrame(
        {
            "chrom": ["chr22"] * len(spans),
            "chromStart": [s for s, _ in spans],
            "chromEnd": [e for _, e in spans],
            "name": [f"r{i}" for i in range(len(spans))],
        }
    )

    out = tmp_path / "repro.gvl"
    gvl.write(
        out,
        bed=bed,
        tracks=[BigWigs("dummy", {"s0": str(bw_path)})],
        annot_tracks={"t": str(bw_path)},
        overwrite=True,
    )
    ds = gvl.Dataset.open(out)
    fr = (
        ds.with_seqs(None)
        .with_output_format("flat")
        .with_settings(realign_tracks=False)
        .with_tracks(["t"])
    )[0 : ds.n_regions, 0]
    data, offs = np.asarray(fr.data), np.asarray(fr.offsets)

    src = pyBigWig.open(str(bw_path))
    for i, (s, e) in enumerate(spans):
        sv = src.values("chr22", s, e, numpy=True)
        sv = sv[~np.isnan(sv)]
        seg = data[offs[i] : offs[i + 1]]
        # full width filled (catches the collapse), value correct
        assert np.count_nonzero(seg) == (e - s)
        assert np.isclose(np.nanmean(seg), sv.mean(), atol=1e-4)
