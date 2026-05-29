"""Regression test for #176: Dataset.open(splice_info, var_filter) must produce
the same result as .with_seqs("haplotypes").with_settings(splice_info, var_filter).

Pre-fix:
- Path A (open(splice_info, var_filter)): hit choose_exonic_variants with a
  2-D SVAR offsets layout, which was mis-indexed -> MemoryError /
  negative-dim ValueError.
- Path B (with_settings): silently dropped var_filter from _recon and produced
  un-filtered haplotypes.

Post-fix: both paths produce the same, correctly-filtered output.
"""

import shutil

import awkward as ak
import genvarloader as gvl
import polars as pl
import pytest


@pytest.fixture(scope="module")
def spliced_svar_ds_path(tmp_path_factory, filtered_svar, source_bed):
    """Build an SVAR-backed GVL store with per-region single-exon transcripts.

    Each BED row becomes its own single-exon transcript so SpliceMap has
    something to do, and the SVAR backend ensures 2-D geno_offsets are
    exercised end-to-end.
    """
    assert filtered_svar.is_dir(), (
        f"missing fixture {filtered_svar}; run pixi run -e dev gen"
    )

    tmp = tmp_path_factory.mktemp("issue_176_parity")
    out = tmp / "ds.gvl"
    gvl.write(path=out, bed=source_bed, variants=filtered_svar, overwrite=True)

    # Inject transcript_id / exon_number so SpliceMap.from_bed can resolve.
    regions_path = out / "input_regions.arrow"
    bed = pl.read_ipc(regions_path)
    bed = bed.with_columns(
        transcript_id=pl.arange(0, pl.len()).cast(pl.Utf8),
        exon_number=pl.lit(1, pl.Int32),
    )
    tmp_arrow = regions_path.with_suffix(".arrow.tmp")
    bed.write_ipc(tmp_arrow)
    shutil.move(tmp_arrow, regions_path)
    return out


def test_open_vs_with_settings_parity_state(spliced_svar_ds_path, ref_fasta):
    """Internal state probe: both paths produce the same filter / spliced state."""
    ds_a = gvl.Dataset.open(
        spliced_svar_ds_path,
        reference=ref_fasta,
        splice_info=("transcript_id", "exon_number"),
        var_filter="exonic",
    ).with_seqs("haplotypes")

    ds_b = (
        gvl.Dataset.open(spliced_svar_ds_path, reference=ref_fasta)
        .with_seqs("haplotypes")
        .with_settings(
            splice_info=("transcript_id", "exon_number"),
            var_filter="exonic",
        )
    )

    assert ds_a._seqs.filter == ds_b._seqs.filter == "exonic"
    assert ds_a._recon.filter == ds_b._recon.filter == "exonic"
    assert ds_a._sp_idxer is not None and ds_b._sp_idxer is not None


def test_open_vs_with_settings_parity_output(spliced_svar_ds_path, ref_fasta):
    """Materialized output: __getitem__ must agree byte-for-byte."""
    ds_a = gvl.Dataset.open(
        spliced_svar_ds_path,
        reference=ref_fasta,
        splice_info=("transcript_id", "exon_number"),
        var_filter="exonic",
    ).with_seqs("haplotypes")

    ds_b = (
        gvl.Dataset.open(spliced_svar_ds_path, reference=ref_fasta)
        .with_seqs("haplotypes")
        .with_settings(
            splice_info=("transcript_id", "exon_number"),
            var_filter="exonic",
        )
    )

    haps_a = ds_a[0, :].to_ak()
    haps_b = ds_b[0, :].to_ak()

    assert ak.all(haps_a == haps_b), (
        "Path A (open(splice_info, var_filter)) and Path B "
        "(with_seqs.with_settings) produced different output."
    )
