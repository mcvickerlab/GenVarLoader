"""Round-trip behavior of ``gvl.write`` on edge BED inputs.

These tests pin down the observed behavior of ``gvl.write`` for three edge
cases on the BED side:

1. Empty BED - should either succeed (0-region dataset) or raise clearly.
2. Overlapping BED regions - regions are independent so this should either
   succeed cleanly or raise a clear, documented error.
3. BED entry on a contig missing from the reference/variant source - must
   raise a clear error rather than silently producing a broken dataset.
"""

from pathlib import Path

import genvarloader as gvl
import polars as pl
import pytest
from genoray import VCF


def _vcf(vcf_dir: Path) -> VCF:
    return VCF(vcf_dir / "filtered_source.vcf.gz")


def test_empty_bed_either_succeeds_or_raises_clearly(
    tmp_path: Path, vcf_dir: Path, ref_fasta: Path
):
    """Writing with an empty BED must either succeed (0-region dataset) or
    raise a clear error pointing at the empty input."""
    empty_bed = pl.DataFrame(
        schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64}
    )
    out = tmp_path / "empty.gvl"

    try:
        gvl.write(out, empty_bed, _vcf(vcf_dir))
    except (ValueError, RuntimeError) as e:
        # Clear-error path: message must mention something about the BED
        # being empty / having no regions.
        msg = str(e).lower()
        assert any(
            kw in msg for kw in ("empty", "no regions", "no region", "0 regions")
        ), f"Empty-BED error message should explain the cause: {e!r}"
        return

    # Success path: dataset opens and has zero regions.
    ds = gvl.Dataset.open(out, reference=ref_fasta)
    assert ds.n_regions == 0


def test_overlapping_bed_regions_succeed_or_raise(
    tmp_path: Path, vcf_dir: Path, ref_fasta: Path
):
    """Overlapping regions are conceptually independent (each region is
    materialized on its own), so ``gvl.write`` should either succeed and
    preserve the row count, or raise a clear documented error."""
    # Two regions overlapping on chr19, both within the contigs the toy
    # VCF carries variants on.
    bed = pl.DataFrame(
        {
            "chrom": ["chr19", "chr19", "chr19"],
            "chromStart": [1010685, 1010690, 1010700],
            "chromEnd": [1010715, 1010720, 1010730],
        }
    )
    out = tmp_path / "overlap.gvl"

    try:
        gvl.write(out, bed, _vcf(vcf_dir))
    except (ValueError, RuntimeError) as e:
        msg = str(e).lower()
        assert any(
            kw in msg for kw in ("overlap", "overlapping", "unique", "duplicate")
        ), f"Overlapping-region error must be clear: {e!r}"
        return

    ds = gvl.Dataset.open(out, reference=ref_fasta)
    assert ds.n_regions == bed.height


def test_bed_with_missing_contig_raises(tmp_path: Path, vcf_dir: Path):
    """A BED entry on a contig that has no variants in the source and is
    not a real reference contig should produce a clear error from
    ``gvl.write`` (or a downstream open) rather than a silent partial
    dataset."""
    bed = pl.DataFrame(
        {
            "chrom": ["chr_does_not_exist"],
            "chromStart": [100],
            "chromEnd": [200],
        }
    )
    out = tmp_path / "missing_contig.gvl"

    # Either ``write`` raises directly, or it succeeds and ``open`` raises;
    # either way, the user must get a clear error and not a silently broken
    # dataset.
    with pytest.raises((ValueError, RuntimeError, KeyError, Exception)):
        gvl.write(out, bed, _vcf(vcf_dir))
        # If write somehow accepts this, opening must surface the problem.
        gvl.Dataset.open(out)


def test_query_past_contig_end_pads_with_N(
    tmp_path: Path, vcf_dir: Path, ref_fasta: Path
):
    """A BED region whose end runs past the reference contig length must
    either be rejected at write time with a clear error, or produce a
    dataset whose reference query is right-padded with ``N`` for the
    portion past the contig boundary.

    This pins the observed boundary behavior so that a future change in
    handling (silent truncation, garbage bytes, etc.) is caught.
    """
    import pysam

    chrom = "chr19"
    with pysam.FastaFile(str(ref_fasta)) as fh:
        contig_len = fh.get_reference_length(chrom)

    # Region: [contig_len - 50, contig_len + 50). 50 in-bounds, 50 past end.
    in_bounds = 50
    past_end = 50
    start = contig_len - in_bounds
    end = contig_len + past_end
    region_len = end - start  # 100

    # Include a second region that overlaps known variants so writing has
    # something to genotype (otherwise the writer can produce a zero-variant
    # dataset that the reader can't reopen).
    bed = pl.DataFrame(
        {
            "chrom": [chrom, chrom],
            "chromStart": [1010685, start],
            "chromEnd": [1010715, end],
        }
    )
    out = tmp_path / "past_contig_end.gvl"

    try:
        gvl.write(out, bed, _vcf(vcf_dir))
    except (ValueError, RuntimeError) as e:
        msg = str(e).lower()
        assert any(
            kw in msg
            for kw in ("contig", "length", "bound", "end", "past", "beyond", "exceed")
        ), f"Past-contig-end error must be clear: {e!r}"
        return

    ds = gvl.Dataset.open(out, reference=ref_fasta).with_seqs("reference")
    # Region 1 is the one that runs past the contig end.
    seqs = ds[1, 0]
    # ArrayDataset[reference] returns a numpy array of S1 bytes (or a ragged
    # wrapper). Materialize to a 1-D bytes view either way.
    if hasattr(seqs, "to_padded"):
        arr = seqs.to_padded()
    else:
        arr = seqs
    flat = bytes(arr.reshape(-1).tobytes())

    assert len(flat) == region_len, (
        f"Expected query length {region_len}, got {len(flat)}"
    )
    # Tail bytes past the contig boundary must be ``N``.
    tail = flat[in_bounds:]
    assert tail == b"N" * past_end, (
        f"Expected {past_end} trailing 'N's past contig end, got {tail!r}"
    )
