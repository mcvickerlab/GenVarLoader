"""Regression tests for the ``_rc`` + ``_cat_length`` packing bug.

Before the fix, ``Dataset._rc`` built a ``Ragged`` from the raw ``ak.where(...)``
output. ``ak.where`` eagerly evaluates both branches and leaves an unpacked
layout whose content buffer holds ``[rc_branch, orig_branch]`` concatenated,
even though the virtual offsets only index into the selected branch. The
visible symptoms were:

1. ``rag.data`` exposed the full (2x) buffer rather than just the logical
   bytes, so direct buffer readers saw garbage followed by the real sequence.
2. ``_cat_length`` wrapped this unpacked layout in a new ``ListOffsetArray``
   and then called ``ak.flatten``/``ak.concatenate``, which walked the *wrong*
   half of the buffer for spliced datasets (e.g. positive-strand CDS came out
   reverse-complemented).

The fix adds ``ak.to_packed(...)`` around each ``ak.where`` result. These
tests pin that behavior by checking both the buffer size and the concrete
content against the reference.
"""

import shutil
from pathlib import Path

import awkward as ak
import genvarloader as gvl
import numpy as np
import polars as pl
import pysam
import pytest
from genoray import VCF
from pytest_cases import parametrize_with_cases

from genvarloader._dataset._impl import _cat_length, _cat_length_inner
from genvarloader._ragged import Ragged, reverse_complement


data_dir = Path(__file__).resolve().parents[1] / "data"
ref_path = data_dir / "fasta" / "hg38.fa.bgz"
source_bed = data_dir / "source.bed"
source_vcf = data_dir / "vcf" / "filtered_source.vcf.gz"


def _buffer_matches_lengths(rag: Ragged) -> bool:
    """Packed invariant: raw content equals the sum of the logical lengths."""
    return len(rag.data) == int(rag.lengths.sum())


# ---------------------------------------------------------------------------
# Unit: _rc produces a packed layout regardless of the to_rc mask
# ---------------------------------------------------------------------------


def case_all_false():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, False, False])


def case_all_true():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([True, True, True])


def case_mixed():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, True, False])


@parametrize_with_cases("rag, to_rc", cases=".", prefix="case_")
def test_rc_returns_packed_buffer(rag: Ragged, to_rc: np.ndarray):
    # mimic Dataset._rc exactly
    packed = Ragged(
        ak.to_packed(ak.where(to_rc, reverse_complement(rag.to_ak()), rag.to_ak()))
    )
    assert _buffer_matches_lengths(packed), (
        f"buffer doubled (len={len(packed.data)}, expected={int(packed.lengths.sum())})"
    )
    # and the content is correct for each row
    original = rag.to_ak().to_list()
    rc = reverse_complement(rag.to_ak()).to_list()
    got = packed.to_ak().to_list()
    for i, flip in enumerate(to_rc):
        expected = rc[i] if flip else original[i]
        assert got[i] == expected


# ---------------------------------------------------------------------------
# Unit: _cat_length over an unpacked source silently corrupted content. After
# the fix, feeding it a *packed* source (which is what _rc now guarantees)
# produces correct spliced output.
# ---------------------------------------------------------------------------


def test_cat_length_with_packed_input_preserves_content():
    # 4 "exons" × 2 ploidy × 3 bytes each, grouped 2+2.
    data = np.frombuffer(b"ATG" * 8, dtype="S1")
    lengths = np.array([[3, 3], [3, 3], [3, 3], [3, 3]])
    rag = Ragged.from_lengths(data, lengths)

    # Route through _rc with all-False to exercise the code path that used to
    # leak the rc branch into the buffer.
    to_rc = np.array([False, False, False, False])
    after_rc = Ragged(
        ak.to_packed(ak.where(to_rc, reverse_complement(rag.to_ak()), rag.to_ak()))
    )
    assert _buffer_matches_lengths(after_rc)

    offsets = np.array([0, 2, 4], dtype=np.int64)
    cat = _cat_length(after_rc, offsets)
    # Each splice concatenates 2 exons of 3 bytes on each of 2 ploidies
    assert cat.shape[0] == 2
    for splice_idx in range(2):
        for ploid in range(2):
            seq = bytes(ak.to_numpy(cat.to_ak()[splice_idx, ploid]))
            assert seq == b"ATGATG", (
                f"splice {splice_idx} ploid {ploid} content corrupted: {seq!r}"
            )


# ---------------------------------------------------------------------------
# Integration fixtures: write a fresh dataset into tmp so we don't mutate the
# shared test fixtures, and can attach custom transcript_id / exon_number
# columns to exercise the splice path.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spliced_ds_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a VCF-backed GVL store with single-exon transcript annotations.

    Each BED row becomes its own one-exon transcript so the spliced output
    should equal the reference slice (for + strand) or its reverse complement
    (for - strand). This exercises _getitem_spliced -> _rc -> _cat_length.
    """
    tmp = tmp_path_factory.mktemp("rc_packing")
    out = tmp / "single_exon.gvl"
    reader = VCF(source_vcf)
    reader._write_gvi_index()
    reader._load_index()
    gvl.write(path=out, bed=source_bed, variants=reader)

    # Re-write input_regions.arrow with transcript_id / exon_number columns so
    # _parse_splice_info can discover them.
    regions_path = out / "input_regions.arrow"
    bed = pl.read_ipc(regions_path)
    bed = bed.with_columns(
        transcript_id=pl.arange(0, pl.len()).cast(pl.Utf8),
        exon_number=pl.lit(1, pl.Int32),
    )
    # sink_ipc can hit memmap issues on macOS; write to a tmp file then move.
    tmp_arrow = regions_path.with_suffix(".arrow.tmp")
    bed.write_ipc(tmp_arrow)
    shutil.move(tmp_arrow, regions_path)
    return out


@pytest.fixture(scope="module")
def spliced_ds(spliced_ds_path: Path) -> gvl.Dataset:
    return (
        gvl.Dataset.open(spliced_ds_path, ref_path)
        .with_seqs("reference")
        .with_settings(splice_info=("transcript_id", "exon_number"))
    )


# ---------------------------------------------------------------------------
# Integration: unspliced single-index access returns a packed buffer
# ---------------------------------------------------------------------------


def test_unspliced_single_item_buffer_packed(spliced_ds_path: Path):
    """dss[region, sample] must not expose a doubled buffer from ak.where.

    Uses the default ragged output. The invariant we pin: the data buffer
    length equals the sum of the per-ploidy lengths. Before the fix
    ``ak.where`` left a 2x buffer behind and this check would fail.
    """
    ds = gvl.Dataset.open(spliced_ds_path, ref_path).with_seqs("haplotypes")
    for region in range(min(ds.n_regions, 5)):
        # ragged of shape (ploidy, var)
        haps = ds[region, 0]
        assert len(haps.data) == int(haps.lengths.sum()), (
            f"region {region}: data buffer len={len(haps.data)}, expected "
            f"{int(haps.lengths.sum())}. Likely unpacked ak.where buffer leaked through."
        )
        # Per-ploidy length should match the stored region length (no jitter).
        sorted_idx = ds._idxer.full_region_idxs[region]
        start, end = ds._full_regions[sorted_idx, 1:3]
        expected_len = int(end - start)
        assert (haps.lengths == expected_len).all(), (
            f"region {region}: lengths {haps.lengths} != expected {expected_len}"
        )


# ---------------------------------------------------------------------------
# Integration: spliced output matches (rc of) the reference
# ---------------------------------------------------------------------------


def test_spliced_reference_pos_strand_matches_fasta(spliced_ds: gvl.Dataset):
    """A single-exon positive-strand splice must equal the reference slice."""
    pos_rows = spliced_ds.spliced_regions.with_row_index("sp_idx").filter(
        pl.col("strand").list.eval(pl.element() == "+").list.all()
    )
    if pos_rows.height == 0:
        pytest.skip("no positive-strand regions in fixture")

    with pysam.FastaFile(str(ref_path)) as fa:
        for row in pos_rows.head(3).iter_rows(named=True):
            sp_idx = row["sp_idx"]
            chrom = row["chrom"][0]
            start = row["chromStart"][0]
            end = row["chromEnd"][0]
            expected = fa.fetch(chrom, start, end).upper().encode()
            got = spliced_ds[sp_idx, 0].tobytes()
            assert got == expected, (
                f"pos-strand splice {sp_idx} ({chrom}:{start}-{end}): "
                f"got {got[:15]}... expected {expected[:15]}..."
            )


def test_spliced_reference_neg_strand_is_rc_of_fasta(spliced_ds: gvl.Dataset):
    """A single-exon negative-strand splice must equal the RC of the reference slice."""
    neg_rows = spliced_ds.spliced_regions.with_row_index("sp_idx").filter(
        pl.col("strand").list.eval(pl.element() == "-").list.all()
    )
    if neg_rows.height == 0:
        pytest.skip("no negative-strand regions in fixture")

    comp = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")
    with pysam.FastaFile(str(ref_path)) as fa:
        for row in neg_rows.head(3).iter_rows(named=True):
            sp_idx = row["sp_idx"]
            chrom = row["chrom"][0]
            start = row["chromStart"][0]
            end = row["chromEnd"][0]
            ref_seq = fa.fetch(chrom, start, end).upper().encode()
            expected = ref_seq.translate(comp)[::-1]  # reverse complement
            got = spliced_ds[sp_idx, 0].tobytes()
            assert got == expected, (
                f"neg-strand splice {sp_idx} ({chrom}:{start}-{end}): "
                f"got {got[:15]}... expected {expected[:15]}..."
            )


# ---------------------------------------------------------------------------
# _cat_length regression: ploidy interleaving used to scramble ploid 1's bytes
# because the fast (ndim==2) path walked a (batch, ploidy)-interleaved data
# buffer sequentially. Lock that down.
# ---------------------------------------------------------------------------


def test_cat_length_preserves_per_ploidy_content():
    """Shape (n_batch, n_ploidy, var) with distinct per-ploidy content.

    Before the fix, ploid 1 of the concatenated output leaked ploid 0 bytes
    from the next batch slot.
    """
    # 4 exons × 2 ploidy, each exon's ploid 0 is "A...", ploid 1 is "B..."
    data = np.frombuffer(b"AAAAABBBBBCCCDDDEEEEFFFFGGHH", dtype="S1")
    lens = np.array([[5, 5], [3, 3], [4, 4], [2, 2]])
    rag = Ragged.from_lengths(data, lens)
    assert rag.ndim == 2  # Ragged's ndim excludes the ragged axis
    # Group exons 0-1 into splice 0 and exons 2-3 into splice 1
    offsets = np.array([0, 2, 4], dtype=np.int64)
    cat = _cat_length(rag, offsets)

    assert cat.shape == (2, 2, None), f"unexpected shape {cat.shape}"
    out = cat.to_ak().to_list()
    assert out == [
        [b"AAAAACCC", b"BBBBBDDD"],
        [b"EEEEGG", b"FFFFHH"],
    ], f"ploidy interleaving corrupted content: {out}"


def test_cat_length_non_bytes_dtype():
    """Non-bytestring dtypes (e.g. int32 annotations) must also concatenate per-ploidy."""
    # integers per (exon, ploidy)
    data = np.arange(18, dtype=np.int32)
    lens = np.array(
        [[2, 2], [3, 3], [2, 2]]
    )  # 3 exons × 2 ploid, total 14 slots... wait
    # 2+2+3+3+2+2 = 14 but data has 18. Let me recompute lengths summing to 18.
    # exon0 p0 len=2, p1 len=3 → 5 bytes; exon1 p0=3, p1=2 → 5 bytes; exon2 p0=4, p1=4 → 8 bytes. Total 18.
    lens = np.array([[2, 3], [3, 2], [4, 4]])
    rag = Ragged.from_lengths(data, lens)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    cat = _cat_length(rag, offsets)
    out = cat.to_ak().to_list()
    # splice 0 = exons 0-1; p0 = [0,1,5,6,7]; p1 = [2,3,4,8,9]
    # splice 1 = exon 2; p0 = [10,11,12,13]; p1 = [14,15,16,17]
    assert out == [
        [[0, 1, 5, 6, 7], [2, 3, 4, 8, 9]],
        [[10, 11, 12, 13], [14, 15, 16, 17]],
    ], f"non-bytes content corrupted: {out}"


# ---------------------------------------------------------------------------
# End-to-end: make sure BOTH ploidies come back correct for a multi-exon
# spliced query. This is the test that exposed the ploid-1 corruption.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_exon_ds_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a GVL store where each transcript has multiple exons.

    We reuse the BED rows grouped 2-at-a-time as a fake "transcript_id" so the
    splice logic has to concatenate multiple per-region outputs per ploidy.
    """
    tmp = tmp_path_factory.mktemp("rc_packing_multi")
    out = tmp / "multi_exon.gvl"
    reader = VCF(source_vcf)
    reader._write_gvi_index()
    reader._load_index()
    gvl.write(path=out, bed=source_bed, variants=reader)

    regions_path = out / "input_regions.arrow"
    bed = pl.read_ipc(regions_path)
    # Assign every 2 consecutive rows the same transcript_id, with exon_number
    # 1/2 inside each group.
    bed = bed.with_columns(
        transcript_id=(pl.arange(0, pl.len()) // 2).cast(pl.Utf8),
        exon_number=(pl.arange(0, pl.len()) % 2 + 1).cast(pl.Int32),
    )
    tmp_arrow = regions_path.with_suffix(".arrow.tmp")
    bed.write_ipc(tmp_arrow)
    shutil.move(tmp_arrow, regions_path)
    return out


def test_multi_exon_spliced_buffer_packed(multi_exon_ds_path: Path):
    """Spliced haplotype output: data buffer must equal sum of lengths for both ploidies.

    The _cat_length ploidy-interleaving bug produced an oversized/mispopulated
    data buffer. This invariant catches both the packing leak and the
    interleaving bug in one check.
    """
    ds = (
        gvl.Dataset.open(multi_exon_ds_path, ref_path)
        .with_seqs("haplotypes")
        .with_settings(splice_info=("transcript_id", "exon_number"))
    )
    for sp_idx in range(min(ds.shape[0], 5)):
        haps = ds[sp_idx, :]  # (n_samples, 2, var)
        assert len(haps.data) == int(haps.lengths.sum()), (
            f"splice {sp_idx}: data buffer len={len(haps.data)}, "
            f"expected {int(haps.lengths.sum())}. "
            f"Likely unpacked ak.where buffer or ploidy interleaving bug."
        )


def test_multi_exon_spliced_matches_fasta_concat(multi_exon_ds_path: Path):
    """Reference-mode spliced output must equal the concat of per-exon FASTA slices."""
    ds = (
        gvl.Dataset.open(multi_exon_ds_path, ref_path)
        .with_seqs("reference")
        .with_settings(splice_info=("transcript_id", "exon_number"))
    )
    comp = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")
    with pysam.FastaFile(str(ref_path)) as fa:
        for row in (
            ds.spliced_regions.with_row_index("sp_idx").head(5).iter_rows(named=True)
        ):
            sp_idx = row["sp_idx"]
            strands = set(row["strand"])
            if len(strands) > 1:
                continue  # mixed-strand synthetic rows are undefined
            strand = strands.pop()
            parts = []
            for chrom, start, end in zip(
                row["chrom"], row["chromStart"], row["chromEnd"]
            ):
                part = fa.fetch(chrom, start, end).upper().encode()
                if strand == "-":
                    part = part.translate(comp)[::-1]
                parts.append(part)
            expected = b"".join(parts)
            got = ds[sp_idx, 0].tobytes()
            assert got == expected, (
                f"splice {sp_idx} ({strand}): got {got[:20]}..., expected {expected[:20]}..."
            )


# ---------------------------------------------------------------------------
# Start / stop codon sanity on a real CDS dataset. Guarded behind an env var
# so the CI default doesn't need the 1kGP/Ensembl data. Run with
# ``GVL_CDS_DATASET=<path> GVL_CDS_REF=<bgz> pytest ...`` to enable.
# ---------------------------------------------------------------------------


import os

CDS_DS = os.environ.get("GVL_CDS_DATASET")
CDS_REF = os.environ.get("GVL_CDS_REF")


@pytest.mark.skipif(
    not (CDS_DS and CDS_REF and Path(CDS_DS).exists() and Path(CDS_REF).exists()),
    reason="Set GVL_CDS_DATASET and GVL_CDS_REF to enable real-CDS codon tests",
)
def test_cds_start_codon_is_atg_nearly_always():
    """Most CDS transcripts start with ATG.

    A small fraction of Ensembl transcripts use alternative start codons
    (GTG/CTG/TTG) or have annotation quirks, so we require ≥95% rather than
    100%. The pre-fix bug dropped this to ~50% on positive strand.
    """
    import genvarloader as gvl

    dss = (
        gvl.Dataset.open(CDS_DS, CDS_REF)
        .with_seqs("haplotypes")
        .with_settings(splice_info=("transcript_id", "exon_number"))
    )
    test = dss[:, :]
    starts = ak.str.slice(test, 0, 3)
    rate = float(ak.mean(starts == b"ATG"))
    assert rate >= 0.95, f"ATG-at-position-0 rate is {rate:.3f}, expected ≥ 0.95"


@pytest.mark.skipif(
    not (CDS_DS and CDS_REF and Path(CDS_DS).exists() and Path(CDS_REF).exists()),
    reason="Set GVL_CDS_DATASET and GVL_CDS_REF to enable real-CDS codon tests",
)
def test_cds_internal_stops_bounded():
    """A well-reconstructed CDS haplotype should translate to at most one stop.

    Ensembl CDS features EXCLUDE the terminal stop codon, so a reference
    haplotype should translate cleanly with zero internal stops. Variants can
    introduce a premature stop (nonsense mutations, frameshifts), but a
    correctly reconstructed transcript has no more than one. Any haplotype
    reporting ≥2 stops almost always indicates a reconstruction bug rather
    than real biology, so we cap it at 1.
    """
    import seqpro as sp
    import genvarloader as gvl

    dss = (
        gvl.Dataset.open(CDS_DS, CDS_REF)
        .with_seqs("haplotypes")
        .with_settings(splice_info=("transcript_id", "exon_number"))
    )
    max_stops = 0
    n_checked = 0
    for r in range(min(dss.n_regions, 50)):
        haps = dss[r, :]
        L = int(haps.lengths[0, 0])
        if not (L == haps.lengths).all():
            continue  # skip haplotypes with indels (different lengths)
        # sp.AA.translate wants shape ([..., L]) with dtype S1
        arr = haps.to_ak()
        for s in range(haps.shape[0]):
            for p in range(2):
                seq = np.frombuffer(bytes(ak.to_numpy(arr[s, p])), dtype="S1")
                prot = sp.AA.translate(seq.reshape(1, -1), -1).tobytes()
                stops = prot.count(b"*")
                max_stops = max(max_stops, stops)
                n_checked += 1
    assert n_checked > 0, "no indel-free haplotypes were analyzed"
    assert max_stops <= 1, (
        f"Found a haplotype with {max_stops} stop codons; expected ≤ 1 "
        f"(CDS excludes the terminal stop in Ensembl GTF, and in-frame variants "
        f"introduce at most one premature stop). Likely a reconstruction bug."
    )
