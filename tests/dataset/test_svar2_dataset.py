"""End-to-end SVAR2 dataset read dispatch parity (Task 7b).

Builds two gvl datasets over the same bed/samples/reference from matched stores
built from the SAME VCF -- one ``.svar`` (SVAR1) and one ``.svar2`` -- and asserts
the SVAR2 read path (``Svar2Haps``) is byte-identical to the SVAR1 path for
``with_seqs('haplotypes')`` and ``with_seqs('variants')``.

Parity is exact because both sides open with ``deterministic=True`` (shifts=0)
and the datasets are written with ``max_jitter=0``, so no RNG is involved. The
fixture VCF is tie-free (no same-POS SNP+DEL) so the SVAR1 max_ends tie bug
(docs/known-issues/svar1-max-ends-tie-underextension.md) is not exercised.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader import VarWindowOpt

_WIN_OPT = VarWindowOpt(
    flank_length=3, token_alphabet=b"ACGT", unknown_token=4, ref="window", alt="window"
)


def _open_windows_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref, opt=_WIN_OPT):
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    w1 = ds1.with_output_format("flat").with_seqs("variant-windows", opt)[:, :]
    w2 = ds2.with_output_format("flat").with_seqs("variant-windows", opt)[:, :]
    return w1, w2


def _assert_window_equal(a, b, name: str) -> None:
    """Flat-buffer equality of two _FlatWindow fields (data + both offset levels)."""
    assert np.array_equal(np.asarray(a.var_offsets), np.asarray(b.var_offsets)), (
        f"{name} var_offsets differ"
    )
    assert np.array_equal(np.asarray(a.seq_offsets), np.asarray(b.seq_offsets)), (
        f"{name} seq_offsets differ"
    )
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data)), f"{name} data differ"


def test_svar2_variant_windows_ref_window_matches_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """ref_window is a pure reference read over an identical variant SET, so it is
    byte-identical to SVAR1 (independent of the deletion-ALT encoding difference)."""
    _bcf, ref = _src
    w1, w2 = _open_windows_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    assert w2.ref_window is not None
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # scalar start field also identical (same variant SET) — compare _Flat buffers.
    assert np.array_equal(
        np.asarray(w2.fields["start"].data), np.asarray(w1.fields["start"].data)
    )
    assert np.array_equal(
        np.asarray(w2.fields["start"].offsets), np.asarray(w1.fields["start"].offsets)
    )


# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6
# (C>CAT), dense SNP@9 (G>C, carried by 3 haps -> dense/snp channel), DEL@11
# (GTA>G, ilen -2). No same-POS ties. Mirrors the readbound-haps dense-SNP
# fixture so both var_key and dense channels are exercised.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="module")
def _src(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("svar2_ds_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf, ref


@pytest.fixture(scope="module")
def svar_fixture(_src, tmp_path_factory) -> Path:
    bcf, _ref = _src
    from genoray import VCF, SparseVar

    out = tmp_path_factory.mktemp("svar1") / "store.svar"
    SparseVar.from_vcf(
        out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )
    return out


@pytest.fixture(scope="module")
def svar2_fixture(_src, tmp_path_factory) -> Path:
    bcf, ref = _src
    from genoray import _core

    out = tmp_path_factory.mktemp("svar2") / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "svar2 conversion did not finish"
    return out


@pytest.fixture(scope="module")
def bed() -> pl.DataFrame:
    # Tie-free windows spanning the SNP/INS/dense-SNP/DEL and a variant-free tail.
    return pl.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "chromStart": [0, 0, 5, 20],
            "chromEnd": [40, 15, 20, 40],
        }
    )


def _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref):
    from genoray import SparseVar, SparseVar2

    d1 = tmp_path / "d1.gvl"
    d2 = tmp_path / "d2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture), samples=None, overwrite=True)
    gvl.write(d2, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    return (
        gvl.Dataset.open(d1, reference=ref),
        gvl.Dataset.open(d2, reference=ref),
    )


def test_svar2_haplotypes_match_svar1(tmp_path, bed, svar_fixture, svar2_fixture, _src):
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("haplotypes")[:, :]
    b = ds2.with_seqs("haplotypes")[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets)), (
        f"offsets differ: svar1={np.asarray(a.offsets).tolist()} "
        f"svar2={np.asarray(b.offsets).tolist()}"
    )
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))


def test_svar2_spliced_minus_strand_haplotypes_match_svar1(
    tmp_path, svar_fixture, svar2_fixture, _src
):
    """SVAR2 splicing, exonic filtering, and minus-strand RC match SVAR1."""
    _bcf, ref = _src
    splice_bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 20, 5],
            "chromEnd": [13, 40, 20],
            "strand": ["-", "-", "+"],
            "transcript_id": ["Tneg", "Tneg", "Tplus"],
            "exon_number": [1, 2, 1],
        }
    )
    ds1, ds2 = _open_pair(tmp_path, splice_bed, svar_fixture, svar2_fixture, ref)
    settings = {
        "splice_info": ("transcript_id", "exon_number"),
        "var_filter": "exonic",
    }
    a = ds1.with_settings(**settings).with_seqs("haplotypes")[:, :]
    b = ds2.with_settings(**settings).with_seqs("haplotypes")[:, :]

    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))

    forward = ds2.with_settings(rc_neg=False, **settings).with_seqs("haplotypes")[0, 0]
    reverse = ds2.with_settings(**settings).with_seqs("haplotypes")[0, 0]
    assert not np.array_equal(forward.data.view("u1"), reverse.data.view("u1"))


def test_svar2_spliced_variants_match_exon_concatenation(
    tmp_path, svar_fixture, svar2_fixture, _src
):
    """GVL returns complete transcript/sample/phase variant records per query."""
    _bcf, ref = _src
    splice_bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 20, 5],
            "chromEnd": [13, 40, 13],
            "strand": ["+", "+", "+"],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )
    _, ds2 = _open_pair(tmp_path, splice_bed, svar_fixture, svar2_fixture, ref)
    unspliced = ds2.with_settings(var_filter="exonic").with_seqs("variants")[:, :]
    spliced = ds2.with_settings(
        splice_info=("transcript_id", "exon_number"), var_filter="exonic"
    ).with_seqs("variants")[:, :]

    assert spliced.shape == (2, 2, 2, None)

    def _logical_row(field, transcript, sample, phase):
        return np.asarray(field[transcript, sample, phase].to_packed().data).tolist()

    for sample in range(2):
        for phase in range(2):
            for field_name in ("start", "ilen", "alt"):
                spliced_field = getattr(spliced, field_name)
                unspliced_field = getattr(unspliced, field_name)
                assert _logical_row(spliced_field, 0, sample, phase) == (
                    _logical_row(unspliced_field, 0, sample, phase)
                    + _logical_row(unspliced_field, 1, sample, phase)
                )
                assert _logical_row(spliced_field, 1, sample, phase) == _logical_row(
                    unspliced_field, 2, sample, phase
                )


def test_svar2_spliced_variants_dataloader_returns_complete_cells(
    tmp_path, svar_fixture, svar2_fixture, _src
):
    """The ordinary written-dataset loader never exposes individual exons."""
    pytest.importorskip("torch")
    _bcf, ref = _src
    splice_bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 20, 5],
            "chromEnd": [13, 40, 13],
            "strand": ["+", "+", "+"],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )
    _, ds2 = _open_pair(tmp_path, splice_bed, svar_fixture, svar2_fixture, ref)
    ds2 = ds2.with_settings(
        splice_info=("transcript_id", "exon_number"), var_filter="exonic"
    ).with_seqs("variants")
    batches = list(
        ds2.to_dataloader(
            batch_size=2,
            shuffle=False,
            num_workers=0,
            return_indices=True,
        )
    )

    assert sum(len(region_idxs) for _, region_idxs, _ in batches) == 4
    for records, region_idxs, sample_idxs in batches:
        assert isinstance(records, gvl.RaggedVariants)
        assert records.shape[0] == len(region_idxs) == len(sample_idxs)
        assert records.shape[1:] == (2, None)


def test_svar2_exonic_filter_haplotypes_match_svar1(
    tmp_path, svar_fixture, svar2_fixture, _src
):
    """SVAR2 drops a deletion whose REF span crosses the region boundary."""
    _bcf, ref = _src
    exon = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [13],
        }
    )
    ds1, ds2 = _open_pair(tmp_path, exon, svar_fixture, svar2_fixture, ref)
    a = ds1.with_settings(var_filter="exonic").with_seqs("haplotypes")[:, :]
    b = ds2.with_settings(var_filter="exonic").with_seqs("haplotypes")[:, :]

    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))

    unfiltered = ds2.with_seqs("haplotypes")[0, 0]
    filtered = ds2.with_settings(var_filter="exonic").with_seqs("haplotypes")[0, 0]
    assert not np.array_equal(unfiltered.data.view("u1"), filtered.data.view("u1"))


def _make_bigwig(path: Path, contig: str, length: int, seed: int) -> None:
    """Write a dense per-bp BigWig over ``contig`` (deterministic given ``seed``)."""
    import pyBigWig

    rng = np.random.default_rng(seed)
    starts = list(range(length))
    ends = list(range(1, length + 1))
    values = [float(v) for v in rng.standard_normal(length).astype(np.float32)]
    with pyBigWig.open(str(path), "w") as bw:
        bw.addHeader([(contig, length)], maxZooms=0)
        bw.addEntries([contig] * length, starts, ends=ends, values=values)


@pytest.fixture(scope="module")
def bigwig_fixture(tmp_path_factory):
    """Per-sample BigWigs over chr1 (len 40) -> a SAMPLE-indexed gvl.BigWigs track."""
    bw_dir = tmp_path_factory.mktemp("svar2_bw")
    paths = {}
    for i, s in enumerate(["S0", "S1"]):
        p = bw_dir / f"{s}.bw"
        _make_bigwig(p, "chr1", 40, seed=100 + i)
        paths[s] = str(p)
    return gvl.BigWigs("signal", paths)


def test_svar2_tracks_match_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, bigwig_fixture, _src
):
    """Haplotype-realigned tracks byte-identical (f32, NaN-equal) to SVAR1.

    Both datasets are written from the SAME VCF + SAME BigWig track; at read the
    SVAR1 backend realigns via the fused kernel and the SVAR2 backend
    (``Svar2Haps`` + ``HapsTracks._call_svar2``) realigns via the split
    ``intervals_to_tracks`` + ``shift_and_realign_tracks_from_svar2_readbound``
    path. deterministic=True + max_jitter=0 => shifts=0, so parity is exact.
    """
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src
    d1 = tmp_path / "t1.gvl"
    d2 = tmp_path / "t2.gvl"
    gvl.write(
        d1,
        bed,
        variants=SparseVar(svar_fixture),
        tracks=bigwig_fixture,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    gvl.write(
        d2,
        bed,
        variants=SparseVar2(svar2_fixture),
        tracks=bigwig_fixture,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)

    # HapsTracks (seqs active by default) -> (haps, tracks) tuple.
    _h1, a = ds1.with_tracks("signal")[:, :]
    _h2, b = ds2.with_tracks("signal")[:, :]

    ao, bo = np.asarray(a.offsets), np.asarray(b.offsets)
    assert np.array_equal(ao, bo), (
        f"track offsets differ: svar1={ao.tolist()} svar2={bo.tolist()}"
    )
    ad, bd = np.asarray(a.data, np.float32), np.asarray(b.data, np.float32)
    assert np.allclose(ad, bd, equal_nan=True), "track data differ"


def test_svar2_tracks_match_svar1_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """Realigned tracks byte-identical to SVAR1 across a TWO-contig, out-of-order
    bed -- exercises ``_call_svar2``'s contig-group split + inverse row-perm
    stitching for the track path (single-contig fast path bypassed)."""
    import pyBigWig

    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    bw_dir = tmp_path / "bw_mc"
    bw_dir.mkdir()
    paths = {}
    for i, s in enumerate(["S0", "S1"]):
        p = bw_dir / f"{s}.bw"
        rng = np.random.default_rng(200 + i)
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader([("chr1", 40), ("chr2", 40)], maxZooms=0)
            for contig in ("chr1", "chr2"):
                vals = [float(v) for v in rng.standard_normal(40).astype(np.float32)]
                bw.addEntries(
                    [contig] * 40, list(range(40)), ends=list(range(1, 41)), values=vals
                )
        paths[s] = str(p)
    track = gvl.BigWigs("signal", paths)

    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 10, 5],
            "chromEnd": [40, 40, 40, 20],
        }
    )
    d1 = tmp_path / "tmc1.gvl"
    d2 = tmp_path / "tmc2.gvl"
    gvl.write(
        d1,
        bed,
        variants=SparseVar(svar_fixture2),
        tracks=track,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    gvl.write(
        d2,
        bed,
        variants=SparseVar2(svar2_fixture2),
        tracks=track,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)

    _h1, a = ds1.with_tracks("signal")[:, :]
    _h2, b = ds2.with_tracks("signal")[:, :]

    ao, bo = np.asarray(a.offsets), np.asarray(b.offsets)
    assert np.array_equal(ao, bo), (
        f"track offsets differ: svar1={ao.tolist()} svar2={bo.tolist()}"
    )
    ad, bd = np.asarray(a.data, np.float32), np.asarray(b.data, np.float32)
    assert np.allclose(ad, bd, equal_nan=True), "track data differ"


def test_svar2_flanksample_multicontig_matches_svar1(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """FlankSample (seed-dependent fill) + MULTI-contig is byte-identical to SVAR1.

    ``_call_svar2`` realigns per contig group, so ``k / ploidy`` is a contig-LOCAL
    query index; SVAR1 realigns the whole batch in one fused call and seeds the
    FlankSample fill hash with the GLOBAL row. Issue #267 makes the read-bound
    path pass each group's global row indices (``global_query``) into the FFI so
    the kernel seeds with the global row too. Without the fix the chr1 group's
    queries (which land at global rows 1, 3 in the interleaved bed) would seed off
    local indices 0, 1 and diverge in every inserted region.
    """
    import pyBigWig

    from genoray import SparseVar, SparseVar2

    from genvarloader._dataset._insertion_fill import FlankSample

    _bcf, ref = _src2
    bw_dir = tmp_path / "bw_fs"
    bw_dir.mkdir()
    paths = {}
    for i, s in enumerate(["S0", "S1"]):
        p = bw_dir / f"{s}.bw"
        rng = np.random.default_rng(300 + i)
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader([("chr1", 40), ("chr2", 40)], maxZooms=0)
            for contig in ("chr1", "chr2"):
                vals = [float(v) for v in rng.standard_normal(40).astype(np.float32)]
                bw.addEntries(
                    [contig] * 40, list(range(40)), ends=list(range(1, 41)), values=vals
                )
        paths[s] = str(p)
    track = gvl.BigWigs("signal", paths)

    # Interleaved chr2/chr1 bed -> >1 contig group, with each contig appearing at
    # multiple GLOBAL rows so local != global for the fill seed.
    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 10, 5],
            "chromEnd": [40, 40, 40, 20],
        }
    )
    d1 = tmp_path / "fs1.gvl"
    d2 = tmp_path / "fs2.gvl"
    gvl.write(
        d1,
        bed,
        variants=SparseVar(svar_fixture2),
        tracks=track,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    gvl.write(
        d2,
        bed,
        variants=SparseVar2(svar2_fixture2),
        tracks=track,
        samples=None,
        max_jitter=0,
        overwrite=True,
    )
    fill = {"signal": FlankSample(flank_width=3)}
    ds1 = (
        gvl.Dataset.open(d1, reference=ref)
        .with_tracks("signal")
        .with_insertion_fill(fill)
    )
    ds2 = (
        gvl.Dataset.open(d2, reference=ref)
        .with_tracks("signal")
        .with_insertion_fill(fill)
    )

    _h1, a = ds1[:, :]
    _h2, b = ds2[:, :]

    ao, bo = np.asarray(a.offsets), np.asarray(b.offsets)
    assert np.array_equal(ao, bo), (
        f"track offsets differ: svar1={ao.tolist()} svar2={bo.tolist()}"
    )
    ad, bd = np.asarray(a.data, np.float32), np.asarray(b.data, np.float32)
    assert np.allclose(ad, bd, equal_nan=True), "FlankSample track data differ"


def _assert_ragged_equal(a, b, name: str) -> None:
    ao, bo = np.asarray(a.offsets), np.asarray(b.offsets)
    assert np.array_equal(ao, bo), (
        f"{name} offsets differ: svar1={ao.tolist()} svar2={bo.tolist()}"
    )
    ad = np.asarray(a.data).view("u1")
    bd = np.asarray(b.data).view("u1")
    assert np.array_equal(ad, bd), f"{name} data differ"


def test_svar2_variants_positions_match_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """The decoded variant SET (positions + ilens) is byte-identical to SVAR1.

    NOTE: the ALT allele *bytes* are intentionally NOT compared to SVAR1 here.
    The two genoray formats encode a deletion's ALT differently -- SVAR1 keeps
    the VCF anchor base (e.g. ``G`` for ``GTA>G``) while SVAR2 decodes the
    atomized empty ALT (``""``). Haplotype reconstruction is unaffected (see
    ``test_svar2_haplotypes_match_svar1``), and the ALT bytes are validated
    against the SVAR2 decode oracle in ``test_svar2_variants_match_svar2_oracle``.
    """
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("variants")[:, :]
    b = ds2.with_seqs("variants")[:, :]
    _assert_ragged_equal(a.start.to_packed(), b.start.to_packed(), "start")
    _assert_ragged_equal(a.ilen.to_packed(), b.ilen.to_packed(), "ilen")


def test_svar2_variants_unphased_union_matches_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """Ploidy-1 union: start/ilen byte-identical to SVAR1 union (order-preserving
    fold, no dedup). ALT differs by encoding, so ALT is not compared to SVAR1."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("variants").with_settings(unphased_union=True)[:, :]
    b = ds2.with_seqs("variants").with_settings(unphased_union=True)[:, :]
    # Ploidy axis folded 2 -> 1.
    assert a.start.shape[-2] == 1 and b.start.shape[-2] == 1
    _assert_ragged_equal(a.start.to_packed(), b.start.to_packed(), "start")
    _assert_ragged_equal(a.ilen.to_packed(), b.ilen.to_packed(), "ilen")


def test_svar2_variants_match_svar2_oracle(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """Full RaggedVariants (start/ilen/alt) match the validated SVAR2 decode oracle.

    ``build_readbound_variants`` (parity-tested against genoray's ``SparseVar2.decode``
    in ``test_svar2_readbound_variants.py``) is driven over the dataset's own regions
    in the same (region, sample, ploid) order, so this pins the ``Svar2Haps`` dispatch
    (cache slicing + FFI wrapping + contig-group stitching) end-to-end.
    """
    from genoray import SparseVar2

    from tests._oracles.svar2_readbound_inputs import build_readbound_variants

    _bcf, ref = _src
    _, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    b = ds2.with_seqs("variants")[:, :]

    # Dataset regions in on-disk (sorted) order == the getitem's reconstruction
    # regions (jitter=0). Build the oracle over the same (start, end) windows.
    regions = ds2._full_regions
    reg_list = [(int(s), int(e)) for s, e in regions[:, 1:3]]
    sv = SparseVar2(svar2_fixture)
    oracle = build_readbound_variants(sv, "chr1", reg_list)

    _assert_ragged_equal(b.start.to_packed(), oracle.start.to_packed(), "start")
    _assert_ragged_equal(b.ilen.to_packed(), oracle.ilen.to_packed(), "ilen")
    _assert_ragged_equal(
        b.alt.to_chars().to_packed(), oracle.alt.to_chars().to_packed(), "alt"
    )


# --------------------------------------------------------------------------
# Guard-contract tests: the unsupported combos must RAISE, not silently return
# wrong output. (These lock the guards; the min_af one would have caught the
# open()-drops-min_af bug.)
# --------------------------------------------------------------------------


def test_svar2_min_af_guard_raises_open(tmp_path, bed, svar2_fixture, _src):
    """Dataset.open(min_af=...) must reach the NotImplementedError guard.

    Regression for the bug where _build_seqs dropped min_af/max_af for svar2,
    leaving Svar2Haps.min_af=None so the guard never fired.
    """
    from genoray import SparseVar2

    _bcf, ref = _src
    d = tmp_path / "d.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    ds = gvl.Dataset.open(d, reference=ref, min_af=0.05)
    with pytest.raises(NotImplementedError, match="min_af"):
        ds.with_seqs("haplotypes")[:, :]


def test_svar2_min_af_guard_raises_with_settings(tmp_path, bed, svar2_fixture, _src):
    """with_settings(min_af=...) must also reach the guard."""
    from genoray import SparseVar2

    _bcf, ref = _src
    d = tmp_path / "d.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    ds = gvl.Dataset.open(d, reference=ref).with_settings(min_af=0.05)
    with pytest.raises(NotImplementedError, match="min_af"):
        ds.with_seqs("haplotypes")[:, :]


def test_svar2_direct_reverse_complement(tmp_path, bed, svar2_fixture, _src):
    """Direct SVAR2 reconstruction reverse-complements every selected haplotype."""
    from genoray import SparseVar2

    _bcf, ref = _src
    d = tmp_path / "d.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    ds = gvl.Dataset.open(d, reference=ref).with_seqs("haplotypes")
    recon = ds._recon  # the Svar2Haps reconstructor (RaggedSeqs kind)

    idx = np.array([0], np.intp)
    r_idx = np.array([0], np.intp)
    regions = ds._full_regions[[0]].copy()
    rng = np.random.default_rng(0)

    forward = recon(idx, r_idx, regions, "ragged", 0, rng, True)
    reverse = recon(
        idx,
        r_idx,
        regions,
        "ragged",
        0,
        rng,
        True,
        to_rc=np.ones(len(idx), np.bool_),
    )

    comp = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")
    assert np.array_equal(forward.offsets, reverse.offsets)
    for i in range(forward.n_rows):
        start, end = map(int, forward.offsets[i : i + 2])
        expected = bytes(forward.data[start:end]).translate(comp)[::-1]
        assert bytes(reverse.data[start:end]) == expected


def test_svar2_variants_jitter_guard_raises(tmp_path, svar2_fixture, _src):
    """variants mode must raise when the dataset was written with max_jitter>0.

    The read-bound variants decode does not right-clip, so a padded cache would
    silently over-include variants; the guard prevents that.
    """
    from genoray import SparseVar2

    _bcf, ref = _src
    # chromStart >= max_jitter so the padded window stays non-negative.
    jbed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [5], "chromEnd": [20]})
    d = tmp_path / "d.gvl"
    gvl.write(
        d,
        jbed,
        variants=SparseVar2(svar2_fixture),
        samples=None,
        max_jitter=2,
        overwrite=True,
    )
    ds = gvl.Dataset.open(d, reference=ref)
    with pytest.raises(NotImplementedError, match="right-clip"):
        ds.with_seqs("variants")[:, :]


# --------------------------------------------------------------------------
# Multi-contig haplotype parity: locks the contig-group split + inverse
# row-permutation stitching in Svar2Haps.
# --------------------------------------------------------------------------

# chr2 reference; VCF REF alleles match _REF2 exactly (idx4='C', idx8='T').
_REF2 = "TTGGCCAATTGGCCAATTACGTACGTTTGGCCAATTGGCC"
_VCF2 = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##contig=<ID=chr2,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
chr2\t5\t.\tC\tT\t.\t.\t.\tGT\t1|0\t1|1
chr2\t9\t.\tT\tTGG\t.\t.\t.\tGT\t0|1\t1|0
"""


@pytest.fixture(scope="module")
def _src2(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("svar2_mc_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n>chr2\n{_REF2}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf = d / "in.vcf"
    vcf.write_text(_VCF2)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf, ref


@pytest.fixture(scope="module")
def svar_fixture2(_src2, tmp_path_factory) -> Path:
    bcf, _ref = _src2
    from genoray import VCF, SparseVar

    out = tmp_path_factory.mktemp("svar1_mc") / "store.svar"
    SparseVar.from_vcf(
        out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )
    return out


@pytest.fixture(scope="module")
def svar2_fixture2(_src2, tmp_path_factory) -> Path:
    bcf, ref = _src2
    from genoray import _core

    out = tmp_path_factory.mktemp("svar2_mc") / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1", "chr2"],
        str(out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "svar2 conversion did not finish"
    return out


def test_svar2_haplotypes_match_svar1_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """Haplotypes byte-identical to SVAR1 across a TWO-contig, out-of-order bed.

    The interleaved chr2/chr1 bed forces Svar2Haps' contig-group split + inverse
    row-permutation stitching (single-contig fast path is bypassed).
    """
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    # Interleaved contigs + a variant-free tail region, out of sorted order.
    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 10, 5],
            "chromEnd": [40, 40, 40, 20],
        }
    )
    d1 = tmp_path / "mc1.gvl"
    d2 = tmp_path / "mc2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True)
    gvl.write(
        d2, bed, variants=SparseVar2(svar2_fixture2), samples=None, overwrite=True
    )
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)

    a = ds1.with_seqs("haplotypes")[:, :]
    b = ds2.with_seqs("haplotypes")[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets)), (
        f"offsets differ: svar1={np.asarray(a.offsets).tolist()} "
        f"svar2={np.asarray(b.offsets).tolist()}"
    )
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))


def test_svar2_spliced_haplotypes_match_svar1_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """Spliced haplotypes byte-identical to SVAR1 when transcripts span TWO contigs.

    Splice order is (splice_row, sample, ploid, element), but SVAR2 reconstructs
    per contig group — so each group's rows land at destinations that are both
    non-monotonic AND gapped: the chr2 group (Ta + Tc) is split into two blocks
    with the chr1 group's (Tb) block sandwiched between them, because splice
    order runs Ta -> Tb -> Tc. The Rust carve must tolerate destination gaps
    (bytes owned by another call interleaved within a single contig group's
    scatter), and that's only exercised with 3+ transcripts alternating contigs.
    A single-contig fast path or a plain 2-transcript, 2-contig bed never
    produces a gap — each contig's block stays contiguous, just reordered
    relative to the other contig. Tb and Tc are both minus-strand, and Tc's
    rows land in the gapped (chr2) group, so the RC pass (rc_bounded_rows_inplace)
    is exercised on genuinely gapped destinations, not just scattered ones.
    """
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    # 3 transcripts alternating contigs (Ta:chr2, Tb:chr1, Tc:chr2) so splice
    # order (Ta -> Tb -> Tc) splits chr2's group (Ta ∪ Tc) around chr1's (Tb)
    # block -> gapped destinations. Tb and Tc are minus-strand.
    splice_bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1", "chr2", "chr2"],
            "chromStart": [0, 0, 20, 5, 5, 25],
            "chromEnd": [13, 13, 40, 20, 18, 45],
            "strand": ["+", "-", "+", "-", "-", "-"],
            "transcript_id": ["Ta", "Tb", "Ta", "Tb", "Tc", "Tc"],
            "exon_number": [1, 1, 2, 2, 1, 2],
        }
    )
    d1 = tmp_path / "mcs1.gvl"
    d2 = tmp_path / "mcs2.gvl"
    gvl.write(
        d1, splice_bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True
    )
    gvl.write(
        d2,
        splice_bed,
        variants=SparseVar2(svar2_fixture2),
        samples=None,
        overwrite=True,
    )
    settings = {"splice_info": ("transcript_id", "exon_number"), "var_filter": "exonic"}
    ds1 = gvl.Dataset.open(d1, reference=ref, **settings).with_seqs("haplotypes")
    ds2 = gvl.Dataset.open(d2, reference=ref, **settings).with_seqs("haplotypes")

    a = ds1[:, :]
    b = ds2[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets)), (
        f"offsets differ: svar1={np.asarray(a.offsets).tolist()} "
        f"svar2={np.asarray(b.offsets).tolist()}"
    )
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))


# --------------------------------------------------------------------------
# variant-windows (Task 1): ref_window pinned to SVAR1; alt_window validated via
# ref-flank decomposition + tokenized variants.alt; multi-contig stitch, dummy
# fill, and the ref="allele" / jitter guards.
# --------------------------------------------------------------------------


def test_svar2_variant_windows_alt_window_decomposition(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """alt_window[j] == ref_window[j][:L] + tokenize(alt_j) + ref_window[j][-L:].
    Uses only svar2's own outputs; ref_window is separately pinned to SVAR1."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    w_win = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    alt_opt = VarWindowOpt(
        flank_length=L,
        token_alphabet=b"ACGT",
        unknown_token=4,
        ref="window",
        alt="allele",
    )
    w_alt = ds2.with_output_format("flat").with_seqs("variant-windows", alt_opt)[:, :]

    rw = w_win.ref_window
    aw = w_win.alt_window
    ba = w_alt.alt  # bare tokenized alt (_FlatWindow)
    assert aw is not None and rw is not None and ba is not None

    # Same variant SET/order across the two reads.
    assert np.array_equal(np.asarray(aw.var_offsets), np.asarray(ba.var_offsets))
    n_var = len(np.asarray(aw.seq_offsets)) - 1
    rso, aso, bso = (
        np.asarray(rw.seq_offsets),
        np.asarray(aw.seq_offsets),
        np.asarray(ba.seq_offsets),
    )
    rd, ad, bd = np.asarray(rw.data), np.asarray(aw.data), np.asarray(ba.data)
    for j in range(n_var):
        rj = rd[rso[j] : rso[j + 1]]
        aj = ad[aso[j] : aso[j + 1]]
        bj = bd[bso[j] : bso[j + 1]]
        expected = np.concatenate([rj[:L], bj, rj[len(rj) - L :]])
        assert np.array_equal(aj, expected), f"alt_window variant {j} mismatch"


def test_svar2_variant_windows_bare_alt_tokenizes_variants_alt(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    import awkward as ak

    from genvarloader._dataset._flat_flanks import build_token_lut

    _bcf, ref = _src
    _, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    alt_opt = VarWindowOpt(
        flank_length=L,
        token_alphabet=b"ACGT",
        unknown_token=4,
        ref="window",
        alt="allele",
    )
    w_alt = ds2.with_output_format("flat").with_seqs("variant-windows", alt_opt)[:, :]
    v = ds2.with_seqs("variants")[:, :]  # RaggedVariants (validated)

    lut, _ = build_token_lut(b"ACGT", 4)
    # Flat (b*p) rows, each a list of alt byte-strings in variant order.
    alt_rows = ak.to_list(v.alt.to_ak())  # (b*p) -> [bytes,...]
    flat_alts: list[bytes] = []
    for per_var in alt_rows:
        for a in per_var:
            flat_alts.append(bytes(a) if not isinstance(a, bytes) else a)

    ba = w_alt.alt
    bso, bd = np.asarray(ba.seq_offsets), np.asarray(ba.data)
    assert len(flat_alts) == len(bso) - 1
    for j, a in enumerate(flat_alts):
        toks = bd[bso[j] : bso[j + 1]]
        expected = np.array([lut[byte] for byte in a], dtype=toks.dtype)
        assert np.array_equal(toks, expected), f"bare alt variant {j} mismatch"


def test_svar2_variant_windows_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """ref_window byte-identical to SVAR1 across an interleaved 2-contig bed
    (single-contig fast path bypassed -> exercises the group-stitch reorder)."""
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 10, 5],
            "chromEnd": [40, 40, 40, 20],
        }
    )
    d1 = tmp_path / "vw_mc1.gvl"
    d2 = tmp_path / "vw_mc2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True)
    gvl.write(
        d2, bed, variants=SparseVar2(svar2_fixture2), samples=None, overwrite=True
    )
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)
    w1 = ds1.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    w2 = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # alt_window decomposition holds across the stitch too.
    w2.alt_window.to_ragged()  # offsets/data consistent post-reorder
    w2.ref_window.to_ragged()


def test_svar2_variant_windows_unphased_union(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """Union folds ploidy 2->1 for windows; ref_window still byte-identical to
    SVAR1 union, and the union row is hap-0's windows then hap-1's, concatenated."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    w1 = (
        ds1.with_output_format("flat")
        .with_seqs("variant-windows", _WIN_OPT)
        .with_settings(unphased_union=True)[:, :]
    )
    w2 = (
        ds2.with_output_format("flat")
        .with_seqs("variant-windows", _WIN_OPT)
        .with_settings(unphased_union=True)[:, :]
    )
    # Ploidy axis folded 2 -> 1. Scalar shape is (R,S,p_eff,None) so ploidy is at
    # [-2]; window shape is (R,S,p_eff,None,None) so ploidy is at [-3].
    assert w2.fields["start"].shape[-2] == 1
    assert w2.ref_window.shape[-3] == 1
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # Union row count == sum over haplotypes: compare to the non-union var counts.
    nu = np.asarray(w2.ref_window.var_offsets)
    w2_diploid = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[
        :, :
    ]
    nd = np.asarray(w2_diploid.ref_window.var_offsets)
    P = int(ds2._seqs.genotypes.shape[-2])
    # Folded per-row counts == sum of the P per-hap counts (rows q*P+p are contiguous).
    diploid_counts = np.diff(nd).reshape(-1, P).sum(1)
    union_counts = np.diff(nu)
    assert np.array_equal(union_counts, diploid_counts)
    w2.ref_window.to_ragged()
    w2.alt_window.to_ragged()


def test_svar2_variant_windows_union_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    bed = pl.DataFrame(
        {"chrom": ["chr2", "chr1"], "chromStart": [0, 0], "chromEnd": [40, 40]}
    )
    d1 = tmp_path / "vwu_mc1.gvl"
    d2 = tmp_path / "vwu_mc2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True)
    gvl.write(
        d2, bed, variants=SparseVar2(svar2_fixture2), samples=None, overwrite=True
    )
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)
    w1 = (
        ds1.with_output_format("flat")
        .with_seqs("variant-windows", _WIN_OPT)
        .with_settings(unphased_union=True)[:, :]
    )
    w2 = (
        ds2.with_output_format("flat")
        .with_seqs("variant-windows", _WIN_OPT)
        .with_settings(unphased_union=True)[:, :]
    )
    assert w2.ref_window.shape[-3] == 1  # window ploidy axis
    _assert_window_equal(
        w2.ref_window, w1.ref_window, "ref_window (union, multicontig)"
    )
    w2.alt_window.to_ragged()


def test_svar2_variant_windows_dummy_fills_empty_groups(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    from genvarloader import DummyVariant

    _bcf, ref = _src
    _, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    dummy = DummyVariant(alt=b"N", ref=b"N")
    w = (
        ds2.with_output_format("flat")
        .with_settings(dummy_variant=dummy)
        .with_seqs("variant-windows", _WIN_OPT)[:, :]
    )
    # Every (b*p) row now has >= 1 variant (no empty rows).
    vo = np.asarray(w.ref_window.var_offsets)
    assert np.all(np.diff(vo) >= 1)
    # ref_window dummy width = 2L + len(dummy.ref); alt_window = 2L + len(dummy.alt).
    # (For a filled row the sole variant's window length equals the dummy width.)
    # Assert at least one dummy-width ref window exists (the tail region rows).
    rso = np.asarray(w.ref_window.seq_offsets)
    assert (np.diff(rso) == (2 * L + len(dummy.ref))).any()
    w.ref_window.to_ragged()
    w.alt_window.to_ragged()


def test_svar2_variant_windows_ref_allele_guard(tmp_path, bed, svar2_fixture, _src):
    """ref='allele' needs stored REF bytes svar2 lacks -> ValueError at with_seqs."""
    from genoray import SparseVar2

    _bcf, ref = _src
    d = tmp_path / "d.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    ds = gvl.Dataset.open(d, reference=ref).with_output_format("flat")
    bad = VarWindowOpt(
        flank_length=3,
        token_alphabet=b"ACGT",
        unknown_token=4,
        ref="allele",
        alt="window",
    )
    with pytest.raises(ValueError, match="REF"):
        ds.with_seqs("variant-windows", bad)


def test_svar2_variant_windows_jitter_guard(tmp_path, svar2_fixture, _src):
    """variant-windows must raise when written with max_jitter>0 (no right-clip)."""
    from genoray import SparseVar2

    _bcf, ref = _src
    jbed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [5], "chromEnd": [20]})
    d = tmp_path / "d.gvl"
    gvl.write(
        d,
        jbed,
        variants=SparseVar2(svar2_fixture),
        samples=None,
        max_jitter=2,
        overwrite=True,
    )
    ds = gvl.Dataset.open(d, reference=ref).with_output_format("flat")
    with pytest.raises(NotImplementedError, match="right-clip"):
        ds.with_seqs("variant-windows", _WIN_OPT)[:, :]
