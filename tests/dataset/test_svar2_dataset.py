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

    from genvarloader._dataset._svar2_store_py import build_readbound_variants

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


def test_svar2_splice_and_rc_guards_raise(tmp_path, bed, svar2_fixture, _src):
    """splice_plan and a real (all-True) in-kernel to_rc must raise for haplotypes."""
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

    with pytest.raises(NotImplementedError, match="[Ss]plice"):
        recon(idx, r_idx, regions, "ragged", 0, rng, True, splice_plan=object())

    with pytest.raises(NotImplementedError, match="reverse-complement"):
        recon(
            idx,
            r_idx,
            regions,
            "ragged",
            0,
            rng,
            True,
            to_rc=np.ones(len(idx), np.bool_),
        )


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
