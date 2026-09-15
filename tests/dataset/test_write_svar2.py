"""Tests for the `.svar2` write path: `_write_from_svar2` + dispatch.

Builds a `.svar2` store (and a matched `.svar` store from the same VCF+FASTA)
using the same recipe as `tests/test_svar2_reconstruct.py`'s `svar2_store`
fixture, then exercises `gvl.write(..., variants=SparseVar2(...))`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._svar2_link import Svar2Link

# _REF, _VCF, and the vcf_and_ref/svar2_store/svar1_store fixtures they back
# live in tests/dataset/conftest.py: both this module and test_concat_svar2.py
# need the same .svar2 store.


def test_write_svar2_emits_cache(svar2_store: Path, tmp_path: Path):
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    bed = pl.DataFrame(
        {
            # [25, 40) holds no variants at all: an entirely empty region row,
            # which the sparse layout must round-trip as (0, 0) everywhere.
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 5, 25],
            "chromEnd": [20, 15, 40],
        }
    )
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=svar2, samples=None, overwrite=True)

    rd = out / "genotypes" / "svar2_ranges"
    meta = json.loads((rd / "svar2_meta.json").read_text())
    assert meta["layout"] == "sparse"
    assert set(meta) >= {
        "layout",
        "n_regions",
        "n_samples",
        "n_entries",
        "fill",
        "region_ptr",
        "cell_id",
        "cell_vk",
        "dense_snp_range",
        "dense_indel_range",
        "sample_cols",
    }
    assert "vk_snp_range" not in meta, "the dense layout must no longer be written"
    assert meta["ploidy"] == svar2.ploidy

    md = json.loads((out / "metadata.json").read_text())
    assert md["svar2_link"] is not None
    assert md["ploidy"] == svar2.ploidy
    Svar2Link.model_validate(md["svar2_link"])  # shape check

    # ---- The layout oracle. Replays _find_ranges over the same regions and the
    # sorted sample list gvl.write wrote, then compares through _ranges_reader.
    # This checks the (region, sample, ploid) VALUES the cache holds against an
    # independent oracle for this fixture's grid. It does NOT by itself prove a
    # mis-transposed axis order fails loudly here -- `_find_ranges`'s test
    # fixture happens not to distinguish some axis permutations, so that
    # property is enforced structurally instead, by `nonempty_entries`'s own
    # shape check (raises on any transpose that changes rank or the ploidy
    # axis) and pinned end-to-end by tests/dataset/test_svar2_fields_read.py.
    #
    # It compares WIDTHS and NON-EMPTY entries, not raw bytes: the sparse layout
    # deliberately discards an empty cell's insertion point, which is exactly the
    # semantic #357 buys. Asserting byte-equality would assert the bug back in.
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    sorted_samples = sorted(svar2.available_samples)
    S, P = len(sorted_samples), svar2.ploidy
    reader = _ranges_reader(rd)
    assert (reader.n_regions, reader.n_samples, reader.ploidy) == (bed.height, S, P)

    sample_cols = np.load(rd / "sample_cols.npy")
    assert sample_cols.tolist() == [
        svar2.available_samples.index(s) for s in sorted_samples
    ]

    def mm(name: str) -> np.ndarray:
        shape = tuple(meta[name]["shape"])
        return np.array(
            np.memmap(rd / f"{name}.npy", dtype=np.int64, mode="r", shape=shape)
        )

    dense_snp = mm("dense_snp_range")  # (R, 2), still dense: dense_abs_row
    dense_indel = mm("dense_indel_range")  # uses .start as an index base.

    contig_offset = 0
    n_empty_seen = 0
    for (c,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        rc = df.height
        lo, hi = contig_offset, contig_offset + rc
        d = svar2._find_ranges(
            c,
            df["chromStart"].to_numpy(),
            df["chromEnd"].to_numpy(),
            samples=sorted_samples,
        )
        exp_snp = np.asarray(d["vk_snp_range"], np.int64)  # (rc*S*P, 2)
        exp_indel = np.asarray(d["vk_indel_range"], np.int64)

        r_q, si_q = np.unravel_index(np.arange(rc * S), (rc, S))
        got_snp, got_indel = reader.lookup(r_q + lo, si_q, P)

        # Presence is a property of the CELL, not of one channel: a stored cell
        # whose SNP range happens to be empty still carries the real snp_start
        # with snp_len == 0, so lookup returns the true insertion point there,
        # NOT (0, 0) -- only a genuinely absent cell (both channels empty)
        # returns (0, 0). Masking per channel (`widths_snp > 0` alone) is wrong
        # and would incorrectly demand (0, 0) at a present cell's empty channel;
        # see _assert_parity in tests/unit/dataset/test_svar2_ranges.py, which
        # this mirrors.
        widths_snp = exp_snp[:, 1] - exp_snp[:, 0]
        widths_indel = exp_indel[:, 1] - exp_indel[:, 0]
        np.testing.assert_array_equal(got_snp[:, 1] - got_snp[:, 0], widths_snp)
        np.testing.assert_array_equal(got_indel[:, 1] - got_indel[:, 0], widths_indel)
        present = (widths_snp > 0) | (widths_indel > 0)
        for got, exp in ((got_snp, exp_snp), (got_indel, exp_indel)):
            np.testing.assert_array_equal(got[present], exp[present])
            np.testing.assert_array_equal(got[~present], 0)
        n_empty_seen += int((~present).sum())

        np.testing.assert_array_equal(
            dense_snp[lo:hi], np.asarray(d["dense_snp_range"], np.int64)
        )
        np.testing.assert_array_equal(
            dense_indel[lo:hi], np.asarray(d["dense_indel_range"], np.int64)
        )
        contig_offset += rc

    assert n_empty_seen > 0, "fixture regressed to 100% fill (see Task 1)"

    # Sparse must be smaller than the dense layout would have been, at this fill.
    n = meta["n_entries"]
    assert n * 28 < bed.height * S * P * 32
    assert meta["fill"] == pytest.approx(n / (bed.height * S * P))


def test_write_svar2_max_ends_matches_svar1(
    svar2_store: Path, svar1_store: Path, tmp_path: Path
):
    """SVAR1 parity gate: end-extension semantics must match exactly.

    Regions are chosen to overlap the DEL at (0-based) POS 11 with varying
    windows, so the extension is non-trivial and exercises the "no variants"
    (keep chromEnd) branch too.
    """
    from genoray import SparseVar, SparseVar2

    svar2 = SparseVar2(svar2_store)
    svar1 = SparseVar(svar1_store)

    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "chromStart": [0, 0, 5, 12, 20],
            "chromEnd": [15, 20, 10, 13, 30],
        }
    )

    out2 = tmp_path / "ds_svar2.gvl"
    gvl.write(out2, bed, variants=svar2, samples=None, overwrite=True)

    out1 = tmp_path / "ds_svar1.gvl"
    gvl.write(out1, bed, variants=svar1, samples=None, overwrite=True)

    regions2 = np.load(out2 / "regions.npy")
    regions1 = np.load(out1 / "regions.npy")

    # columns: chrom_idx, chromStart, chromEnd, strand
    chrom_end_2 = regions2[:, 2]
    chrom_end_1 = regions1[:, 2]

    assert chrom_end_2.tolist() == chrom_end_1.tolist(), (
        f"svar2 max_ends {chrom_end_2.tolist()} != svar1 max_ends {chrom_end_1.tolist()}"
    )


# Same-POS tie fixture (FIX 2): two records at POS 12 (0-based 11) with different
# ends -- a SNP (G>A, end=12) and a DEL (GTA>G, ILEN -2, end=14) -- placed on
# DIFFERENT haplotypes of S0 (SNP on hap0, DEL on hap1). A single haplotype
# cannot carry both an overlapping SNP and DEL, so putting them on the same hap
# would make the svar2 encoder drop one; different haps keeps both variants
# present and reachable. Ordering is the coordinator's exact example (SNP record
# first, DEL record second), so in store order the DEL gets the higher v_idx.
# SVAR1's max_ends picks the max-v_idx variant's end; svar2 picks the max-end
# variant on a POS tie -- here both rules select the DEL (end 14), so the paths
# agree. See the task-2 report for the reverse store order (DEL-first), where
# the two rules provably diverge.
_TIE_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t12\t.\tG\tA\t.\t.\t.\tGT\t1|0\t0|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t0|1\t0|0
"""


@pytest.fixture(scope="module")
def tie_stores(tmp_path_factory) -> tuple[Path, Path]:
    """Matched .svar2 and .svar stores from the same two-same-POS-records VCF."""
    from genoray import VCF, SparseVar, _core

    from tests.dataset.conftest import _REF

    d = tmp_path_factory.mktemp("svar2_tie")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_TIE_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar2_out = d / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(svar2_out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (svar2_out / "meta.json").exists(), "svar2 conversion did not finish"

    svar1_out = d / "store.svar"
    SparseVar.from_vcf(
        svar1_out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )
    return svar2_out, svar1_out


def test_write_svar2_max_ends_same_pos_tie(
    tie_stores: tuple[Path, Path], tmp_path: Path
):
    """SVAR1 parity on a same-POS tie: a SNP and a DEL at the same position.

    The bed region ends 1bp short of the DEL's footprint so the extension is
    variant-driven (not masked by the region's own chromEnd). Both paths must
    agree on the extended chromEnd.
    """
    from genoray import SparseVar, SparseVar2

    svar2_out, svar1_out = tie_stores
    svar2 = SparseVar2(svar2_out)
    svar1 = SparseVar(svar1_out)

    # POS 12 -> 0-based 11. Region [11, 13) overlaps it; region chromEnd 13 is
    # below the DEL end (14), so the max_ends extension is variant-driven.
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [11], "chromEnd": [13]})

    out2 = tmp_path / "tie_svar2.gvl"
    gvl.write(out2, bed, variants=svar2, samples=None, overwrite=True)
    out1 = tmp_path / "tie_svar1.gvl"
    gvl.write(out1, bed, variants=svar1, samples=None, overwrite=True)

    chrom_end_2 = np.load(out2 / "regions.npy")[:, 2]
    chrom_end_1 = np.load(out1 / "regions.npy")[:, 2]

    assert chrom_end_2.tolist() == chrom_end_1.tolist(), (
        f"same-POS tie: svar2 max_ends {chrom_end_2.tolist()} != "
        f"svar1 max_ends {chrom_end_1.tolist()}"
    )


def test_svar2_extend_to_length_false_raises(svar2_store: Path, tmp_path: Path):
    """extend_to_length=False is unsupported for a .svar2 source: it must raise
    NotImplementedError, not silently produce an extended dataset."""
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 5],
            "chromEnd": [20, 15],
        }
    )
    out = tmp_path / "ds.gvl"
    with pytest.raises(NotImplementedError, match="extend_to_length"):
        gvl.write(
            out,
            bed,
            variants=svar2,
            samples=None,
            extend_to_length=False,
            overwrite=True,
        )


def test_write_svar2_chunked_matches_unchunked(svar2_store: Path, tmp_path):
    """A tiny max_mem must force multiple chunks and produce identical output."""
    from genoray import SparseVar2

    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 5], "chromEnd": [20, 30]}
    )

    calls: list[int] = []
    real = SparseVar2._find_ranges_chunked

    def spy(self, *args, **kwargs):
        stream = real(self, *args, **kwargs)
        calls.append(stream.samples_per_chunk)
        return stream

    big = tmp_path / "big.gvl"
    gvl.write(
        big,
        bed,
        variants=SparseVar2(svar2_store),
        samples=None,
        max_mem="4g",
        overwrite=True,
    )

    SparseVar2._find_ranges_chunked = spy
    try:
        small = tmp_path / "small.gvl"
        # 2 regions x ploidy 2 x 2 channels x 2 endpoints x 8 bytes = 128 bytes
        # per sample; the chunker's own 2x safety margin needs 256 bytes for
        # even one sample, so 256 is the smallest budget that both succeeds
        # and forces one-sample-per-chunk (this store has S=2, so that's 2
        # chunks).
        gvl.write(
            small,
            bed,
            variants=SparseVar2(svar2_store),
            samples=None,
            max_mem=256,
            overwrite=True,
        )
    finally:
        SparseVar2._find_ranges_chunked = real

    assert calls and all(c == 1 for c in calls), (
        f"expected one sample per chunk under a 256-byte budget, got {calls}"
    )

    # Byte-identical output between a single-chunk write and a one-sample-per-
    # chunk write. This pins that `append_contig`'s running `cursor` ends up
    # correct across chunk boundaries -- it is NOT a targeted probe for any one
    # accumulator bug (e.g. deleting `cursor += cnt` also breaks the S=1
    # no-second-chunk "big" path the same way, so it fails here too, just not
    # for the reason the name might suggest); it is a black-box round-trip
    # check over the whole on-disk table.
    for name in (
        "region_ptr.npy",
        "cell_id.npy",
        "cell_vk.npy",
        "dense_snp_range.npy",
        "dense_indel_range.npy",
        "sample_cols.npy",
        "svar2_meta.json",
    ):
        a = (big / "genotypes" / "svar2_ranges" / name).read_bytes()
        b = (small / "genotypes" / "svar2_ranges" / name).read_bytes()
        assert a == b, name

    # regions.npy (not input_regions.arrow, which holds the pre-extension bed
    # verbatim) carries the write-time-extended chromEnd; columns are
    # chrom_idx, chromStart, chromEnd, strand.
    ra = np.load(big / "regions.npy")
    rb = np.load(small / "regions.npy")
    assert ra[:, 2].tolist() == rb[:, 2].tolist()


def test_write_svar2_max_ends_extend_chromend(svar2_store: Path, tmp_path):
    """chromEnd must extend past a deletion that starts inside the region.

    The fixture's DEL is at 0-based POS 11 with ilen -2, so it ends at 14. A
    region of [0, 12) must be extended to 14.
    """
    from genoray import SparseVar2

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [12]})
    out = tmp_path / "ext.gvl"
    gvl.write(
        out,
        bed,
        variants=SparseVar2(svar2_store),
        samples=None,
        max_mem="1g",
        overwrite=True,
    )
    # regions.npy carries the write-time-extended chromEnd (input_regions.arrow
    # holds the pre-extension bed verbatim); columns are chrom_idx, chromStart,
    # chromEnd, strand.
    regions = np.load(out / "regions.npy")
    assert regions[:, 2].tolist() == [14]


def test_svar2_ranges_cache_bytes():
    """The DENSE formula: 2 * R * S * P * 2 endpoints * 8 bytes.

    Retained after #357 as the "what the old layout would have cost" reference
    the preflight logs for context. It is no longer a projection.
    """
    from genvarloader._dataset._write import _svar2_ranges_cache_bytes

    assert _svar2_ranges_cache_bytes(1, 1, 2) == 2 * 1 * 1 * 2 * 2 * 8
    # The scale from gvl#333: ~98 GiB for one chromosome/panel.
    big = _svar2_ranges_cache_bytes(3964, 414830, 2)
    assert 90 * 1024**3 < big < 110 * 1024**3


def test_svar2_preflight_logs_dense_equivalent_without_warning(tmp_path, monkeypatch):
    """Preflight must NOT warn on the dense figure: the sparse cache is ~200x smaller.

    Warning on `28 * R * S * P` would fire on every cohort build for a cache that
    actually fits, which is exactly the false alarm #357 removes.
    """
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        n = _write._svar2_preflight(tmp_path, 3964, 414830, 2)
    finally:
        logger.remove(sink)

    assert n == _write._svar2_ranges_cache_bytes(3964, 414830, 2)
    assert not msgs, f"preflight must not warn on the dense-equivalent figure: {msgs}"


def test_svar2_fill_projection_warns_when_disk_is_short(tmp_path, monkeypatch):
    """After the first contig, the realized fill drives the free-space check."""
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        # 1 of 100 regions done, 1e6 entries => ~1e8 entries * 28 B projected.
        n = _write._svar2_fill_projection(tmp_path, 1_000_000, 1, 100, 500, 2)
    finally:
        logger.remove(sink)

    assert n == 28 * 100 * 1_000_000
    assert any("free" in m for m in msgs), msgs


def test_svar2_fill_projection_zero_entries_projects_nothing(tmp_path, monkeypatch):
    """A variant-free first contig must not project 0 bytes and call it a day.

    This is the failure mode the `projected` latch at the call site exists for: a
    leading contig that happens to hold no variant extrapolates to 0, which would
    pass any free-space check and, without the latch, suppress the projection for
    the whole build. The helper is honest about having nothing to say; the caller
    is responsible for asking again.

    Both guard clauses (`regions_done <= 0` and `n_entries <= 0`) short-circuit
    before the `logger.info` fill-percentage line, so this pins BOTH by asserting
    no log record at all -- not just no warning. A return value of 0 alone does
    not distinguish the early return from `28 * (0 / regions_done) * n_regions`,
    which is also 0 by construction; only "did it log" tells the two apart.

    Three calls, each isolating a different clause: (0, 0) exercises both at
    once, (0, 10) exercises `n_entries <= 0` with a positive `regions_done`, and
    (1_000_000, 0) exercises `regions_done <= 0` with a positive `n_entries` --
    without that clause the third call divides by zero, so it alone is what
    makes `regions_done <= 0` load-bearing for this test (the first two calls
    never reach it, since `n_entries <= 0` short-circuits first).
    """
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        assert _write._svar2_fill_projection(tmp_path, 0, 0, 100, 500, 2) == 0
        assert not msgs, f"regions_done <= 0 must short-circuit before any log: {msgs}"
        assert _write._svar2_fill_projection(tmp_path, 0, 10, 100, 500, 2) == 0
        assert not msgs, f"n_entries <= 0 must short-circuit before any log: {msgs}"
        assert _write._svar2_fill_projection(tmp_path, 1_000_000, 0, 100, 500, 2) == 0
        assert not msgs, (
            f"regions_done <= 0 must short-circuit even with n_entries > 0: {msgs}"
        )
    finally:
        logger.remove(sink)


@pytest.fixture(scope="module")
def svar2_store_unsorted(vcf_and_ref, tmp_path_factory) -> Path:
    """A store whose own sample order is NOT the lexicographic order gvl writes.

    `write` sorts the selection, so this is the case where `sample_cols` is a real
    permutation and the `samples=None` fast path in `_write_from_svar2` must NOT fire.
    """
    bcf, ref = vcf_and_ref
    from genoray import _core

    out = tmp_path_factory.mktemp("svar2_write_unsorted") / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(out),
        ["S1", "S0"],  # reversed vs. the lexicographic order gvl.write emits
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


def test_write_svar2_sample_cols_permutes_unsorted_store(
    svar2_store_unsorted: Path, tmp_path: Path
):
    """sample_cols must map sorted slot -> store column, not slot -> slot.

    Guards the `list.index` -> hirola swap (#351): `HashTable.add` returns the rank
    in its deduped key array, which equals the store position only because sample
    names are unique. A store that is already sorted cannot tell the two apart, so
    use a reversed one.
    """
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store_unsorted)
    assert svar2.available_samples == ["S1", "S0"], "fixture lost its store order"

    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 5], "chromEnd": [20, 15]}
    )
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=svar2, samples=None, overwrite=True)

    rd = out / "genotypes" / "svar2_ranges"
    sorted_samples = sorted(svar2.available_samples)  # ["S0", "S1"]
    sample_cols = np.load(rd / "sample_cols.npy")
    assert (
        sample_cols.tolist()
        == [svar2.available_samples.index(s) for s in sorted_samples]
        == [1, 0]
    )

    # The cache must be laid out in the SORTED slot order, i.e. match a direct
    # _find_ranges over the sorted names -- the `samples=None` fast path must not
    # have fired and silently written the store's own order.
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    reader = _ranges_reader(rd)
    S, P = len(sorted_samples), svar2.ploidy
    r_q, si_q = np.unravel_index(np.arange(bed.height * S), (bed.height, S))
    got_snp, got_indel = reader.lookup(r_q, si_q, P)

    d = svar2._find_ranges(
        "chr1",
        bed["chromStart"].to_numpy(),
        bed["chromEnd"].to_numpy(),
        samples=sorted_samples,
    )
    exp_snp = np.asarray(d["vk_snp_range"], np.int64).reshape(-1, 2)
    exp_indel = np.asarray(d["vk_indel_range"], np.int64).reshape(-1, 2)
    # Presence is a property of the CELL (either channel non-empty), not of the
    # SNP channel alone -- see the matching comment in test_write_svar2_emits_cache.
    present = (exp_snp[:, 1] > exp_snp[:, 0]) | (exp_indel[:, 1] > exp_indel[:, 0])
    np.testing.assert_array_equal(got_snp[present], exp_snp[present])
    np.testing.assert_array_equal(got_snp[~present], 0)


def test_write_svar2_duplicate_store_samples_raises(
    svar2_store: Path, tmp_path: Path, monkeypatch
):
    """Duplicate sample names in the store must be refused, not silently mapped.

    `list.index` used to point two slots at one column and `HashTable.add` would
    shift every column after the duplicate; both write a wrong dataset (#351).
    """
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    monkeypatch.setattr(svar2, "available_samples", ["S0", "S0", "S1"])

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [20]})
    with pytest.raises(ValueError, match="duplicate sample names"):
        gvl.write(tmp_path / "ds.gvl", bed, variants=svar2, overwrite=True)


def test_fixture_has_empty_cells(svar2_store: Path, tmp_path: Path):
    """The fixture must NOT be 100% fill, or the sparse cache is untested.

    Empty cells are the entire point of the sparse layout (#357): if every
    (region, sample, ploid) window holds a variant, the "cell is absent" branch
    never executes and a sparse/dense divergence there is invisible. S2 is 0|0
    everywhere (empty column) and [25, 40) holds no variants (empty row).

    genoray also routes each variant to the per-sample SPARSE (vk) channel or
    the per-region DENSE channel by carrier-call count (see
    `choose_representation` in genoray's cost model), and only the sparse
    channel reaches this grid. This fixture's single-carrier SNP stays sparse
    but both three-carrier indels route dense, so the grid this test inspects
    is already 1/18 fill by construction, not 8/8 -- exactly one cell
    (region 0, S0, ploid 0) is occupied. A cost-model shift that pushed that
    last SNP dense too would empty the grid entirely and make every
    sparse/dense parity test built on this fixture pass vacuously; guard
    against that directly rather than assuming any occupancy at all.
    """
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    sorted_samples = sorted(svar2.available_samples)
    assert sorted_samples == ["S0", "S1", "S2"], "fixture lost its third sample"

    d = svar2._find_ranges(
        "chr1",
        np.array([0, 5, 25]),
        np.array([20, 15, 40]),
        samples=sorted_samples,
    )
    snp = np.asarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
    indel = np.asarray(d["vk_indel_range"], np.int64)
    nonempty = (snp[:, 1] > snp[:, 0]) | (indel[:, 1] > indel[:, 0])

    S, P = len(sorted_samples), svar2.ploidy
    assert len(nonempty) == 3 * S * P
    # Row-major (R, S, P) -- pinned by the layout oracle in
    # test_write_svar2_emits_cache, which asserts this same reshape against the
    # cache memmaps. Assert each empty structure SEPARATELY: a single
    # `not nonempty.all()` is a disjunction that stays green when either one
    # regresses alone, which is exactly the regression this guard exists to catch.
    grid = nonempty.reshape(3, S, P)
    s2 = sorted_samples.index("S2")
    assert not grid[:, s2].any(), (
        "S2 is no longer all-reference; the empty COLUMN is gone"
    )
    assert not grid[2].any(), (
        "region [25, 40) now holds variants; the empty ROW is gone"
    )
    # Non-vacuity. genoray routes each variant to the per-sample SPARSE (vk)
    # channel or the per-region DENSE channel by carrier-call count, and only
    # the sparse channel lands in this grid: the fixture's single-carrier SNP
    # stays sparse, both three-carrier indels route dense. Exactly one cell is
    # therefore occupied -- (region 0, S0, ploid 0). If a cost-model change
    # pushed that last variant dense too, this grid would be ALL empty and
    # every sparse/dense parity test built on this fixture would pass
    # trivially against an empty table, green and meaningless. Pin it.
    assert grid.any(), (
        "vk channel is entirely empty: genoray routed every variant to the "
        "dense channel, so all sparse-cache parity tests on this fixture are "
        "now vacuous"
    )
    assert grid[0, sorted_samples.index("S0"), 0], (
        "the fixture's one sparse-channel cell (region 0, S0, ploid 0) is gone"
    )


def test_sparse_writer_rejects_oversized_grid(tmp_path):
    """cell_id is int32, so S * P must stay under 2**31."""
    from genvarloader._dataset._svar2_ranges import _SparseWriter

    with pytest.raises(ValueError, match="int32"):
        _SparseWriter(tmp_path, n_samples=2**30, ploidy=2)


def test_sparse_writer_rejects_noncontiguous_regions(tmp_path):
    """The write path's one load-bearing invariant, asserted directly."""
    import numpy as np

    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseWriter

    w = _SparseWriter(tmp_path, n_samples=2, ploidy=2)
    w.append(np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE), lo=0, rc=3)
    with pytest.raises(ValueError, match="contiguous region blocks"):
        w.append(np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE), lo=7, rc=2)
    w.close()


def test_sparse_writer_rejects_global_region_indices(tmp_path):
    """bincount overshoot would silently lengthen region_ptr past R + 1.

    `np.bincount(x, minlength=rc)` returns MORE than `rc` bins when an index
    exceeds `rc` rather than raising, so a caller that passed dataset-global
    region indices would write a longer `region_ptr` than the meta declares and
    the reader would memmap a truncated prefix -- garbage lookups, no error.
    """
    import numpy as np

    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseWriter

    w = _SparseWriter(tmp_path, n_samples=2, ploidy=2)
    with pytest.raises(ValueError, match="contig-local"):
        w.append_contig(
            [np.array([7], np.int32)],
            [np.zeros(1, np.int32)],
            [np.zeros(1, ENTRY_DTYPE)],
            lo=0,
            rc=3,
        )
    w.close()


def test_dense_layout_dataset_still_opens_and_reads(
    svar2_store: Path, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """A pre-0.43.0 (dense-layout) dataset must still open and read end-to-end.

    #357 bumps the on-disk layout but NOT DATASET_FORMAT_VERSION (which matches
    on MAJOR only, so a bump would make new GVL refuse every old dataset). The
    dense reader is what keeps old datasets openable.

    This is a round-trip smoke test, not a parity pin: this fixture's grid has
    only one non-empty cell (see `svar2_store`), so it can't distinguish a
    correct dense reader from one that always returns the same wrong answer.
    `test_dense_ranges_matches_fancy_indexing` in
    `tests/unit/dataset/test_svar2_ranges.py` is what actually pins
    `_DenseRanges.lookup` against dense fancy-indexing over a randomized grid;
    this test's job is only to confirm the dense layout still opens and reads
    through the full `Dataset.open` -> `with_seqs` stack.

    Deviation from the brief: `Dataset.open` (not `gvl.write`) is what takes
    `reference=` -- `with_seqs("haplotypes")` raises `ValueError` without one,
    which the brief's snippet omitted. Added `vcf_and_ref` for the FASTA path.
    """
    from genoray import SparseVar2

    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    _bcf, ref = vcf_and_ref
    bed = pl.DataFrame(
        {"chrom": ["chr1"] * 3, "chromStart": [0, 5, 25], "chromEnd": [20, 15, 40]}
    )
    sparse_ds = tmp_path / "sparse.gvl"
    gvl.write(
        sparse_ds, bed, variants=SparseVar2(svar2_store), samples=None, overwrite=True
    )
    dense_ds = rewrite_as_dense(sparse_ds, tmp_path / "dense.gvl")

    a = gvl.Dataset.open(sparse_ds, reference=ref).with_seqs("haplotypes")
    b = gvl.Dataset.open(dense_ds, reference=ref).with_seqs("haplotypes")
    for r in range(a.n_regions):
        for s in range(a.n_samples):
            np.testing.assert_array_equal(
                np.asarray(a[r, s].to_padded(b"N")),
                np.asarray(b[r, s].to_padded(b"N")),
            )


def test_write_svar2_empty_cell_is_zero_not_insertion_point(
    svar2_store: Path, tmp_path: Path
):
    """A genuinely empty cell reads back as (0, 0), not genoray's insertion point.

    `nonempty_entries` filters on width, so an all-empty (region, sample, ploid)
    cell is never written. `_SparseRanges.lookup` then returns (0, 0) for it --
    the one place `_SparseRanges` and `_DenseRanges` diverge (see
    `_SparseRanges`'s docstring and the comment in `_write_from_svar2`), since a
    dense cell at the same coordinate would instead carry genoray's real
    insertion point (x, x) for x > 0.

    This fixture's grid (bed [0,20)/[5,15)/[25,40) over samples S0/S1/S2) has
    exactly one non-empty cell: (region 0, S0, ploid 0). Region 0 / S1 / ploid 0
    is empty in both channels, and genoray's own dense insertion point there is
    (1, 1) -- confirmed directly via `_find_ranges` below -- so this test would
    fail loudly (assert (1, 1) == (0, 0)) if the sparse writer ever stored
    empty cells verbatim instead of collapsing them to (0, 0).
    """
    from genoray import SparseVar2

    from genvarloader._dataset._svar2_ranges import _ranges_reader

    svar2 = SparseVar2(svar2_store)
    sorted_samples = sorted(svar2.available_samples)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 5, 25],
            "chromEnd": [20, 15, 40],
        }
    )
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=svar2, samples=None, overwrite=True)

    # Confirm genoray's own dense insertion point for the empty cell is
    # nonzero -- i.e. this is actually testing a divergence, not a coincidence.
    d = svar2._find_ranges(
        "chr1",
        bed["chromStart"].to_numpy(),
        bed["chromEnd"].to_numpy(),
        samples=sorted_samples,
    )
    P = svar2.ploidy
    s1 = sorted_samples.index("S1")
    exp_snp = np.asarray(d["vk_snp_range"], np.int64).reshape(bed.height, -1, P, 2)
    exp_indel = np.asarray(d["vk_indel_range"], np.int64).reshape(bed.height, -1, P, 2)
    assert tuple(exp_snp[0, s1, 0]) != (0, 0), (
        "fixture assumption broken: genoray's dense insertion point for the"
        " empty (region 0, S1, ploid 0) cell is expected to be nonzero"
    )
    assert exp_indel[0, s1, 0, 1] == exp_indel[0, s1, 0, 0], (
        "fixture assumption broken: expected the indel channel empty too"
    )

    reader = _ranges_reader(out / "genotypes" / "svar2_ranges")
    got_snp, got_indel = reader.lookup(
        np.array([0], np.int64), np.array([s1], np.int64), P
    )
    np.testing.assert_array_equal(got_snp[0], (0, 0))
    np.testing.assert_array_equal(got_indel[0], (0, 0))
