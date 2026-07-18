"""Shared fixtures for tests/dataset/."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import pyBigWig

import genvarloader as gvl

SEQLEN = 20

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6
# (C>CAT), dense SNP@9 (G>C, carried by 3 haps), DEL@11 (GTA>G, ilen -2).
# Mirrors the SVAR2-parity fixture in tests/dataset/test_svar2_dataset.py
# (`_src`/`svar_fixture`) -- same VCF, reused here for the streaming SVAR1
# window-read parity test (Task 4).
_SVAR1_STREAM_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_SVAR1_STREAM_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@dataclass(slots=True)
class Svar1DatasetFixture:
    """Matched inputs for the streaming SVAR1 window-read parity test: the
    live `.svar` store + reference + 1-region bed to drive `_Svar1Backend`
    directly, and an independently-written+opened `gvl.Dataset` over the same
    store/bed/reference to compare against."""

    svar_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset: gvl.Dataset


@pytest.fixture(scope="module")
def svar1_dataset_fixture(tmp_path_factory) -> Svar1DatasetFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_stream_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_STREAM_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_stream_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    out = tmp_path_factory.mktemp("svar1_stream_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)
    dataset = gvl.Dataset.open(out, reference=ref)

    return Svar1DatasetFixture(
        svar_path=svar_path,
        reference_path=ref,
        contigs=["chr1"],
        bed=bed,
        dataset=dataset,
    )


# Two-contig reference + variants for the Task 5 multi-region/multi-contig
# streaming parity test. chr1 reuses `_SVAR1_STREAM_REF`/its 4 variants
# (SNP@3, INS@7, SNP@10, DEL@12); chr2 is a distinct 40bp reference with its
# own SNP + insertion, matching the `_REF2`/`_VCF2` pattern in
# `test_svar2_dataset.py`'s multicontig fixtures. Extended to 3 samples.
_SVAR1_MC_REF2 = "TTGGCCAATTGGCCAATTACGTACGTTTGGCCAATTGGCC"
_SVAR1_MC_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##contig=<ID=chr2,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|1
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t1|0
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0\t0|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\t1|1
chr2\t5\t.\tC\tT\t.\t.\t.\tGT\t1|0\t1|1\t0|1
chr2\t9\t.\tT\tTGG\t.\t.\t.\tGT\t0|1\t1|0\t1|1
chr2\t21\t.\tG\tA\t.\t.\t.\tGT\t1|1\t0|1\t1|0
"""


@dataclass(slots=True)
class Svar1MultiContigFixture:
    """Matched inputs for the Task 5 public `StreamingDataset` parity test:
    an unsorted, interleaved-contig bed (>=2 contigs, >=10 regions, 3
    samples) over a live `.svar` store + reference, plus the path to an
    independently-written `gvl.Dataset` directory over the same bed/store to
    compare against (opened lazily by the test via `gvl.Dataset.open`)."""

    svar_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset_path: Path


@pytest.fixture(scope="module")
def svar1_multicontig_fixture(tmp_path_factory) -> Svar1MultiContigFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_mc_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n>chr2\n{_SVAR1_MC_REF2}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_MC_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_mc_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path,
        VCF(bcf),
        max_mem="1g",
        samples=["S0", "S1", "S2"],
        overwrite=True,
    )

    # 12 regions (6 per contig, 20bp sliding windows), UNSORTED / interleaved
    # chrom order -- exercises the original-bed-row-order <-> sorted-storage
    # translation on both the streaming and written-dataset sides.
    starts = [0, 4, 8, 12, 16, 20]
    rows = []
    for i, s in enumerate(starts):
        # alternate contigs per row so the bed is genuinely interleaved, not
        # just "all of chr2 then all of chr1".
        c1, c2 = ("chr2", "chr1") if i % 2 == 0 else ("chr1", "chr2")
        rows.append({"chrom": c1, "chromStart": s, "chromEnd": s + 20})
        rows.append({"chrom": c2, "chromStart": s, "chromEnd": s + 20})
    bed = pl.DataFrame(rows)

    out = tmp_path_factory.mktemp("svar1_mc_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)

    return Svar1MultiContigFixture(
        svar_path=svar_path,
        reference_path=ref,
        contigs=["chr1", "chr2"],
        bed=bed,
        dataset_path=out,
    )


@dataclass(slots=True)
class Svar1MixedNamingFixture:
    """Regression fixture for the `ref_c_idx` contig-naming-style bug: the `.svar`
    store/VCF uses UCSC-style contig names (``chr1``) while one of the two
    references provided uses Ensembl style (``1``, no prefix). `Reference.from_path`
    documents mixed UCSC/Ensembl naming as supported (it normalizes contig names to
    match the FASTA), so a correct `StreamingDataset` must handle this pairing.

    `reference_chr_path` names its single contig ``chr1`` (matches the store) and
    is used to build/open the comparison `gvl.Dataset`; `reference_ensembl_path`
    names it ``1`` and is the one actually fed to `StreamingDataset` under test --
    same underlying sequence, different naming style, so streamed output must
    still be byte-identical to the written-and-opened comparison dataset."""

    svar_path: Path
    reference_ensembl_path: Path
    reference_chr_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset_path: Path


@pytest.fixture(scope="module")
def svar1_mixed_naming_fixture(tmp_path_factory) -> Svar1MixedNamingFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_mixed_naming_src")
    ref_chr = d / "ref_chr.fa"
    ref_chr.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref_chr)], check=True)

    ref_ensembl = d / "ref_ensembl.fa"
    ref_ensembl.write_text(f">1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref_ensembl)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_STREAM_VCF)  # contig "chr1", UCSC-style, like the store
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_mixed_naming_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    out = tmp_path_factory.mktemp("svar1_mixed_naming_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)

    return Svar1MixedNamingFixture(
        svar_path=svar_path,
        reference_ensembl_path=ref_ensembl,
        reference_chr_path=ref_chr,
        contigs=["chr1"],
        bed=bed,
        dataset_path=out,
    )


# Samples in NATIVE VCF column order deliberately not already lexicographically
# sorted ("S10" < "S2" lexicographically, but S10 is declared before S2 here) --
# see `sample_idx` semantics fixture below. Single diagnostic SNP at chr1:3 (A>G,
# 0-based pos 2): S1 and S10 are homozygous REF ("A"), S2 is homozygous ALT ("G"),
# so the three samples are distinguishable by the base at that position without
# needing 3-way-distinct genotypes.
_SVAR1_SAMPLE_ORDER_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS10\tS2\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|0\t1|1\t0|0
"""


@dataclass(slots=True)
class Svar1SampleOrderFixture:
    """Regression fixture for `StreamingDataset.samples`: native VCF column order
    (S10, S2, S1) differs from lexicographic order (S1, S10, S2), and a diagnostic
    SNP distinguishes S2 (ALT="G") from S1/S10 (REF="A") at 0-based position 2.
    `expected_base_by_sorted_name` gives the expected byte at that position, keyed
    by sorted sample name, so a test can check `sds.samples[i]`'s reconstructed
    data against the *name*, not just against a re-sorted list (which would be
    tautological)."""

    svar_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    diagnostic_pos: int  # 0-based position of the diagnostic SNP
    expected_base_by_sorted_name: dict[str, bytes]


@pytest.fixture(scope="module")
def svar1_sample_order_fixture(tmp_path_factory) -> Svar1SampleOrderFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_sample_order_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_SAMPLE_ORDER_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_sample_order_store") / "store.svar"
    # `samples=` here is the store's NATIVE column order -- deliberately the same
    # (non-lexicographic) order as the VCF header, not pre-sorted.
    SparseVar.from_vcf(
        svar_path, VCF(bcf), max_mem="1g", samples=["S10", "S2", "S1"], overwrite=True
    )

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    return Svar1SampleOrderFixture(
        svar_path=svar_path,
        reference_path=ref,
        contigs=["chr1"],
        bed=bed,
        diagnostic_pos=2,
        expected_base_by_sorted_name={"S1": b"A", "S10": b"A", "S2": b"G"},
    )


@pytest.fixture(scope="session")
def snap_dataset(source_bed, vcf_dir, reference, tmp_path_factory):
    """Phased VCF dataset with a "5ss" BigWig track, opened with a reference.

    Mirrors the ``base_ds`` fixture in ``tests/dataset/test_with_methods.py``.
    Opened with default settings (output_length="ragged", sequence_type="haplotypes",
    jitter=0, max_jitter=2, deterministic=True, rc_neg=True).
    """
    from genoray import VCF

    tmp_dir = tmp_path_factory.mktemp("snap_ds")
    out = tmp_dir / "snap.gvl"

    vcf_samples = ["s0", "s1", "s2"]
    # Header lengths are generous upper bounds for the regions in source.bed.
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(vcf_samples):
        bw_path = tmp_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            # One short interval per contig region in source.bed; values differ
            # per sample. Mirrors base_ds in tests/dataset/test_with_methods.py.
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[value, value, value, value],
            )
        bw_paths[sample] = str(bw_path)

    bigwigs = gvl.BigWigs("5ss", bw_paths)
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(
        path=out,
        bed=source_bed,
        variants=vcf,
        tracks=bigwigs,
        max_jitter=2,
    )
    return gvl.Dataset.open(out, reference=reference)


@dataclass(slots=True)
class StreamingVcfFixture:
    """VCF-source counterpart to `svar1_dataset_fixture` for the record-stream
    engine (`RecordStreamEngine`, issue #276 tasks 5+). Reuses the two-sample/
    two-variant VCF fixture Task 4 committed for the Rust `VcfWindowFiller`
    tests (`tests/data/streaming/two_var_two_sample.vcf.gz` +
    `src/record_stream/vcf.rs`'s `vcf_filler_decodes_window_to_local_table`),
    so the Rust and Python suites exercise the SAME variant data: chr1
    SNP@POS=11 (0-based 10, A>G) and DEL@POS=21 (0-based 20, ACGT>A,
    ilen=-3), samples s1/s2, ploidy 2.

    `chr1_ref_bytes` mirrors the 100 'A' bytes the Rust fixture uses (long
    enough to cover both variants and a `[0, 100)` test region). `fasta` is a
    matching FASTA (same 100bp of 'A', samtools-indexed) for callers that
    want a real reference path (e.g. a future left-align opt-in); the Task 5
    FFI-seam test itself passes `fasta_path=None`, matching `gvl.write`'s VCF
    parity contract (no read-time left-align -- see `vcf.rs`'s module doc).
    `regions` is a single-contig, single-region bed in the same
    `{"chrom", "chromStart", "chromEnd"}` shape `gvl.write`/`StreamingDataset`
    use, so Task 6 can reuse it directly.
    """

    vcf: Path
    fasta: Path
    chr1_ref_bytes: bytes
    sample_names: list[str]
    n_samples: int
    ploidy: int
    regions: pl.DataFrame


@pytest.fixture(scope="module")
def streaming_vcf_fixture(tmp_path_factory) -> StreamingVcfFixture:
    vcf = (
        Path(__file__).parent.parent
        / "data"
        / "streaming"
        / "two_var_two_sample.vcf.gz"
    )
    chr1_ref_bytes = b"A" * 100

    d = tmp_path_factory.mktemp("streaming_vcf_fasta")
    fasta = d / "ref.fa"
    fasta.write_text(f">chr1\n{chr1_ref_bytes.decode()}\n")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)

    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(chr1_ref_bytes)]}
    )

    return StreamingVcfFixture(
        vcf=vcf,
        fasta=fasta,
        chr1_ref_bytes=chr1_ref_bytes,
        sample_names=["s1", "s2"],
        n_samples=2,
        ploidy=2,
        regions=regions,
    )


# 250bp reference for the Task 7 parity-gate fixture (issue #276). Built once via
# vcfixture's `ReferenceBuilder(seed=7)` + explicit overrides at each variant locus
# (anchor bases pinned to match the VCF's REF alleles, flank-guard bases pinned to
# something other than the indel's trailing base to prevent `bcftools norm`
# left-alignment from silently shifting a variant off its intended position) --
# see `.superpowers/sdd/task-7-report.md` for the exact generation script. Stored as
# a literal here (like `_SVAR1_STREAM_REF` above) so the fixture doesn't need a
# checked-in FASTA binary; only the harder-to-regenerate VCF is committed.
_VCF_PARITY_REF = (
    "TGGTGTTAACCTTACTATACTCCCGCTCCAGGGTTTGGCTCATATGAACAAGTCTTTGCG"
    "CCCATAAAGCTAGCCAGTGAGCTTAGTTGGAGCAAGGGGTGCGGAAGCGGTACTCCGTCG"
    "CGCGGGTAGCCAACTACTTAAGACCTAGGATTCTGTTGCAGATTAGAACTTGGGACTCAA"
    "GATTGCTGCCCTAAGCTATACTAGGCAGCTGCAGCGTCTGGTTTTACTCAGTGTGATCTT"
    "TATGCTTGAG"
)
assert len(_VCF_PARITY_REF) == 250


@dataclass(slots=True)
class VcfSnpInsDelMultiFixture:
    """Task 7 (issue #276) parity-gate fixture: a richer VCF than
    `streaming_vcf_fixture` covering a SNP, an insertion (ILEN > 0), a deletion
    (ILEN < 0), and a multiallelic site (2 ALTs, pre-split biallelic by
    `bcftools norm -m -` since `gvl.write` rejects multi-allelic records) across
    3 samples -- exercises ILEN's sign both ways plus the biallelic-split path
    on the SAME position (both split atoms anchor at 0-based pos 149).

    Committed under `tests/data/streaming/vcf_snp_ins_del_multi.vcf.gz` (+ `.tbi`)
    already left-aligned/atomized/split via `bcftools norm -f <ref> -a
    --atom-overlaps . -f <ref> -m -` against `_VCF_PARITY_REF`, so both the
    written-dataset oracle (`gvl.write`, Python cyvcf2 decode) and the streamed
    table (`RecordStreamEngine.debug_decode_window`, Rust `ChunkAssembler`
    decode) read the identical already-normalized records -- the differential
    test is checking that two INDEPENDENT decoders agree on an already-atomic
    input, not asking either one to normalize anything.

    Variants (0-based pos, REF>ALT, ilen):
      pos=29   A>G     ilen=0   (SNP)
      pos=69   C>CAT   ilen=+2  (insertion)
      pos=109  GTAC>G  ilen=-3  (deletion)
      pos=149  A>G     ilen=0   (multiallelic split, atom 1)
      pos=149  A>T     ilen=0   (multiallelic split, atom 2)

    Same-POS tie-break CAVEAT: the streamed decoder (Rust `ChunkAssembler`)
    orders same-POS atoms via a `BinaryHeap` keyed on `(pos, seq)` --
    lexicographic ALT order -- while the written oracle (`gvl.write`) preserves
    genoray's `.gvi` file-row order (VCF file order). For the pos=149 pair
    above, `A>G` sorts before `A>T` BOTH lexicographically and in file order,
    so the two mechanisms agree COINCIDENTALLY. This is not a guarantee: if a
    future edit reorders these split ALTs, or adds a same-POS triallelic site
    where lexicographic order != file order, the two decoders would legitimately
    tie-break differently and `test_streaming_vcf_parity.py` would fail for a
    reason unrelated to any real decoder bug. Don't reorder/add same-POS atoms
    here without re-checking that lexicographic and file order still coincide
    (or updating the parity test to tolerate a permutation).
    """

    vcf: Path
    fasta: Path
    contig: str
    n_samples: int
    ploidy: int
    sample_names: list[str]
    regions: pl.DataFrame


@pytest.fixture(scope="module")
def vcf_snp_ins_del_multi(tmp_path_factory) -> VcfSnpInsDelMultiFixture:
    vcf = (
        Path(__file__).parent.parent
        / "data"
        / "streaming"
        / "vcf_snp_ins_del_multi.vcf.gz"
    )

    d = tmp_path_factory.mktemp("vcf_snp_ins_del_multi_fasta")
    fasta = d / "ref.fa"
    fasta.write_text(f">chr1\n{_VCF_PARITY_REF}\n")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)

    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_VCF_PARITY_REF)]}
    )

    return VcfSnpInsDelMultiFixture(
        vcf=vcf,
        fasta=fasta,
        contig="chr1",
        n_samples=3,
        ploidy=2,
        sample_names=["s0", "s1", "s2"],
        regions=regions,
    )


@pytest.fixture(scope="module")
def vcf_snp_ins_del_multi_regions(
    vcf_snp_ins_del_multi: VcfSnpInsDelMultiFixture,
) -> pl.DataFrame:
    """Task 8 (issue #276): a 3-region bed splitting `vcf_snp_ins_del_multi`'s
    single 250bp contig into disjoint sub-windows, over the SAME committed VCF
    + FASTA (no fixture regeneration needed -- `regions` is the only axis that
    changes). This exercises `RecordBackend::generate`'s per-(region, sample)
    CSR expansion (`src/record_stream/engine.rs`, the Critical bug fixed in
    Task 3b) across >=2 regions in one window, which the single-region
    `vcf_snp_ins_del_multi.regions` (reused unmodified by Task 7's table gate)
    does not.

    Region boundaries land clear of every variant's extent so no variant spans
    a region edge:
      region 0 [0, 90)    contains pos=29 (SNP) and pos=69 (INS, extent [69,70))
      region 1 [90, 170)  contains pos=109 (DEL, extent [109,113)) and both
                           pos=149 multiallelic-split atoms
      region 2 [170, 250) no variants -- pure-reference region
    """
    contig = vcf_snp_ins_del_multi.contig
    return pl.DataFrame(
        {
            "chrom": [contig, contig, contig],
            "chromStart": [0, 90, 170],
            "chromEnd": [90, 170, 250],
        }
    )


# Task 8 (issue #276) multi-contig fixture: hand-authored (already-atomic,
# already-left-of-any-ambiguity) SNP/INS/DEL records across TWO contigs, built
# inline (bgzip+tabix at fixture time) rather than committed as a binary --
# unlike `vcf_snp_ins_del_multi` (which needed `vcfixture` + `bcftools norm`
# to produce a genuinely-multiallelic-then-split site), these records need no
# normalization pass: neither `gvl.write`'s VCF read path nor the streaming
# `VcfWindowFiller` left-aligns or REF-checks when `fasta_path=None` (see
# `src/record_stream/vcf.rs`'s module doc), so a hand-written already-biallelic
# VCF is read literally by both decoders -- same convention `_SVAR1_MC_VCF`
# already uses for its inline (non-committed) multi-contig fixture.
_VCF_MC_REF1 = (
    "GACGGGATCTGCGGTACGCATAACTTTGCGTGAATCGATGATCTGCTGATATTCTATTATCCATTGAATG"
    "CTTGGCCCCCCTGCAGTCATAATCGTCATAGTCAGTATGCTCCACTCATC"
)
assert len(_VCF_MC_REF1) == 120
_VCF_MC_REF2 = (
    "CAGTCCACGGCTGTCATGAGCTCAAACTTATGGCCAAATGAAACTCGTGCGTTAGAACAACTCTCCTATA"
    "CACCTACTTGCACGAGCGGCCATCGACGGT"
)
assert len(_VCF_MC_REF2) == 100
# chr1: SNP@0-based-20 (T>C), INS@0-based-50 (A>AGG, ilen+2), DEL@0-based-85
# (GTCA>G, ilen-3). chr2: SNP@0-based-15 (A>G), SNP@0-based-70 (C>T). REF
# alleles match `_VCF_MC_REF1`/`_VCF_MC_REF2` at each 0-based position exactly.
_VCF_MC_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=120>
##contig=<ID=chr2,length=100>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2
chr1\t21\t.\tT\tC\t.\t.\t.\tGT\t1|0\t0|1\t0|0
chr1\t51\t.\tA\tAGG\t.\t.\t.\tGT\t0|1\t1|1\t1|0
chr1\t86\t.\tGTCA\tG\t.\t.\t.\tGT\t1|0\t0|1\t1|1
chr2\t16\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|0\t1|1
chr2\t71\t.\tC\tT\t.\t.\t.\tGT\t1|1\t0|1\t0|0
"""


@dataclass(slots=True)
class VcfMultiContigFixture:
    """Task 8 (issue #276) end-to-end parity fixture spanning two contigs, so
    the streamed haplotype comparison exercises the engine's per-contig window
    boundary (`_plan`'s contig-run splitting, `_streaming.py`) in addition to
    the per-region CSR expansion `vcf_snp_ins_del_multi_regions` covers."""

    vcf: Path
    fasta: Path
    contigs: list[str]
    n_samples: int
    ploidy: int
    sample_names: list[str]
    regions: pl.DataFrame


@pytest.fixture(scope="module")
def vcf_multi_contig(tmp_path_factory) -> VcfMultiContigFixture:
    d = tmp_path_factory.mktemp("vcf_mc_src")
    fasta = d / "ref.fa"
    fasta.write_text(f">chr1\n{_VCF_MC_REF1}\n>chr2\n{_VCF_MC_REF2}\n")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)

    vcf_txt = d / "in.vcf"
    vcf_txt.write_text(_VCF_MC_VCF)
    vcf_gz = d / "in.vcf.gz"
    with vcf_gz.open("wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf_txt)], stdout=fh, check=True)
    subprocess.run(["tabix", "-p", "vcf", str(vcf_gz)], check=True)

    # 2 regions per contig (chr1: variants split SNP+INS in region 0, DEL in
    # region 1; chr2: one SNP per region) -- contiguous per-contig row blocks,
    # matching `_plan`'s `np.diff(contig_idxs)` run-detection requirement.
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "chromStart": [0, 60, 0, 50],
            "chromEnd": [60, 120, 50, 100],
        }
    )

    return VcfMultiContigFixture(
        vcf=vcf_gz,
        fasta=fasta,
        contigs=["chr1", "chr2"],
        n_samples=3,
        ploidy=2,
        sample_names=["s0", "s1", "s2"],
        regions=regions,
    )
