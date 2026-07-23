"""Wave B PR-B1: streaming with_seqs("variants") is byte-identical to the written path.

`StreamingDataset.with_seqs("variants")` drives `RecordStreamEngine.next_batch_variants`
(`RecordBackend::generate_variants`, Task 2's Rust core exposed via Task 3's FFI) and packs
the returned flat buffers into a `RaggedVariants`. This compares that streamed output,
cell by cell, against the SAME VCF/PGEN source written via `gvl.write` and read back with
`Dataset.with_seqs("variants")` -- two independent decoders (Rust `ChunkAssembler` for
streaming vs. Python cyvcf2/pgenlib + `dense2sparse` for the written path) that must agree
byte-for-byte on `alt`/`start`/`ilen`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from genoray import SparseVar

import genvarloader as gvl

BACKENDS = ["svar1", "vcf", "pgen"]


def _assert_variants_cell_matches(streamed, expected, ploidy) -> int:
    """Assert the streamed cell matches the written cell hap-by-hap; return the total
    number of variants seen across all haps (so callers can guard against a vacuous
    all-empty pass -- see the module docstring's byte-identity claim)."""
    n_variants = 0
    for h in range(ploidy):
        streamed_alt = np.asarray(streamed.alt[h])
        np.testing.assert_array_equal(streamed_alt, np.asarray(expected.alt[h]))
        np.testing.assert_array_equal(
            np.asarray(streamed.start[h]), np.asarray(expected.start[h])
        )
        np.testing.assert_array_equal(
            np.asarray(streamed.ilen[h]), np.asarray(expected.ilen[h])
        )
        # A ragged hap with exactly one variant collapses `.alt[h]` to a 0-d scalar
        # (bytes) rather than a length-1 array -- `atleast_1d` normalizes both cases
        # before counting.
        n_variants += np.atleast_1d(streamed_alt).shape[0]
    return n_variants


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_variants_matches_written(streaming_case, backend):
    regions, reference, variants, written = streaming_case(backend)
    ds = written.with_seqs("variants")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variants")

    seen = set()
    total_variants = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            total_variants += _assert_variants_cell_matches(
                data[k], ds[r, s], sds.ploidy
            )
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}
    # Guard against a vacuous pass: the streaming_case fixtures carry real SNP/INS/DEL
    # variants, so a byte-identical parity check that saw zero variants everywhere would
    # be trivially (and wrongly) green.
    assert total_variants > 0


# `svar1_multicontig_fixture`'s 7 variants have cached AFs of
# {0.333(chr1:3), 0.667(chr1:7,10; chr2:5,9,21 -- 5 variants), 0.833(chr1:12)}
# (verified directly against `SparseVar.cache_afs()`'s `index["AF"]`). `(0.3, None)`
# from the task brief's template band would keep all 7 (min AF is 0.333 > 0.3) --
# a no-op filter that can't prove partial filtering -- so the min-only band is
# bumped to 0.4 (excludes only the 0.333 variant). The max-only and both-bounds
# bands both already exclude the 0.833 variant unmodified from the brief.
@pytest.mark.parametrize("min_af,max_af", [(0.4, None), (None, 0.7), (0.2, 0.8)])
def test_streaming_svar1_af_matches_written(streaming_case, tmp_path, min_af, max_af):
    regions, reference, variants, _ = streaming_case("svar1")
    svar = tmp_path / "af.svar"
    shutil.copytree(variants, svar)
    SparseVar(str(svar)).cache_afs()
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(svar), overwrite=True)
    # `var_fields` must explicitly list "AF" -- `Dataset.open`'s default var_fields
    # (`["alt", "ilen", "start"]`) never eagerly loads INFO columns, so without
    # this the cached AF is present in the on-disk schema (passing the
    # `Haps.__post_init__` availability check) but absent from the loaded
    # `variants.info` dict, and `with_settings(min_af=...)` fails downstream at
    # read time with `KeyError: 'AF'`. Same pattern as
    # `test_query_filters.py`/`test_unphased_union.py`.
    written = gvl.Dataset.open(
        out, reference=reference, var_fields=["alt", "ilen", "start", "AF"]
    )
    ds = written.with_seqs("variants").with_settings(min_af=min_af, max_af=max_af)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=str(svar))
        .with_seqs("variants")
        .with_settings(min_af=min_af, max_af=max_af)
    )

    # Count via `.start[h]` rather than `.alt[h]`: `_assert_variants_cell_matches`'s
    # own `.alt[h]` count (used for its no-filter vacuous-pass guard, where every
    # cell has >=1 variant) collapses a hap with exactly 1 variant AND a hap with
    # 0 variants to the SAME 0-d `b""`-shaped scalar -- `atleast_1d(...).shape[0]`
    # then reports 1 for both. This fixture's 20bp windows mean nearly every
    # (region, sample, hap) has 0 or 1 variant, so that ambiguity would make
    # `total`/`n_unf` constant (= n_regions * n_samples * ploidy) regardless of
    # the AF filter, silently defeating the `0 < total < n_unf` proof below.
    # `.start[h]` doesn't have this collapse (confirmed empirically: an empty
    # hap's `.start[h]` is a proper 0-length array, never a scalar), so counting
    # variants off it is exact for both 0 and 1-variant groups.
    seen, total = set(), 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_variants_cell_matches(data[k], ds[r, s], sds.ploidy)
            for h in range(sds.ploidy):
                total += np.asarray(data[k].start[h]).shape[0]
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}

    unfiltered = written.with_seqs("variants")
    n_unf = sum(
        np.asarray(unfiltered[r, s].start[h]).shape[0]
        for r in range(ds.shape[0])
        for s in range(ds.shape[1])
        for h in range(sds.ploidy)
    )
    # Proves this is a genuine partial filter (not vacuously all-kept or
    # all-dropped): at least one variant survives the AF band and at least one
    # is excluded by it.
    assert 0 < total < n_unf


# --- VCF/BCF INFO/AF parity (Task 10, #319) ---------------------------------
#
# Written-path AF is read via `_fetch_info_cols` at `gvl.write` time (cached
# into the `.gvi`, Task 5); streaming-path AF is read live per-window by the
# Rust `FieldSpec` (`VcfWindowFiller`). These are two independently
# implemented AF-reading mechanisms that must agree byte-for-byte, INCLUDING
# on a multiallelic source record -- gvl requires bi-allelic input, so a
# multiallelic record only reaches gvl after `bcftools norm -m -any` atomizes
# it into one biallelic record per ALT, each carrying that ALT's slice of the
# original `Number=A` INFO/AF list. This is the primary parity risk: if
# streaming and written resolve "which post-split AF belongs to this ALT"
# differently, they diverge only on multiallelic-derived records.
#
# Reference (1-based positions annotated):
#   pos:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
#   base: A  C  A  C  G  T  A  C  G  G  A  C  G  T  A  T  C  G  A  T
#   pos: 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
#   base: C  G  A  C  A  G  C  T  A  G  C  A  G  T  C
_AF_REF = "ACACGTACGGACGTATCGATCGACAGCTAGCAGTC"
assert len(_AF_REF) == 35
for _pos, _base in [(3, "A"), (10, "G"), (16, "T"), (20, "T"), (25, "A"), (30, "G")]:
    assert _AF_REF[_pos - 1] == _base

# Pre-normalization source: one multiallelic SNP (pos 3, A -> C,G,
# AF=0.1,0.2) plus 5 biallelic SNPs spanning a range of AF values. Sample
# genotypes at pos 3 are chosen so the post-split C/G records land on
# different haps of different samples (S0: C only; S1: C on hap0, G on
# hap1; S2: G only; S3: neither) -- this is what lets the multiallelic
# assertion below distinguish "the C-derived record was dropped" from "the
# G-derived record was kept" per-hap, not just per-cell.
_AF_VCF_RAW = f"""\
##fileformat=VCFv4.2
##contig=<ID=chr1,length={len(_AF_REF)}>
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\tS3
chr1\t3\t.\tA\tC,G\t.\t.\tAF=0.1,0.2\tGT\t0|1\t1|2\t2|2\t0|0
chr1\t10\t.\tG\tC\t.\t.\tAF=0.42\tGT\t1|1\t0|0\t1|0\t0|0
chr1\t16\t.\tT\tA\t.\t.\tAF=0.15\tGT\t0|0\t1|1\t0|0\t1|0
chr1\t20\t.\tT\tC\t.\t.\tAF=0.65\tGT\t1|0\t0|1\t1|1\t0|0
chr1\t25\t.\tA\tT\t.\t.\tAF=0.85\tGT\t0|1\t1|1\t0|0\t1|1
chr1\t30\t.\tG\tA\t.\t.\tAF=0.55\tGT\t1|1\t0|0\t0|1\t1|0
"""
# Expected post-`bcftools norm -m -any` AF values, sorted:
# {0.1 (C@3), 0.15 (A@16), 0.2 (G@3), 0.42 (C@10), 0.55 (A@30), 0.65 (C@20),
#  0.85 (T@25)}. 0-based `start` for the multiallelic-derived records is 2
# (pos 3, 1-based).
_AF_MULTIALLELIC_START = 2  # 0-based start of the split pos-3 C/G records


def _build_af_vcf(tmp_path: Path) -> tuple[Path, Path]:
    """Build the reference + a bi-allelically normalized, bgzipped+indexed
    VCF from `_AF_VCF_RAW`. Mirrors `test_write_af.py:_build_indexed_vcf`
    plus the `bcftools norm -m -any` atomization step from
    `tests/_builders/case.py`.
    """
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_AF_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    raw = tmp_path / "raw.vcf"
    raw.write_text(_AF_VCF_RAW)
    raw_gz = tmp_path / "raw.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(raw_gz), str(raw)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(raw_gz)], check=True)

    normed_gz = tmp_path / "normed.vcf.gz"
    subprocess.run(
        [
            "bcftools",
            "norm",
            "-m",
            "-any",
            "-f",
            str(ref),
            "-Oz",
            "-o",
            str(normed_gz),
            str(raw_gz),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["bcftools", "index", "-t", str(normed_gz)], check=True)

    # Confirm normalization actually split the multiallelic record: no
    # remaining record should have a comma in its ALT column.
    view = subprocess.run(
        ["bcftools", "view", "-H", str(normed_gz)],
        check=True,
        capture_output=True,
        text=True,
    )
    alts = [line.split("\t")[4] for line in view.stdout.splitlines() if line]
    assert len(alts) == 7, f"expected 7 post-norm records, got {len(alts)}: {alts}"
    assert all("," not in alt for alt in alts), (
        f"post-norm VCF still has a multiallelic ALT: {alts}"
    )

    return normed_gz, ref


@pytest.fixture(scope="module")
def af_vcf(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("af_vcf_src")
    return _build_af_vcf(d)


@pytest.fixture(scope="module")
def af_vcf_regions() -> pl.DataFrame:
    return pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_AF_REF)]}
    )


@pytest.mark.parametrize(
    "min_af,max_af",
    [
        (0.5, None),  # min-only: keeps {0.55, 0.65, 0.85}, drops {0.1, 0.15, 0.2, 0.42}
        (None, 0.5),  # max-only: keeps {0.1, 0.15, 0.2, 0.42}, drops {0.55, 0.65, 0.85}
        # Both-bounds: keeps {0.15, 0.2, 0.42, 0.55, 0.65}, drops {0.1, 0.85}.
        # Crucially this band splits the multiallelic pos-3 pair: G (AF=0.2)
        # is kept, C (AF=0.1) is dropped -- the primary parity risk.
        (0.12, 0.7),
    ],
)
def test_streaming_vcf_af_matches_written(
    af_vcf, af_vcf_regions, tmp_path, min_af, max_af
):
    vcf_gz, ref = af_vcf
    out = tmp_path / "ds"
    gvl.write(out, af_vcf_regions, variants=str(vcf_gz), overwrite=True)

    # See `test_streaming_svar1_af_matches_written`: `var_fields` must
    # explicitly include "AF" or `with_settings(min_af=...)` fails at read
    # time with `KeyError: 'AF'` even though the `.gvi` has AF cached.
    written = gvl.Dataset.open(
        out, reference=ref, var_fields=["alt", "ilen", "start", "AF"]
    )
    ds = written.with_seqs("variants").with_settings(min_af=min_af, max_af=max_af)
    sds = (
        gvl.StreamingDataset(af_vcf_regions, reference=str(ref), variants=str(vcf_gz))
        .with_seqs("variants")
        .with_settings(min_af=min_af, max_af=max_af)
    )

    seen, total = set(), 0
    streamed_cells: dict[tuple[int, int], object] = {}
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_variants_cell_matches(data[k], ds[r, s], sds.ploidy)
            # Count via `.start[h]`, never `.alt[h]` -- see
            # `test_streaming_svar1_af_matches_written`'s comment: an empty
            # hap's `.alt[h]` collapses to the same 0-d shape as a
            # single-variant hap, saturating an `.alt`-based count.
            for h in range(sds.ploidy):
                total += np.asarray(data[k].start[h]).shape[0]
            seen.add((r, s))
            streamed_cells[(r, s)] = data[k]
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}

    unfiltered = written.with_seqs("variants")
    n_unf = sum(
        np.asarray(unfiltered[r, s].start[h]).shape[0]
        for r in range(ds.shape[0])
        for s in range(ds.shape[1])
        for h in range(sds.ploidy)
    )
    # Proves this is a genuine partial filter, not vacuously all-kept/dropped.
    assert 0 < total < n_unf

    # --- Explicit multiallelic-derived-AF assertion -------------------------
    # Expected per-hap presence of the pos-3 C-derived (AF=0.1) and
    # G-derived (AF=0.2) records, from the source GTs:
    #   S0 = 0|1 (C only)   S1 = 1|2 (C on hap0, G on hap1)
    #   S2 = 2|2 (G only)   S3 = 0|0 (neither)
    # Under a band with min_af <= 0.2 <= max_af < 0.1's reach (e.g. the
    # both-bounds (0.12, 0.7) parametrization), the C record is dropped
    # everywhere and the G record is kept everywhere it occurs.
    keep_c = min_af is None or min_af <= 0.1
    keep_c = keep_c and (max_af is None or max_af >= 0.1)
    keep_g = min_af is None or min_af <= 0.2
    keep_g = keep_g and (max_af is None or max_af >= 0.2)
    expected_has_c = {0: True, 1: True, 2: False, 3: False}  # per-sample: any hap has C
    expected_has_g = {0: False, 1: True, 2: True, 3: False}  # per-sample: any hap has G

    for s in range(ds.shape[1]):
        want_c = keep_c and expected_has_c[s]
        want_g = keep_g and expected_has_g[s]
        want_present = want_c or want_g
        for label, cell in [
            ("written", ds[0, s]),
            ("streamed", streamed_cells[(0, s)]),
        ]:
            has_multiallelic_start = any(
                _AF_MULTIALLELIC_START in np.asarray(cell.start[h])
                for h in range(sds.ploidy)
            )
            assert has_multiallelic_start == want_present, (
                f"{label} sample S{s}: expected pos-3 record presence "
                f"{want_present} (want_c={want_c}, want_g={want_g}), "
                f"got {has_multiallelic_start}; "
                f"start={[np.asarray(cell.start[h]).tolist() for h in range(sds.ploidy)]}"
            )
