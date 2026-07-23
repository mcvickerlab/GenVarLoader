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
from genoray._types import DOSAGE_TYPE, V_IDX_TYPE

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


def _build_af_cached_svar1_case(
    streaming_case, tmp_path: Path
) -> tuple[pl.DataFrame, Path, str, "gvl.Dataset"]:
    """Build an AF-cached SVAR1 store: `(regions, reference, variants_path, written)`,
    the same shape `streaming_case` returns. `SparseVar.cache_afs()` must run before
    `gvl.write` for `AF` to show up in the written artifact's schema at all. Shared by
    `test_streaming_svar1_af_matches_written` and
    `test_available_var_fields_af_cached_svar1` -- both need this exact store.
    """
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
    return regions, reference, str(svar), written


# `svar1_multicontig_fixture`'s 7 variants have cached AFs of
# {0.333(chr1:3), 0.667(chr1:7,10; chr2:5,9,21 -- 5 variants), 0.833(chr1:12)}
# (verified directly against `SparseVar.cache_afs()`'s `index["AF"]`). `(0.3, None)`
# from the task brief's template band would keep all 7 (min AF is 0.333 > 0.3) --
# a no-op filter that can't prove partial filtering -- so the min-only band is
# bumped to 0.4 (excludes only the 0.333 variant). The max-only and both-bounds
# bands both already exclude the 0.833 variant unmodified from the brief.
@pytest.mark.parametrize("min_af,max_af", [(0.4, None), (None, 0.7), (0.2, 0.8)])
def test_streaming_svar1_af_matches_written(streaming_case, tmp_path, min_af, max_af):
    regions, reference, svar, written = _build_af_cached_svar1_case(
        streaming_case, tmp_path
    )
    ds = written.with_seqs("variants").with_settings(min_af=min_af, max_af=max_af)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=svar)
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
##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\tS3
chr1\t3\t.\tA\tC,G\t.\t.\tAF=0.1,0.2;DP=11\tGT\t0|1\t1|2\t2|2\t0|0
chr1\t10\t.\tG\tC\t.\t.\tAF=0.42;DP=13\tGT\t1|1\t0|0\t1|0\t0|0
chr1\t16\t.\tT\tA\t.\t.\tAF=0.15;DP=17\tGT\t0|0\t1|1\t0|0\t1|0
chr1\t20\t.\tT\tC\t.\t.\tAF=0.65;DP=19\tGT\t1|0\t0|1\t1|1\t0|0
chr1\t25\t.\tA\tT\t.\t.\tAF=0.85;DP=23\tGT\t0|1\t1|1\t0|0\t1|1
chr1\t30\t.\tG\tA\t.\t.\tAF=0.55;DP=29\tGT\t1|1\t0|0\t0|1\t1|0
"""
# Expected post-`bcftools norm -m -any` AF values, sorted:
# {0.1 (C@3), 0.15 (A@16), 0.2 (G@3), 0.42 (C@10), 0.55 (A@30), 0.65 (C@20),
#  0.85 (T@25)}. 0-based `start` for the multiallelic-derived records is 2
# (pos 3, 1-based).
_AF_MULTIALLELIC_START = 2  # 0-based start of the split pos-3 C/G records

# `DP` is `Number=1` (not `Number=A`/`Number=.`), so unlike `AF` it is NOT
# per-ALT: `bcftools norm -m -any` copies the pre-split value onto BOTH
# records derived from the pos-3 multiallelic site verbatim (no
# ALT-subsetting ambiguity). Keyed by 1-based POS -- both post-norm pos-3
# records (C-derived and G-derived) map to the same DP=11.
_AF_VCF_DP_BY_POS = {3: 11, 10: 13, 16: 17, 20: 19, 25: 23, 30: 29}


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


@pytest.fixture(scope="module")
def af_vcf_case(af_vcf, af_vcf_regions, tmp_path_factory):
    """`(regions, reference, variants, written)` for the AF/DP VCF fixture, the
    same shape `streaming_case` returns -- `written` is opened with `AF` eagerly
    loaded (the one INFO field `gvl.write` actually persists for a VCF source;
    see `_attach_af_column`) so tests can layer `with_settings(var_fields=...)`
    on top without re-triggering the `KeyError: 'AF'` gotcha documented on
    `test_streaming_svar1_af_matches_written`.
    """
    vcf_gz, ref = af_vcf
    out = tmp_path_factory.mktemp("af_vcf_case_ds") / "ds"
    gvl.write(out, af_vcf_regions, variants=str(vcf_gz), overwrite=True)
    written = gvl.Dataset.open(
        out, reference=ref, var_fields=["alt", "ilen", "start", "AF"]
    )
    return af_vcf_regions, ref, str(vcf_gz), written


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


# --- Wave B PR-B3a: var_fields parity (issue #304) --------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_ref_var_field_matches_written(streaming_case, backend):
    """`ref` is byte-identical to the written path on every backend.

    SVAR1 reads REF from its index; VCF/PGEN slice it out of the contig reference at
    [start, start + alt_len - ilen). The written oracle carries REF directly, so this
    is the gate on the reference-slice assumption.

    `ref` is an opaque-string field: indexing a single hap off it (`cell.ref[h]`)
    collapses ALL of that hap's REF alleles into ONE concatenated `bytes` blob
    (confirmed empirically: 3 variants of lengths 1/2/3 -> `b'ACCGGG'`, a 0-d
    scalar under `np.asarray`). That makes the blob-equality assertion below
    boundary-blind (same total bytes with shifted per-variant boundaries would
    still pass) and makes counting via `atleast_1d(...).shape[0]` on the blob
    always exactly 1 -- including for an EMPTY hap, where `b''` is still shape
    `(1,)` (Wave B PR-B3a review, Important 1). Count via `.start[h]` instead,
    which does not have this collapse (matching
    `test_streaming_svar1_af_matches_written`'s pattern), and additionally
    compare per-variant REF lengths via the public `.to_chars()` accessor (which
    restores the per-variant ragged structure the opaque-string blob otherwise
    hides) so a boundary/count divergence is actually caught, not just a total-byte
    match.
    """
    regions, reference, variants, written = streaming_case(backend)
    fields = ["alt", "ilen", "start", "ref"]
    ds = written.with_settings(var_fields=fields).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[k].ref[h]), np.asarray(expected.ref[h])
                )
                # Boundary check (Minor 2, PR-B3a review): the concatenated-bytes
                # equality above can't distinguish "same total REF bytes, different
                # per-variant split" from a genuine match. `.to_chars()` is the
                # public seqpro.rag accessor that recovers per-variant boundaries
                # for an opaque-string field (no private `_rl`/`_layout` reach-in
                # needed); `.lengths` on the resulting per-hap Ragged gives the
                # exact per-variant REF byte length, including 0 variants for an
                # empty hap.
                streamed_ref_lens = data[k].ref.to_chars()[h].lengths
                expected_ref_lens = expected.ref.to_chars()[h].lengths
                np.testing.assert_array_equal(streamed_ref_lens, expected_ref_lens)
                total += np.asarray(data[k].start[h]).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


def test_streaming_vcf_info_var_field_matches_written(af_vcf_case):
    """A Float INFO field (AF) rides along byte-identically between streaming and
    the written path.

    The task brief also asked for a `Type=Integer` ``DP`` case here (the intended
    dtype canary: the written column's dtype comes from genoray's index writer,
    streaming's from the staged `FieldSpec`, and they must agree). That comparison
    is not possible AS SPECIFIED: `gvl.write` never persists a non-``AF`` numeric
    INFO field into a queryable written-path column for a VCF source -- see
    `test_streaming_vcf_available_var_fields_superset_of_written` below for the
    measured mechanism. `written.with_settings(var_fields=[..., "DP"])` raises
    before there is anything to compare. `test_streaming_vcf_dp_field_streaming_only`
    below is the strongest TRUE assertion available for DP: it confirms the written
    path rejects the request, and separately verifies streaming's DP dtype/values
    against the fixture's own known ground truth (not the written path).

    (Minor 3, PR-B3a review): this is single-valued (``AF`` only), not
    parametrized -- ``DP`` can't be tested this way (no written oracle to compare
    against, as explained above) and gets its own test,
    `test_streaming_vcf_dp_field_streaming_only`, instead.
    """
    regions, reference, variants, written = af_vcf_case
    field_name = "AF"
    fields = ["alt", "ilen", "start", field_name]
    ds = written.with_settings(var_fields=fields).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                got = np.asarray(data[k][field_name][h])
                exp = np.asarray(expected[field_name][h])
                assert got.dtype == exp.dtype, (
                    f"{field_name} dtype divergence: streaming {got.dtype} "
                    f"vs written {exp.dtype}"
                )
                np.testing.assert_array_equal(got, exp)
                total += np.atleast_1d(got).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


def test_streaming_vcf_dp_field_streaming_only(af_vcf_case):
    """``DP`` (a `Type=Integer` INFO field declared in the source VCF header, but
    NOT cached into the written `.gvi` index) is streaming-only.

    Measured mechanism (Wave B PR-B3a review): `gvl.write` only ever attaches ONE
    top-level numeric INFO column for a VCF source -- ``AF``, via
    `_attach_af_column` (Wave B PR-B2, #319). Any OTHER declared numeric INFO
    field (here, ``DP``) IS computed by genoray into a nested ``INFO`` struct
    column during `_write_gvi_index` (confirmed directly: `variants.arrow`'s
    schema carries `INFO: Struct({'AF': List(Float32), 'DP': Int32})`), but the
    written-path schema scan (`_Variants.available_info_fields`) only looks at
    TOP-LEVEL numeric columns, so `DP` never reaches `available_var_fields` and
    can never be requested via `with_settings(var_fields=...)`.

    Streaming's VCF backend, by contrast, derives `available_var_fields` from the
    LIVE VCF header (`_declared_info_numeric_dtypes`) and wires every declared
    numeric field through the Rust `VcfWindowFiller`, so it CAN serve `DP`. This
    test confirms both halves: the written path rejects the request, and
    streaming serves correct dtype + values, checked against the fixture's own
    known per-position ground truth (`_AF_VCF_DP_BY_POS`) since there is no
    written oracle to compare against.
    """
    regions, reference, variants, written = af_vcf_case
    fields = ["alt", "ilen", "start", "DP"]

    assert "DP" not in written.available_var_fields
    with pytest.raises(ValueError, match="DP"):
        written.with_settings(var_fields=fields)

    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            for h in range(sds.ploidy):
                starts = np.atleast_1d(np.asarray(data[k].start[h]))
                dps = np.atleast_1d(np.asarray(data[k].DP[h]))
                assert dps.dtype == np.int32, f"DP dtype {dps.dtype}, expected int32"
                assert dps.shape == starts.shape
                for pos0, dp in zip(starts, dps):
                    expected_dp = _AF_VCF_DP_BY_POS[int(pos0) + 1]
                    assert int(dp) == expected_dp, (
                        f"DP mismatch at 0-based start={pos0}: "
                        f"got {dp}, expected {expected_dp}"
                    )
                total += dps.shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", BACKENDS)
def test_available_var_fields_matches_written(streaming_case, backend):
    """Streaming derives the field set from the live source, the written path from
    the written artifact. They must agree, or one of the two definitions has
    drifted -- true (equality) for every `streaming_case` backend used here.

    (Minor 4, PR-B3a review): measured, none of these plain `streaming_case`
    fixtures declare ANY numeric INFO field, and none has a cached AF either --
    all three report only the base set `['alt', 'ilen', 'ref', 'start']`. So this
    equality check never exercises the backend-derived part of
    `available_var_fields` its docstring is nominally about;
    `test_available_var_fields_af_cached_svar1` below exercises that part with a
    store that actually declares `AF`. See
    `test_streaming_vcf_available_var_fields_superset_of_written` below for the
    fixture (`af_vcf_case`, with a `DP` field) where VCF genuinely diverges and
    only `written ⊆ streaming` holds -- that is a real, permanent contract
    difference, not a bug, so it gets its own test rather than a weakened
    assertion here.
    """
    regions, reference, variants, written = streaming_case(backend)
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert set(sds.available_var_fields) == set(written.available_var_fields)


def test_available_var_fields_af_cached_svar1(streaming_case, tmp_path):
    """Strengthens `test_available_var_fields_matches_written` (Improvement (b),
    PR-B3a review): none of the plain `streaming_case` fixtures contribute a
    cached AF or any INFO column, so that test's equality check never exercises
    the backend-derived part of `available_var_fields` its docstring claims to
    gate. This reuses the AF-cached SVAR1 store
    (`_build_af_cached_svar1_case`, shared with
    `test_streaming_svar1_af_matches_written`), where `AF` genuinely appears, and
    asserts both the real equality AND the available-vs-servable gap. Measured:
        written  == streaming == ['AF', 'alt', 'ilen', 'ref', 'start']
        servable == ['alt', 'ilen', 'ref', 'start']  (AF is NOT servable yet)
    """
    regions, reference, svar, written = _build_af_cached_svar1_case(
        streaming_case, tmp_path
    )
    sds = gvl.StreamingDataset(regions, reference=reference, variants=svar)

    expected = {"AF", "alt", "ilen", "ref", "start"}
    assert set(written.available_var_fields) == expected
    assert set(sds.available_var_fields) == expected

    # Available-vs-servable gap: AF is a real on-disk numeric index column (so it
    # is schema-visible / "available") but the streaming engine can't gather it
    # yet ("servable"), deferred to PR-B3b -- see `servable_var_fields`.
    assert "AF" in sds.available_var_fields
    assert "AF" not in sds.servable_var_fields
    assert set(sds.servable_var_fields) == {"alt", "ilen", "ref", "start"}


def test_streaming_vcf_available_var_fields_superset_of_written(af_vcf_case):
    """Measured VCF divergence (Wave B PR-B3a review): streaming's
    `available_var_fields` is a strict superset of the written path's for a VCF
    source declaring a non-``AF`` numeric INFO field.

    Streaming reports every declared numeric INFO field from the LIVE VCF
    header. The written path reports only whatever `_Variants
    .available_info_fields` finds as a TOP-LEVEL numeric column in the written
    `variants.arrow` -- and `gvl.write` only ever attaches one such column for a
    VCF source (`AF`, via `_attach_af_column`, Wave B PR-B2/#319). Measured
    directly against this fixture:
    written  == {'alt', 'ilen', 'start', 'AF', 'ref'}
    streaming == {'alt', 'ilen', 'start', 'AF', 'DP', 'ref'}
    `written ⊆ streaming`, never equal, is the correct contract here -- do not
    loosen streaming's list to match, and do not treat this as a bug to fix
    without a design decision on how (or whether) `gvl.write` should surface
    arbitrary INFO columns as top-level queryable fields.
    """
    regions, reference, variants, written = af_vcf_case
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    written_set = set(written.available_var_fields)
    streaming_set = set(sds.available_var_fields)
    assert written_set <= streaming_set, (
        f"written {sorted(written_set)} must be a subset of "
        f"streaming {sorted(streaming_set)}"
    )
    assert "AF" in written_set and "AF" in streaming_set
    assert "DP" in streaming_set and "DP" not in written_set


# --- Wave B PR-B3b: SVAR1 per-call FORMAT/dosage var_fields parity (issue #304) ---


def _build_svar1_dosage_case(
    streaming_case, tmp_path: Path
) -> tuple[pl.DataFrame, Path, str, "gvl.Dataset"]:
    """SVAR1 store with cached AF (so the AF-filtered parametrization of
    `test_streaming_svar1_dosage_matches_written` actually compacts something --
    same store `_build_af_cached_svar1_case` builds) PLUS a synthetic per-call
    `dosages.npy`.

    None of the shared `streaming_case` fixtures declare a VCF dosage FORMAT
    field, so `SparseVar.from_vcf(with_dosages=True)` isn't an option here --
    this instead synthesizes `dosages.npy` directly on a copy of the store, the
    same trick `tests/integration/dataset/test_issue_191_var_fields.py`'s
    `svar_with_dosages_ds` fixture uses. `dosages.npy` is parallel to
    `variant_idxs.npy` (one float32 value per genotype CALL, NOT per variant);
    `arange`-valued so every CSR position gets a DISTINCT dosage -- this is what
    lets the test actually prove CSR-POSITION (not variant-id) indexing: a
    variant carried by more than one sample/hap gets a DIFFERENT dosage at each
    occurrence, so an indexing bug that gathered by variant id instead of CSR
    position would produce the wrong value, not just the right value in the
    wrong place.
    """
    regions, reference, variants, _ = streaming_case("svar1")
    svar = tmp_path / "dosage.svar"
    shutil.copytree(variants, svar)
    SparseVar(str(svar)).cache_afs()

    n_calls = (svar / "variant_idxs.npy").stat().st_size // np.dtype(
        V_IDX_TYPE
    ).itemsize
    mm = np.memmap(svar / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=(n_calls,))
    mm[:] = np.arange(n_calls, dtype=DOSAGE_TYPE)
    mm.flush()
    del mm

    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(svar), overwrite=True)
    written = gvl.Dataset.open(
        out, reference=reference, var_fields=["alt", "ilen", "start", "AF"]
    )
    return regions, reference, str(svar), written


@pytest.mark.parametrize("min_af,max_af", [(None, None), (0.2, 0.8)])
def test_streaming_svar1_dosage_matches_written(
    streaming_case, tmp_path, min_af, max_af
):
    """SVAR1 `dosage` rides along byte-identically, and is compacted by the SAME
    AF/region keep mask as `start`/`ilen` (hence the AF-filtered parametrization).

    `dosage` is stored parallel to `variant_idxs.npy` on the SAME hap-major CSR
    offsets -- i.e. indexed by CSR POSITION, not by variant id (the opposite of
    Task 5/PR-B3a's INFO columns, which ARE indexed by variant id). See
    `_build_svar1_dosage_case` for how the fixture's `arange`-valued dosages make
    this test actually discriminate the two indexing schemes rather than merely
    exercise the code path.
    """
    regions, reference, svar, written = _build_svar1_dosage_case(
        streaming_case, tmp_path
    )
    fields = ["alt", "ilen", "start", "dosage"]
    ds = written.with_settings(
        var_fields=fields, min_af=min_af, max_af=max_af
    ).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=svar)
        .with_settings(var_fields=fields, min_af=min_af, max_af=max_af)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                got = np.asarray(data[k].dosage[h])
                np.testing.assert_array_equal(got, np.asarray(expected.dosage[h]))
                total += np.atleast_1d(got).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", ["vcf", "pgen"])
def test_dosage_var_field_rejected_on_record_backends(streaming_case, backend):
    """`gvl.write` never persists dosage for VCF/PGEN sources, so a written dataset
    from the same source cannot serve it either -- streaming declines symmetrically.
    """
    regions, reference, variants, _written = streaming_case(backend)
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "dosage" not in sds.available_var_fields
    with pytest.raises(ValueError, match="not available"):
        sds.with_settings(var_fields=["alt", "start", "dosage"])
