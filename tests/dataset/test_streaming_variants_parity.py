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

import json
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


def _build_af_cached_svar1_store(
    streaming_case, tmp_path: Path, svar_name: str
) -> tuple[pl.DataFrame, Path, Path]:
    """Shared setup for every AF-cached SVAR1 fixture in this module (Minor 4, PR-B3b
    review, #304): copy `streaming_case("svar1")`'s store to `tmp_path/svar_name` and
    `SparseVar.cache_afs()` it, WITHOUT writing/opening a `gvl.Dataset` yet -- callers
    add their own per-call field file(s) (dosage, a custom FORMAT column, ...) on top
    of the copy before writing. Returns `(regions, reference, svar_path)`.
    """
    regions, reference, variants, _ = streaming_case("svar1")
    svar = tmp_path / svar_name
    shutil.copytree(variants, svar)
    SparseVar(str(svar)).cache_afs()
    return regions, reference, svar


def _write_and_open_af_cached(
    out: Path, regions: pl.DataFrame, reference: Path, svar: Path
) -> "gvl.Dataset":
    """`gvl.write` the given (already AF-cached) store and open it back. `var_fields`
    must explicitly list "AF" -- `Dataset.open`'s default var_fields (`["alt", "ilen",
    "start"]`) never eagerly loads INFO columns, so without this the cached AF is
    present in the on-disk schema (passing the `Haps.__post_init__` availability
    check) but absent from the loaded `variants.info` dict, and
    `with_settings(min_af=...)` fails downstream at read time with `KeyError: 'AF'`.
    Same pattern as `test_query_filters.py`/`test_unphased_union.py`.
    """
    gvl.write(out, regions, variants=str(svar), overwrite=True)
    return gvl.Dataset.open(
        out, reference=reference, var_fields=["alt", "ilen", "start", "AF"]
    )


def _build_af_cached_svar1_case(
    streaming_case, tmp_path: Path
) -> tuple[pl.DataFrame, Path, str, "gvl.Dataset"]:
    """Build an AF-cached SVAR1 store: `(regions, reference, variants_path, written)`,
    the same shape `streaming_case` returns. Shared by
    `test_streaming_svar1_af_matches_written` and
    `test_available_var_fields_af_cached_svar1` -- both need this exact store.
    """
    regions, reference, svar = _build_af_cached_svar1_store(
        streaming_case, tmp_path, "af.svar"
    )
    written = _write_and_open_af_cached(tmp_path / "ds", regions, reference, svar)
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
    same store `_build_af_cached_svar1_case` builds, via the shared
    `_build_af_cached_svar1_store`/`_write_and_open_af_cached` helpers -- Minor 4,
    PR-B3b review, #304) PLUS a synthetic per-call `dosages.npy`.

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
    regions, reference, svar = _build_af_cached_svar1_store(
        streaming_case, tmp_path, "dosage.svar"
    )

    n_calls = (svar / "variant_idxs.npy").stat().st_size // np.dtype(
        V_IDX_TYPE
    ).itemsize
    mm = np.memmap(svar / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=(n_calls,))
    mm[:] = np.arange(n_calls, dtype=DOSAGE_TYPE)
    mm.flush()
    del mm

    written = _write_and_open_af_cached(tmp_path / "ds", regions, reference, svar)
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

    # Minor 4 (PR-B3b review, #304): lock in that the `(0.2, 0.8)` parametrization
    # is a genuine PARTIAL filter, not decorative -- mirrors
    # `test_streaming_svar1_af_matches_written`'s `0 < total < n_unfiltered`
    # pattern. Measured: this band drops exactly the `chr1:12` variant
    # (AF=0.833) from the 7-variant fixture.
    if min_af is not None or max_af is not None:
        unfiltered = written.with_settings(var_fields=fields).with_seqs("variants")
        n_unfiltered = sum(
            np.atleast_1d(np.asarray(unfiltered[r, s].dosage[h])).shape[0]
            for r in range(ds.shape[0])
            for s in range(ds.shape[1])
            for h in range(sds.ploidy)
        )
        assert 0 < total < n_unfiltered


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


# --- Important 1 (PR-B3b review, #304): custom (non-dosage) FORMAT field dtype
# preservation -- the pre-fix code coerced EVERY non-float per-call field to
# int32 (`np.ascontiguousarray(_arr, np.int32)`), silently breaking byte-identical
# parity for a native int16 field (genoray's real one, `mutcat`) and leaving the
# Rust `CallVals::I32`/streaming int branch entirely untested (only float32
# `dosage` was ever exercised). This section closes that gap: a custom int16
# FORMAT field, registered exactly the way
# `tests/integration/dataset/test_issue_231_custom_format_fields.py`'s
# `custom_field_ds` fixture does (metadata.json["fields"] + a raw `<name>.npy`
# memmap) -- constructing it this way (rather than running
# `SparseVar.annotate_mutcat(write_back=True)`) is the "most direct means
# available" in a test fixture: `annotate_mutcat` needs the full
# adjacency/reference-classification machinery, which none of the shared
# `streaming_case` fixtures are built to feed, whereas the hand-registration
# trick is already the codebase's own established way to test this exact
# surface (issue #231) and is dtype-source-agnostic. ---------------------------

_CUSTOM_FIELD_NAME = "mutcat"
_CUSTOM_FIELD_DTYPE = "int16"


def _build_svar1_custom_format_case(
    streaming_case, tmp_path: Path
) -> tuple[pl.DataFrame, Path, str, "gvl.Dataset"]:
    """SVAR1 store with cached AF (so the AF-filtered parametrization below
    actually compacts something, same as `_build_svar1_dosage_case`) PLUS a
    synthetic int16 custom FORMAT field named `mutcat` -- genoray's one real
    custom FORMAT field (see `genoray/_svar/_annotate.py`'s
    `SparseVar.annotate_mutcat(write_back=True)`, which always registers it as
    `int16`). `arange`-valued for the SAME CSR-position-vs-variant-id reason
    `_build_svar1_dosage_case` documents.
    """
    regions, reference, svar = _build_af_cached_svar1_store(
        streaming_case, tmp_path, "custom_fmt.svar"
    )

    n_calls = (svar / "variant_idxs.npy").stat().st_size // np.dtype(
        V_IDX_TYPE
    ).itemsize
    mm = np.memmap(
        svar / f"{_CUSTOM_FIELD_NAME}.npy",
        dtype=_CUSTOM_FIELD_DTYPE,
        mode="w+",
        shape=(n_calls,),
    )
    mm[:] = np.arange(n_calls, dtype=_CUSTOM_FIELD_DTYPE)
    mm.flush()
    del mm

    # Register the custom field in the SVAR metadata -- same as
    # `test_issue_231_custom_format_fields.py`'s `custom_field_ds` fixture.
    meta_path = svar / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["fields"] = {_CUSTOM_FIELD_NAME: _CUSTOM_FIELD_DTYPE}
    meta_path.write_text(json.dumps(meta))

    written = _write_and_open_af_cached(tmp_path / "ds", regions, reference, svar)
    return regions, reference, str(svar), written


@pytest.mark.parametrize("min_af,max_af", [(None, None), (0.2, 0.8)])
def test_streaming_svar1_custom_format_field_matches_written(
    streaming_case, tmp_path, min_af, max_af
):
    """Custom (non-dosage) per-call FORMAT field parity, INCLUDING dtype.

    This is the regression test for Important 1 (PR-B3b review, #304): the
    pre-fix code force-cast every non-float per-call field through
    `np.ascontiguousarray(_arr, np.int32)`, so a native `int16` field like
    `mutcat` streamed back as `int32` -- value-equal but dtype-DIFFERENT from
    the written path (which preserves the registered dtype verbatim, see
    `_haps.py`'s `custom_fmt[name]` memmap and
    `test_issue_231_custom_format_fields.py:113`'s own dtype assertion). Byte-
    identical parity requires the EXACT dtype, not just equal values, so this
    test asserts `.dtype` explicitly on top of `assert_array_equal`. It also
    exercises the Rust `CallVals::I16`/`InfoVals::I16` branch, which
    `test_streaming_svar1_dosage_matches_written` (float32-only) never touches.
    """
    regions, reference, svar, written = _build_svar1_custom_format_case(
        streaming_case, tmp_path
    )
    fields = ["alt", "ilen", "start", _CUSTOM_FIELD_NAME]
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
                got = np.asarray(data[k][_CUSTOM_FIELD_NAME][h])
                exp = np.asarray(expected[_CUSTOM_FIELD_NAME][h])
                np.testing.assert_array_equal(got, exp)
                assert got.dtype == np.dtype(_CUSTOM_FIELD_DTYPE), (
                    f"streamed dtype {got.dtype} != registered dtype "
                    f"{_CUSTOM_FIELD_DTYPE} -- custom FORMAT field dtype was "
                    "coerced, breaking byte-identical parity"
                )
                assert exp.dtype == np.dtype(_CUSTOM_FIELD_DTYPE)
                total += np.atleast_1d(got).shape[0]
    assert total > 0, "vacuous pass: no variants compared"

    # Same partial-filter lock-in as `test_streaming_svar1_dosage_matches_written`
    # (Minor 4, PR-B3b review): the `(0.2, 0.8)` band must actually drop something.
    if min_af is not None or max_af is not None:
        unfiltered = written.with_settings(var_fields=fields).with_seqs("variants")
        n_unfiltered = sum(
            np.atleast_1d(np.asarray(unfiltered[r, s][_CUSTOM_FIELD_NAME][h])).shape[0]
            for r in range(ds.shape[0])
            for s in range(ds.shape[1])
            for h in range(sds.ploidy)
        )
        assert 0 < total < n_unfiltered


def test_available_var_fields_includes_custom_format_field(streaming_case, tmp_path):
    """`mutcat` shows up in both `available_var_fields` and `servable_var_fields`
    for streaming -- the int16 dtype is one of the three the FFI can preserve
    exactly (Important 1, PR-B3b review), so it is NOT relegated to
    available-but-not-servable the way a numeric INDEX column (e.g. `AF`) is.
    """
    regions, reference, svar, _written = _build_svar1_custom_format_case(
        streaming_case, tmp_path
    )
    sds = gvl.StreamingDataset(regions, reference=reference, variants=svar)
    assert _CUSTOM_FIELD_NAME in sds.available_var_fields
    assert _CUSTOM_FIELD_NAME in sds.servable_var_fields


# --- Wave B PR-B4: variant-windows byte-identical parity (issue #304) --------
#
# `StreamingDataset.with_seqs("variant-windows", opt)` drives Task 8's Rust
# `generate_variant_windows` kernel + Task 9's FFI/Python packing. This
# compares that streamed `dict[str, Ragged]` output, per BATCH, against the
# written oracle queried with the SAME (r_idx, s_idx) arrays in one shot:
# `gvl.write` + `Dataset.open(...).with_output_format("flat")
# .with_seqs("variant-windows", opt)[r_idx, s_idx].to_ragged()` -- the query
# boundary requires the flat output format for variant-windows (`Haps.__call__`
# raises otherwise, see `_haps.py`), and the query result is a
# `_FlatVariantWindows` whose `.to_ragged()` is what actually produces the
# `dict[str, Ragged]` shape comparable to streaming's per-batch dict.
#
# Adaptations from the brief's literal skeleton (see the task report for the
# full list):
#   1. `with_output_format("flat")` is required and was missing from the
#      brief -- without it the query raises "requires the flat output format".
#   2. The brief indexes the written oracle PER CELL (`ds[r, s]`) and per-hap
#      (`data[name][k, h]`, a tuple index). Measured empirically (task report):
#      `seqpro.rag.Ragged.__getitem__` on these doubly-ragged (variant count,
#      window length) token fields does NOT do independent per-axis numpy-style
#      fancy indexing -- a tuple index and sequential bracket indexing both
#      descend into the SAME nested structure one level per index, so neither
#      reliably resolves "the (batch=k, ploidy=h) cell" once more than one
#      ragged axis is involved, and `np.asarray()` on a partially-resolved cell
#      raises "cannot convert a jagged Ragged to a dense array" whenever that
#      cell holds more than one variant with a different window length (e.g. a
#      SNP next to an indel -- exactly the svar1 fixture's shape). Querying the
#      written oracle with the WHOLE BATCH's index arrays at once (matching
#      streaming's per-batch shape exactly) and comparing via `.to_padded()`
#      sidesteps this entirely: both sides pad to a dense array with the same
#      out-of-band sentinel, so real data compares index-for-index and any
#      genuine shape divergence (variant count or window length) still surfaces
#      as an assertion failure.
#
# `var_fields` is intentionally left at its default (`["alt", "ilen", "start"]`)
# throughout -- the written path DOES support `var_fields` ride-along columns
# in window mode, but streaming's Python packing silently drops `info_out` in
# the windows branch (a known, separately-tracked follow-up, not in scope
# here). With defaults, window fields are exactly `{start, ilen}` plus the
# token buffers (`ref_window`/`alt_window` or `ref`/`alt`, depending on `opt`).


def _sentinel_for(dtype) -> int:
    """An out-of-band `.to_padded()` fill value for `dtype`, guaranteed not to
    collide with any real value this test's tiny fixtures can produce (token
    ids, `start`/`ilen`). Padding only ever fills POSITIONS BEYOND each row's
    real length, so as long as the exact same sentinel is used on both the
    streaming and written side, padded positions compare sentinel-to-sentinel
    regardless of what the value would mean as real data -- the dtype's own
    extreme bound is always a safe, simple choice.
    """
    dt = np.dtype(dtype)
    if dt.kind == "u":
        return int(np.iinfo(dt).max)
    if dt.kind == "i":
        return int(np.iinfo(dt).min)
    raise TypeError(f"_sentinel_for: no sentinel defined for dtype {dt}")


def _assert_field_matches(name: str, got, exp, *, check_dtype: bool) -> int:
    """Compare one whole-batch `dict[str, Ragged]` field (streaming vs the
    written oracle queried with the same batch index arrays) and return the
    number of variants it carries (summed over every (region, sample, ploid)
    group), for the caller's vacuous-pass guard. See the module docstring for
    why this pads+densifies rather than indexing per-(k, h) cell."""
    if check_dtype:
        assert got.dtype == exp.dtype, (
            f"{name} dtype divergence: streaming {got.dtype} vs written {exp.dtype}"
        )
    assert got.shape == exp.shape, (
        f"{name} shape divergence: streaming {got.shape} vs written {exp.shape}"
    )
    sentinel = _sentinel_for(got.dtype)
    np.testing.assert_array_equal(got.to_padded(sentinel), exp.to_padded(sentinel))
    # `.lengths` is the per-(region, sample, ploid) VARIANT COUNT for every
    # field here (scalar fields have one ragged axis over variants; window
    # fields have a second, inner ragged axis over window length that
    # `.lengths` does not report -- confirmed empirically, task report) --
    # summing it is exact and dtype-agnostic, unlike counting via an
    # opaque-string field (see the module docstring's earlier `ref`/`alt`
    # collapse warnings elsewhere in this file).
    return int(np.asarray(got.lengths).sum())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ref_mode", ["window", "allele"])
@pytest.mark.parametrize("alt_mode", ["window", "allele"])
@pytest.mark.parametrize("alphabet,unk", [(b"ACGT", 4), (b"ACGTN", 5)])
def test_streaming_variant_windows_matches_written(
    streaming_case, backend, ref_mode, alt_mode, alphabet, unk
):
    """All four (ref, alt) mode combinations, both token dtypes, all three
    backends, byte-identical against the written `_FlatVariantWindows
    .to_ragged()` output.

    PGEN's `start` column has a REAL, PRE-EXISTING, mode-independent dtype
    divergence shared with the already-merged `with_seqs("variants")` path
    (measured directly -- see the task report for the exact dtypes observed
    per backend): the written oracle's `start` comes from
    `haps.variants.start`, which is int64 for PGEN specifically (int32 for
    SVAR1 and VCF), while streaming always emits int32 `start` for every
    backend. `with_seqs("variants")`'s own parity test
    (`test_streaming_variants_matches_written`, above) never caught this
    because it compares via dtype-agnostic `np.testing.assert_array_equal`.
    Here we assert dtype for every field/backend EXCEPT this one documented
    case, where we still assert full value equality (just not dtype) -- a
    genuine divergence must stay explicit, not silently dropped from the gate.
    """
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants, written = streaming_case(backend)
    opt = VarWindowOpt(
        flank_length=4,
        token_alphabet=alphabet,
        unknown_token=unk,
        ref=ref_mode,
        alt=alt_mode,
    )
    ds = written.with_output_format("flat").with_seqs("variant-windows", opt)
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variant-windows", opt)

    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        expected = ds[np.asarray(r_idx), np.asarray(s_idx)].to_ragged()
        assert set(data) == set(expected), (
            f"field-set mismatch: streaming {sorted(data)} vs written "
            f"{sorted(expected)}"
        )
        for name in data:
            # See docstring: PGEN `start` is dtype-divergent (int32 streaming
            # vs int64 written) for a pre-existing, variant-windows-independent
            # reason -- values must still match exactly.
            check_dtype = not (backend == "pgen" and name == "start")
            n = _assert_field_matches(
                name, data[name], expected[name], check_dtype=check_dtype
            )
            if name == "start":
                total += n
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", BACKENDS)
def test_variant_windows_empty_group_matches_written(empty_region_case, backend):
    """A (region, sample, ploid) cell with no in-window variant stays empty on both
    paths at the default `dummy_variant=None` -- no sentinel fill on either side.
    """
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants, written = empty_region_case(backend)
    opt = VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
    ds = written.with_output_format("flat").with_seqs("variant-windows", opt)
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variant-windows", opt)
    saw_empty = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        expected = ds[np.asarray(r_idx), np.asarray(s_idx)].to_ragged()
        # No dtype assertion here (mirrors the brief): this test's job is the
        # empty-group VALUE contract, not the separately-tracked PGEN `start`
        # dtype divergence (see `test_streaming_variant_windows_matches_written`).
        _assert_field_matches(
            "start", data["start"], expected["start"], check_dtype=False
        )
        if np.any(np.asarray(data["start"].lengths) == 0):
            saw_empty = True
    assert saw_empty, "vacuous pass: fixture had no empty groups"
