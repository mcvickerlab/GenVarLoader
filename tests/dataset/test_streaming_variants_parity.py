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

import numpy as np
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
