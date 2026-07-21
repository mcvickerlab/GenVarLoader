"""Coverage gap closed post-Wave-A-review (issue #277): no test previously
composed ``with_seqs("annotated")`` with ``with_len(L)`` or with read-time
jitter -- the two other Wave A output-mode knobs (Tasks 2 and 3). Both are
shipped, independently tested combinators; this file locks their
intersection with annotated output (Task 4) for all three streaming
backends.

Mirrors the two existing single-knob suites:
  - `test_streaming_with_len.py` (`test_fixed_length_matches_written`): at
    jitter=0, fixed-length output must be BYTE-IDENTICAL to the written
    oracle -- `with_len` alone never perturbs data, only truncates/pads
    ragged length.
  - `test_streaming_jitter.py` (`test_jitter_reproducible_same_rng`): jitter
    is a seeded AUGMENTATION, not byte-parity with the written `Dataset` (a
    jittered read window has no single "correct" written-oracle answer) --
    so the jitter test compares two independently-constructed streamed runs
    with the same `rng` to each other, not to `written`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

# Both tests here compose `with_seqs("annotated")`, which is a fail-fast
# NotImplementedError for the VCF backend (its `var_idxs` are dataset-global ids
# a VCF source cannot produce cheaply; issues #305, #311). SVAR1 (ids for free)
# and PGEN (ids from the `.pvar` row index) cover the composition; VCF is excluded.
# See test_streaming_annotated_parity.py::test_vcf_annotated_fails_fast.
BACKENDS = ["svar1", "pgen"]

# <= the smallest region length across all three `streaming_case` backends
# (the SVAR1 multi-contig fixture uses 20bp sliding windows; VCF/PGEN span a
# single 250bp contig) -- see `test_streaming_with_len.py`'s module docstring
# for why this bound matters (avoids tripping `Dataset.with_len`'s
# max-output-length-vs-region-length guard for any backend).
_LENGTHS = [10, 16]

# `with_len(L)` anchors the fixed-length window at the region's chromStart
# (jitter=0, no randomization) -- see `Dataset.with_len`/`get_haps_and_shifts`.
# `streaming_case`'s vcf/pgen regions span the FULL 250bp `vcf_snp_ins_del_multi`
# contig from chromStart=0, so a short `with_len(10/16)` window there ([0,16))
# never reaches that fixture's first variant (pos=29 SNP) -- the non-vacuousness
# guard would fail. Narrowing chromStart to 20 keeps the window ([20, 20+L))
# anchored just before pos=29 for both `_LENGTHS` values, so the SNP is always
# captured, while leaving plenty of room before `chromEnd=250` for the
# max-output-length guard. svar1's regions (20bp sliding windows already
# containing early variants) need no narrowing.
_VCF_PGEN_FIXED_LEN_START = 20


def _fixed_length_case(request, tmp_path_factory, backend: str):
    """Like `streaming_case`, but narrows the vcf/pgen regions so a
    start-anchored `with_len` window lands on a variant (see
    `_VCF_PGEN_FIXED_LEN_START` above). svar1 is unaffected -- its
    `streaming_case` regions already contain variants near the start."""
    if backend == "svar1":
        f = request.getfixturevalue("svar1_multicontig_fixture")
        written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        return f.bed, f.reference_path, f.svar_path, written

    if backend == "vcf":
        f = request.getfixturevalue("vcf_snp_ins_del_multi")
        variants = str(f.vcf)
    elif backend == "pgen":
        f = request.getfixturevalue("pgen_snp_ins_del_multi")
        variants = str(f.pgen)
    else:
        raise ValueError(f"_fixed_length_case: unknown backend {backend!r}")

    regions = f.regions.with_columns(pl.col("chromStart") + _VCF_PGEN_FIXED_LEN_START)
    out = tmp_path_factory.mktemp(f"fl_{backend}") / "ds"
    gvl.write(out, regions, variants=variants, overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    return regions, str(f.fasta), variants, written


def _assert_annotated_row_matches(
    data, row: int, expected, ploidy: int, length: "int | None" = None
) -> bool:
    """Per-haplotype comparison of one streamed batch row (a
    `RaggedAnnotatedHaps` batch's `row`'th cell, accessed via
    `data.haps[row][h]`/etc. -- see `test_annotated_matches_written`'s same
    indexing pattern) against an oracle `RaggedAnnotatedHaps` cell (indexed
    only by `h`, e.g. `ds[r, s]`). Returns whether any hap in this cell
    carried a variant, so callers can accumulate a non-vacuousness guard
    across cells without re-looping.
    """
    saw_variant = False
    for h in range(ploidy):
        got_haps = np.asarray(data.haps[row][h])
        got_vidx = np.asarray(data.var_idxs[row][h])
        got_pos = np.asarray(data.ref_coords[row][h])
        exp_haps = np.asarray(expected.haps[h])
        exp_vidx = np.asarray(expected.var_idxs[h])
        exp_pos = np.asarray(expected.ref_coords[h])
        if length is not None:
            assert got_haps.shape[-1] == length, (
                f"hap {h}: expected fixed length {length}, got shape {got_haps.shape}"
            )
        np.testing.assert_array_equal(got_haps, exp_haps)
        np.testing.assert_array_equal(
            got_vidx,
            exp_vidx,
            err_msg=(
                f"hap={h}: streamed var_idxs must be dataset-GLOBAL, matching "
                "the written oracle exactly"
            ),
        )
        np.testing.assert_array_equal(got_pos, exp_pos)
        if np.any(exp_vidx >= 0):
            saw_variant = True
    return saw_variant


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", _LENGTHS)
def test_annotated_fixed_length_matches_written(
    request, tmp_path_factory, backend, length
):
    """annotated x with_len: byte-parity composition, not augmentation --
    layering `with_len(length)` on `with_seqs("annotated")` must still match
    the written oracle exactly (haps, var_idxs, ref_coords) at jitter=0, with
    every hap truncated/padded to exactly `length`."""
    regions, reference, variants, written = _fixed_length_case(
        request, tmp_path_factory, backend
    )

    ds = written.with_len(length).with_seqs("annotated")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_len(length)
        .with_seqs("annotated")
    )

    n_cells = 0
    saw_variant = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            r, s = int(r_idx[row]), int(s_idx[row])
            exp = ds[r, s]  # RaggedAnnotatedHaps, fixed-length
            if _assert_annotated_row_matches(data, row, exp, sds.ploidy, length=length):
                saw_variant = True
            n_cells += 1

    assert n_cells > 0, "streaming_case fixture yielded no cells"
    assert saw_variant, (
        f"backend={backend} length={length}: fixture produced no "
        "variant-carrying haplotype; the var_idxs comparison proves nothing "
        "without one"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_annotated_jitter_reproducible(streaming_case, backend):
    """annotated x jitter: reproducibility, NOT written-oracle byte-parity --
    jitter translates the read window per region (seeded by `rng`), so there
    is no single "correct" written-`Dataset` answer to compare against (see
    `test_streaming_jitter.py`'s module docstring). Two independently
    constructed `StreamingDataset`s with the same `rng=0` must produce
    identical annotated output (haps, var_idxs, ref_coords) cell-for-cell."""
    regions, reference, variants, _written = streaming_case(backend)

    def mk():
        return (
            gvl.StreamingDataset(regions, reference=reference, variants=variants)
            .with_seqs("annotated")
            .with_settings(jitter=8, rng=0, deterministic=False)
        )

    def collect(sds):
        """Extract each cell's (haps, var_idxs, ref_coords) per hap into a
        plain dict keyed by (region, sample), rather than holding the
        streamed batch objects -- makes a later "compare cell A to cell B"
        pass independent of which batch/row each came from."""
        cells: dict[
            tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        for data, r_idx, s_idx in sds.to_iter(batch_size=4):
            for row in range(len(r_idx)):
                r, s = int(r_idx[row]), int(s_idx[row])
                cells[(r, s)] = [
                    (
                        np.asarray(data.haps[row][h]),
                        np.asarray(data.var_idxs[row][h]),
                        np.asarray(data.ref_coords[row][h]),
                    )
                    for h in range(sds.ploidy)
                ]
        return cells

    sds_a, sds_b = mk(), mk()
    cells_a = collect(sds_a)
    cells_b = collect(sds_b)

    assert cells_a.keys() == cells_b.keys()
    assert len(cells_a) > 0, "streaming_case fixture yielded no cells"

    saw_variant = False
    for key in cells_a:
        for (a_haps, a_vidx, a_pos), (b_haps, b_vidx, b_pos) in zip(
            cells_a[key], cells_b[key]
        ):
            np.testing.assert_array_equal(a_haps, b_haps)
            np.testing.assert_array_equal(
                a_vidx,
                b_vidx,
                err_msg=(
                    f"cell={key}: annotated var_idxs must be reproducible "
                    "across two runs with the same rng"
                ),
            )
            np.testing.assert_array_equal(a_pos, b_pos)
            if np.any(a_vidx >= 0):
                saw_variant = True

    assert saw_variant, (
        f"backend={backend}: fixture produced no variant-carrying haplotype "
        "under jitter; the var_idxs reproducibility comparison proves "
        "nothing without one"
    )
