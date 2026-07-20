"""Task 4 (issue #277, Wave A): annotated output (``AnnotatedHaps``) must be
byte-identical to the written oracle for all three streaming backends
(SVAR1, VCF, PGEN) at jitter=0, including the emitted ``var_idxs`` -- which
must be dataset-GLOBAL variant ids (see the streaming engines' ``var_base``
plumbing: SVAR1 for free, PGEN via the ``.pvar`` per-contig pre-scan, VCF via
``var_base=0`` -- exact for these single-contig/whole-contig fixtures, see
``_VcfBackend.build_engine``'s doc comment and GitHub issue #305 for the
documented multi-contig gap).

``streaming_case`` (``tests/dataset/conftest.py``) supplies
``(regions, reference, variants, written)``; ``written`` is a plain
``gvl.Dataset.open(...)`` with no output mode applied, so this test layers
``with_seqs("annotated")`` on both `written` and a freshly constructed
`StreamingDataset` for the same inputs.

Per-hap comparison (not a whole-cell densify): at jitter=0/ragged length, a
cell's two haplotypes may have different lengths (indels), so
``ds[r, s]``/the streamed row are themselves ragged over ploidy -- mirrors
`test_streaming_parity.py`'s per-hap `expected[h]` pattern rather than
padding a whole (ploidy, len) block (which would require picking a padding
sentinel for `ref_coords`, adding a needless second thing to get right).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

BACKENDS = ["svar1", "vcf", "pgen"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_annotated_matches_written(streaming_case, backend):
    regions, reference, variants, written = streaming_case(backend)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("annotated")

    n_cells = 0
    saw_variant = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # RaggedAnnotatedHaps
            for h in range(sds.ploidy):
                got_haps = np.asarray(data.haps[row][h])
                got_vidx = np.asarray(data.var_idxs[row][h])
                got_pos = np.asarray(data.ref_coords[row][h])
                exp_haps = np.asarray(exp.haps[h])
                exp_vidx = np.asarray(exp.var_idxs[h])
                exp_pos = np.asarray(exp.ref_coords[h])
                np.testing.assert_array_equal(got_haps, exp_haps)
                np.testing.assert_array_equal(
                    got_vidx,
                    exp_vidx,
                    err_msg=(
                        f"backend={backend} row={row} hap={h}: streamed "
                        "var_idxs must be dataset-GLOBAL, matching the "
                        "written oracle exactly"
                    ),
                )
                np.testing.assert_array_equal(got_pos, exp_pos)
                if np.any(exp_vidx >= 0):
                    saw_variant = True
            n_cells += 1
    assert n_cells > 0, "streaming_case fixture yielded no cells"
    assert saw_variant, (
        f"backend={backend}: fixture produced no variant-carrying haplotype; "
        "the var_idxs comparison proves nothing without one"
    )


_NARROWED_PGEN_REGIONS = pl.DataFrame(
    {"chrom": ["chr1"], "chromStart": [71], "chromEnd": [200]}
)


def _narrowed_pgen_case(pgen_snp_ins_del_multi, tmp_path):
    """Shared setup for the narrowed-window `pgen_snp_ins_del_multi` tests
    below: writes the oracle dataset and constructs a matching
    `StreamingDataset`, both restricted to region `[71, 200)`. See
    `test_annotated_pgen_narrowed_window_var_idxs_gap`'s docstring for why
    this region exposes the `var_base` gap (issue #305).
    """
    f = pgen_snp_ins_del_multi
    out = tmp_path / "ds"
    gvl.write(out, _NARROWED_PGEN_REGIONS, variants=str(f.pgen), overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        _NARROWED_PGEN_REGIONS, reference=f.fasta, variants=str(f.pgen)
    ).with_seqs("annotated")
    return ds, sds


def test_annotated_pgen_narrowed_window_haps_match(pgen_snp_ins_del_multi, tmp_path):
    """Sanity check (NOT xfail): haplotype reconstruction and ref_coords for
    a narrowed (non-whole-contig) PGEN window still match the written oracle
    exactly. Only the *global numbering* of `var_idxs` is affected by the
    `var_base` gap (see `test_annotated_pgen_narrowed_window_var_idxs_gap`)
    -- the underlying decode (which variants are applied, and where) is
    correct regardless of `var_base`, since `var_base` is only added as a
    final offset onto already-correctly-decoded local ids.
    """
    ds, sds = _narrowed_pgen_case(pgen_snp_ins_del_multi, tmp_path)

    n_cells = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # RaggedAnnotatedHaps
            for h in range(sds.ploidy):
                got_haps = np.asarray(data.haps[row][h])
                got_pos = np.asarray(data.ref_coords[row][h])
                exp_haps = np.asarray(exp.haps[h])
                exp_pos = np.asarray(exp.ref_coords[h])
                np.testing.assert_array_equal(got_haps, exp_haps)
                np.testing.assert_array_equal(got_pos, exp_pos)
            n_cells += 1
    assert n_cells > 0, "narrowed-window fixture yielded no cells"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#305: record-backend var_base for narrowed/partial-prefix windows "
        "deferred; PGEN currently emits local (var_base=0) var_idxs"
    ),
)
def test_annotated_pgen_narrowed_window_var_idxs_gap(pgen_snp_ins_del_multi, tmp_path):
    """Regression lock for the CONFIRMED bug found in review of Task 4: a
    NARROWED (not whole-contig) window on `pgen_snp_ins_del_multi` used to
    make `PgenWindowFiller::fill` set `slot.var_base` from `var_start`, the
    PADDED (over-inclusive) search lower bound used to construct
    `PgenRecordSource` -- NOT the global id of the first variant the window
    actually KEEPS, since genoray's precise extent-overlap filter can skip
    leading padded-in candidates. That silently undercounted every emitted
    global `var_idx` by the skipped-candidate count.

    `pgen_snp_ins_del_multi`'s variants (0-based global ids): 0=pos29 SNP,
    1=pos69 INS (extent [69,70)), 2=pos109 DEL (extent [109,113)), 3/4=pos149
    multiallelic-split SNPs. The region `[71, 200)` starts just past the
    INS's extent, so the padded search (padded by the contig's max ref_len,
    4, for the DEL) lands on local index 1 (pos69) as `var_start`, but the
    precise filter skips that candidate -- the window's first KEPT variant is
    pos109 (global id 2). The written oracle reports `var_idxs` `[2, 3, 4]`;
    the streaming PGEN backend (var_base=0, issue #277 Wave A / #305 gap)
    reports local ids `[0, 1, 2]` instead.

    This test is expected to XFAIL until #305 gives PGEN a correct per-contig
    global var_base (the first KEPT variant's global id, not the padded
    search lower bound). If it unexpectedly XPASSES, the gap has been fixed
    and this test (and its `xfail` marker) should be removed/updated.
    """
    ds, sds = _narrowed_pgen_case(pgen_snp_ins_del_multi, tmp_path)

    n_cells = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # RaggedAnnotatedHaps
            for h in range(sds.ploidy):
                got_vidx = np.asarray(data.var_idxs[row][h])
                exp_vidx = np.asarray(exp.var_idxs[h])
                np.testing.assert_array_equal(
                    got_vidx,
                    exp_vidx,
                    err_msg=(
                        f"row={row} hap={h}: streamed var_idxs must be "
                        "dataset-GLOBAL, matching the written oracle exactly"
                    ),
                )
            n_cells += 1
    assert n_cells > 0, "narrowed-window fixture yielded no cells"
