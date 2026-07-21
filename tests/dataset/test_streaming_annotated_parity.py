"""Task 4 (issue #277, Wave A): annotated output (``AnnotatedHaps``) must be
byte-identical to the written oracle for all three streaming backends
(SVAR1, VCF, PGEN) at jitter=0, including the emitted ``var_idxs`` -- which
must be dataset-GLOBAL variant ids (see the streaming engines' per-variant
global-id gather: genoray's ``DenseChunk.global_idx`` is copied verbatim into
``DecodedWindow.global_v_idxs`` by ``fill_decoded_window``, then gathered by
``generate_batch_core`` -- SVAR1 for free (already global), PGEN via the
``.pvar`` row index (correct, issue #305 Phase 2), VCF still window-local
until Phase 3 -- see ``_VcfBackend.build_engine``'s doc comment and GitHub
issue #305 for the documented gap).

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

# VCF is intentionally excluded: `with_seqs("annotated")` is a fail-fast
# NotImplementedError for the VCF backend (its `var_idxs` are dataset-global ids
# a VCF source cannot produce cheaply; issues #305, #311). See
# `test_vcf_annotated_fails_fast` for the guard's coverage. SVAR1 (ids for free)
# and PGEN (ids from the `.pvar` row index) exercise the local->global gather.
BACKENDS = ["svar1", "pgen"]


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


def test_vcf_annotated_fails_fast(streaming_case):
    """`with_seqs("annotated")` on a VCF-backed `StreamingDataset` must raise
    `NotImplementedError` at materialization rather than emit silently-wrong
    dataset-global `var_idxs` (issues #305, #311). VCF sources have no cheap
    per-record global id; annotated output requires a PGEN or SVAR source.
    """
    regions, reference, variants, _written = streaming_case("vcf")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("annotated")
    with pytest.raises(NotImplementedError, match="annotated.*VCF backend"):
        # Materialization (and thus the guard) fires when iteration begins.
        next(iter(sds.to_iter(batch_size=4)))


_NARROWED_PGEN_REGIONS = pl.DataFrame(
    {"chrom": ["chr1"], "chromStart": [71], "chromEnd": [200]}
)


def _narrowed_pgen_case(pgen_snp_ins_del_multi, tmp_path):
    """Shared setup for the narrowed-window `pgen_snp_ins_del_multi` tests
    below: writes the oracle dataset and constructs a matching
    `StreamingDataset`, both restricted to region `[71, 200)`. See
    `test_annotated_pgen_narrowed_window_var_idxs_gap`'s docstring for why
    this region exercises the non-contiguous global `var_idxs` set that the
    per-variant global-id gather now handles correctly (issue #305).
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
    exactly. Only the *global numbering* of `var_idxs` was ever affected by
    the #305 gap (see `test_annotated_pgen_narrowed_window_var_idxs_gap`)
    -- the underlying decode (which variants are applied, and where) is
    correct regardless of id numbering, since the global id is only applied
    as a final per-variant gather onto already-correctly-decoded local ids.
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
    pos109 (global id 2). The written oracle reports `var_idxs` `[2, 3, 4]`.

    This is now a PASSING REGRESSION LOCK: #305 gives PGEN correct per-variant
    global ids via genoray's `.pvar` row index (the source of truth for each
    variant's dataset-global id) plus gvl's per-variant gather onto that id
    (retiring the scalar `var_base`, which could only ever encode a single
    offset and thus couldn't represent a skipped leading candidate). The
    narrowed region `[71, 200)` drops leading candidates 0 and 1, so the
    first KEPT variant is global id 2, and the streamed `var_idxs` must equal
    the written oracle's `[2, 3, 4]`.
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


def test_annotated_pgen_interior_exclusion_var_idxs_gapped(
    pgen_interior_exclusion, tmp_path
):
    """Regression lock for a HOLE in the middle of the global variant-id set
    (issue #305), complementing `test_annotated_pgen_narrowed_window_var_idxs_gap`
    above: that test's `[2, 3, 4]` is a leading-gap-only (still contiguous)
    set; this test exercises a genuinely NON-CONTIGUOUS kept set with a gap
    in the interior.

    `pgen_interior_exclusion` (see its docstring/comment in ``conftest.py``)
    is a spanning DEL (global id 0) whose deleted span [20, 26) consumes an
    interior SNP's reference position (global id 1, at 0-based pos 22) --
    genotyped 1|0 (carried on hap0) to confirm empirically that the exclusion
    is genuine interior consumption, not merely "genotype absent": no byte is
    ever emitted at a deleted reference position regardless of what any
    variant record at that position's genotype says. A trailing SNP (global
    id 2, at 0-based pos 28) sits beyond the DEL's extent. Over the query
    region [20, 30), hap0 of sample s0 carries the DEL and the trailing SNP
    but never the interior SNP -- confirmed by inspecting the WRITTEN oracle
    directly: `var_idxs` is `[0, -1, -1, 2, -1]`, i.e. kept global ids
    `{0, 2}` with `np.diff([0, 2]) == [2] > 1`, a genuine gap. The streaming
    PGEN backend must reproduce this exact non-contiguous set, not just a
    uniformly-shifted contiguous one -- this is the case a scalar `var_base`
    offset cannot represent, since offsetting local index `i` by a constant
    can never skip a candidate that falls strictly between two kept ones.
    """
    f = pgen_interior_exclusion
    out = tmp_path / "ds"
    gvl.write(out, f.regions, variants=str(f.pgen), overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        f.regions, reference=f.fasta, variants=str(f.pgen)
    ).with_seqs("annotated")

    n_cells = 0
    saw_gap = False
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
                kept = exp_vidx[exp_vidx >= 0]
                if kept.size >= 2 and np.any(np.diff(kept) > 1):
                    saw_gap = True
            n_cells += 1
    assert n_cells > 0, "interior-exclusion fixture yielded no cells"
    assert saw_gap, (
        "fixture did not produce a non-contiguous kept var_idxs set for any "
        "hap; the test can't guard the interior-exclusion gap without one"
    )
