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
