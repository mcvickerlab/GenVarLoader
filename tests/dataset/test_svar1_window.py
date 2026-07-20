"""Task 4: SVAR1 window read -> sparse arrays -> reconstruct haplotypes via the
existing kernel. Parity test: `_Svar1Backend.read_window` + `generate_batch`
(reading a live `.svar` store directly, no on-disk gvl dataset) must produce
byte-identical haplotypes to an independently-written+opened `gvl.Dataset`
over the same store/bed/reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import genvarloader as gvl
from genvarloader._dataset._streaming import _Svar1Backend


def _assert_streamed_matches_written(backend, written) -> None:
    """Drive one region's samples directly through `read_window` + `generate_batch`
    (a single whole-window call, `lo=0, hi=n_rows`) and assert every haplotype is
    byte-identical to the independently written `gvl.Dataset`.

    Haplotypes are ragged (indels change per-hap length), so a single `(r, s)`
    selection is a jagged `(ploidy, ~len)` Ragged that cannot be densified via
    `.to_numpy()`. Compare each haplotype's bytes individually -- the same
    byte-identical parity contract `test_svar2_dataset.py` checks, one hap at a
    time. `Ragged[h]` yields a 1-D dense `S1` array for hap `h`.

    The caller's `bed` is always a single-region frame, so the backend's sorted
    region index and the caller's (only) row both equal 0 -- no `_sort_order`
    translation is needed here (that lives in `StreamingDataset`, not
    `_Svar1Backend`).
    """
    r_idx = np.array([0], dtype=np.intp)
    s_idx = np.arange(backend.n_samples, dtype=np.intp)
    n_rows = len(r_idx) * len(s_idx)

    o_starts, o_stops = backend.read_window(r_idx, s_idx)
    data = backend.generate_batch(r_idx, s_idx, o_starts, o_stops, 0, n_rows, -1)
    assert len(data) == backend.n_samples

    for s in range(backend.n_samples):
        streamed = data[s]
        expected = written[0, s]
        for h in range(backend.ploidy):
            np.testing.assert_array_equal(
                np.asarray(streamed[h]), np.asarray(expected[h])
            )


def test_single_region_all_samples_matches_written(svar1_dataset_fixture):
    f = svar1_dataset_fixture
    backend = _Svar1Backend(f.svar_path, f.reference_path, f.contigs, f.bed)
    _assert_streamed_matches_written(backend, f.dataset.with_seqs("haplotypes"))


def test_subcontig_region_exercises_window_filter(svar1_dataset_fixture, tmp_path):
    """A sub-contig region that starts *after* the first variant, so
    `read_window`'s position filter (`pos < lo`) must actually exclude a variant.

    The fixture's variants are at 0-based positions 2 (SNP), 6 (INS), 9 (SNP),
    11 (DEL). A `[3, 20)` window drops the SNP at 2 (2 < 3) while keeping the
    insertion, SNP, and deletion -- so an off-by-one in the half-open bound or in
    the 0-basedness of the stored `pos` would break parity here even though the
    full-contig `[0, 40)` test (which admits every variant regardless) would not.
    """
    from genoray import SparseVar

    f = svar1_dataset_fixture
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [3], "chromEnd": [20]})

    out = tmp_path / "d_subcontig.gvl"
    gvl.write(out, bed, variants=SparseVar(f.svar_path), samples=None, overwrite=True)
    written = gvl.Dataset.open(out, reference=f.reference_path).with_seqs("haplotypes")

    backend = _Svar1Backend(f.svar_path, f.reference_path, f.contigs, bed)
    _assert_streamed_matches_written(backend, written)
