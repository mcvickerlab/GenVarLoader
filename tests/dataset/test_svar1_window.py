"""Task 4: SVAR1 window read -> sparse arrays -> reconstruct haplotypes via the
existing kernel. Parity test: `_Svar1Backend.reconstruct_window` (reading a
live `.svar` store directly, no on-disk gvl dataset) must produce
byte-identical haplotypes to an independently-written+opened `gvl.Dataset`
over the same store/bed/reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import genvarloader as gvl
from genvarloader._dataset._streaming import StreamingDataset, _Svar1Backend


def _assert_streamed_matches_written(backend, bed, contigs, written) -> None:
    """Drive one region's samples through the streaming backend and assert every
    haplotype is byte-identical to the independently written `gvl.Dataset`.

    Haplotypes are ragged (indels change per-hap length), so a single `(r, s)`
    selection is a jagged `(ploidy, ~len)` Ragged that cannot be densified via
    `.to_numpy()`. Compare each haplotype's bytes individually -- the same
    byte-identical parity contract `test_svar2_dataset.py` checks, one hap at a
    time. `Ragged[h]` yields a 1-D dense `S1` array for hap `h`.
    """
    sds = StreamingDataset(
        bed,
        contigs=contigs,
        n_samples=backend.n_samples,
        ploidy=backend.ploidy,
        _reconstruct_window=backend.reconstruct_window,
    )

    batches = list(
        sds.to_iter(batch_size=backend.n_samples)
    )  # one region, all samples in one batch
    assert len(batches) == 1
    data, r_idx, s_idx = batches[0]
    assert len(r_idx) == backend.n_samples

    for i, (r, s) in enumerate(zip(r_idx, s_idx)):
        streamed = data[i]
        expected = written[int(r), int(s)]
        for h in range(backend.ploidy):
            np.testing.assert_array_equal(
                np.asarray(streamed[h]), np.asarray(expected[h])
            )


def test_single_region_all_samples_matches_written(svar1_dataset_fixture):
    f = svar1_dataset_fixture
    backend = _Svar1Backend(f.svar_path, f.reference_path, f.contigs, f.bed)
    _assert_streamed_matches_written(
        backend, f.bed, f.contigs, f.dataset.with_seqs("haplotypes")
    )


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
    _assert_streamed_matches_written(backend, bed, f.contigs, written)
