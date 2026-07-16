"""Task 4: SVAR1 window read -> sparse arrays -> reconstruct haplotypes via the
existing kernel. Parity test: `_Svar1Backend.reconstruct_window` (reading a
live `.svar` store directly, no on-disk gvl dataset) must produce
byte-identical haplotypes to an independently-written+opened `gvl.Dataset`
over the same store/bed/reference.
"""

from __future__ import annotations

import numpy as np

from genvarloader._dataset._streaming import StreamingDataset, _Svar1Backend


def test_single_region_all_samples_matches_written(svar1_dataset_fixture):
    f = svar1_dataset_fixture
    backend = _Svar1Backend(f.svar_path, f.reference_path, f.contigs, f.bed)
    sds = StreamingDataset(
        f.bed,
        contigs=f.contigs,
        n_samples=backend.n_samples,
        ploidy=backend.ploidy,
        _reconstruct_window=backend.reconstruct_window,
    )._with_batch_size(backend.n_samples)

    batches = list(sds)  # one region, all samples in one batch
    assert len(batches) == 1
    data, r_idx, s_idx = batches[0]
    assert len(r_idx) == backend.n_samples

    written = f.dataset.with_seqs("haplotypes")
    # Haplotypes are ragged (indels change per-hap length), so a single (r, s)
    # selection is a jagged `(ploidy, ~len)` Ragged that cannot be densified via
    # `.to_numpy()`. Compare each haplotype's bytes individually -- the same
    # byte-identical parity contract `test_svar2_dataset.py` checks, one hap at a
    # time. `Ragged[h]` yields a 1-D dense `S1` array for hap `h`.
    for i, (r, s) in enumerate(zip(r_idx, s_idx)):
        streamed = data[i]
        expected = written[int(r), int(s)]
        for h in range(backend.ploidy):
            np.testing.assert_array_equal(
                np.asarray(streamed[h]), np.asarray(expected[h])
            )
