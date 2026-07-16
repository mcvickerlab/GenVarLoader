"""Task 5: public `StreamingDataset` API parity across many regions/contigs,
driven through a real PyTorch `DataLoader` (`to_dataloader`).

`test_streaming_matches_written_all_cells` is the multi-contig ordering test:
it uses a genuinely unsorted, interleaved-contig bed
(`svar1_multicontig_fixture`) so that streamed `(r_idx, s_idx)` indices must
be correctly translated back to the user's original bed-row order to match
`gvl.Dataset.open(...)[r, s]` -- see `.superpowers/sdd/task-5-context.md`
item 1 for the write/open-side proof that `ds[r, s]` uses original-bed-row
order, not sorted-storage order.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


def test_streaming_matches_written_all_cells(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )

    dl = sds.to_dataloader(batch_size=4, return_indices=True)

    seen = set()
    for data, r_idx, s_idx in dl:
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            streamed = data[i]
            expected = written[r, s]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(streamed[h]), np.asarray(expected[h])
                )
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_no_map_style_access(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)

    with pytest.raises(TypeError):
        sds.to_torch_dataset()
    with pytest.raises(TypeError):
        _ = sds[0, 0]
