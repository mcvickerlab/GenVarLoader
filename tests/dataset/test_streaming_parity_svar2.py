"""Byte-identical parity: StreamingDataset over a .svar2 store vs a written gvl.Dataset."""

from __future__ import annotations

import numpy as np

import genvarloader as gvl


def test_streaming_svar2_matches_written_all_cells(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(fx.dataset_path, reference=fx.reference_path).with_seqs(
        "haplotypes"
    )

    seen = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp = written[r, s]
            for p in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][p]),
                    np.asarray(exp[p]),
                    err_msg=f"mismatch at region={r} sample={s} ploid={p}",
                )
            seen += 1
    assert seen == fx.bed.height * sds.n_samples


def test_streaming_svar2_covers_every_cell_once(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    cells = set()
    for _, r_idx, s_idx in sds.to_iter(batch_size=3, return_indices=True):
        for i in range(len(r_idx)):
            cells.add((int(r_idx[i]), int(s_idx[i])))
    assert cells == {(r, s) for r in range(fx.bed.height) for s in range(sds.n_samples)}
