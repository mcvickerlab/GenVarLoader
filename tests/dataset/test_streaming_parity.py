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

from pathlib import Path

import numpy as np
import pytest

import genvarloader as gvl


@pytest.mark.parametrize("prefetch_strategy", ["engine", "readahead"])
def test_streaming_matches_written_all_cells(
    svar1_multicontig_fixture, prefetch_strategy
):
    """Byte-identical parity must hold under BOTH prefetch strategies (issue #283):
    Design C ("readahead") reuses the exact same `read_window`/`generate_batch` calls
    Design A ("engine") does, so prefetch is a pure page-warming no-op on output --
    same assertions, same fixture, only the internal `_prefetch_strategy` toggle
    differs."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    object.__setattr__(sds, "_prefetch_strategy", prefetch_strategy)
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


def test_backend_read_then_generate_matches_whole_window(svar1_multicontig_fixture):
    """read_window + per-batch generate must reproduce the same bytes a single
    whole-window reconstruction would, for every batch slice."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )

    for data, r_idx, s_idx in sds.to_iter(batch_size=3):
        for i in range(len(r_idx)):
            expected = written[int(r_idx[i]), int(s_idx[i])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][h]), np.asarray(expected[h])
                )


def test_dataloader_len_matches_batches_yielded(svar1_multicontig_fixture):
    """`len(dl)` must count window-batched batches, not `ceil(cells/batch_size)`.

    `_plan` batches *within* each window, so every window's last batch may be
    partial. Here each contig run (6 regions x 3 samples = 18 cells) fits in a
    single window (`_window_regions=64` >> 6 regions), so a window coincides with
    a contig run: with 2 contigs and batch_size=4, the naive `ceil(36/4)=9`
    under-reports the 2*ceil(18/4)=10 batches actually yielded. `DataLoader.__len__`
    forwards to the wrapper, so anything trusting `len(dl)` (progress bars, epoch
    accounting) would be wrong.
    """
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    dl = sds.to_dataloader(batch_size=4, return_indices=True)

    assert len(dl) == sum(1 for _ in dl)


def test_no_map_style_access(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)

    # to_torch_dataset() no longer raises -- it returns an IterableDataset wrapping
    # to_iter(). Only random access is refused.
    with pytest.raises(TypeError):
        _ = sds[0, 0]


def test_to_iter_is_the_one_entry_point(svar1_multicontig_fixture):
    """`to_iter` is the single iteration API. `__iter__` is REMOVED -- one and only
    one obvious way to do it; `to_torch_dataset`/`to_dataloader` wrap `to_iter`."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")

    assert not hasattr(sds, "__iter__"), "__iter__ must be removed; use to_iter()"
    with pytest.raises(TypeError):
        next(iter(sds))

    batches = list(sds.to_iter(batch_size=4))
    assert len(batches) > 0
    data, r_idx, s_idx = batches[0]
    assert len(r_idx) == len(s_idx)


def test_to_iter_return_indices_false_yields_data_only(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    first = next(iter(sds.to_iter(batch_size=2, return_indices=False)))
    assert not isinstance(first, tuple), "return_indices=False must yield data alone"


def test_to_torch_dataset_wraps_to_iter(svar1_multicontig_fixture):
    """`to_torch_dataset` now RETURNS an IterableDataset (it used to raise TypeError,
    because StreamingDataset itself was one). Same name as Dataset.to_torch_dataset."""
    import torch.utils.data as td

    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    tds = sds.to_torch_dataset(batch_size=4)
    assert isinstance(tds, td.IterableDataset)
    assert len(list(tds)) == len(list(sds.to_iter(batch_size=4)))


def test_to_iter_covers_every_cell_exactly_once(svar1_multicontig_fixture):
    """Window/batch separation must not drop or duplicate cells."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    seen = []
    for _data, r_idx, s_idx in sds.to_iter(batch_size=3):
        seen.extend(zip(r_idx.tolist(), s_idx.tolist()))
    n_regions, n_samples = sds.shape
    assert sorted(seen) == sorted(
        (r, s) for r in range(n_regions) for s in range(n_samples)
    )


def test_streamingdataset_is_public_and_documented():
    """Task 6 gate: `StreamingDataset` must be exported from the top-level
    `genvarloader` package and documented in `docs/source/api.md` -- CLAUDE.md's
    "api.md must stay in sync with `__all__`" rule."""
    assert "StreamingDataset" in gvl.__all__
    assert hasattr(gvl, "StreamingDataset")
    # tests/dataset/test_streaming_parity.py -> repo root is 2 parents up.
    api_md = Path(__file__).parents[2] / "docs" / "source" / "api.md"
    assert "StreamingDataset" in api_md.read_text()


def test_streaming_handles_mixed_contig_naming_style(svar1_mixed_naming_fixture):
    """Regression test (final-review Finding 1): `_Svar1Backend.generate_batch`
    used to look up `self._ref.c_map.contigs.index(contig_name)`, where
    `contig_name` is the STORE's (unnormalized) contig name. But `Reference.from_path`
    normalizes its `c_map` to the FASTA's own naming style (UCSC `chr1` vs Ensembl
    `1`), so pairing a UCSC-style `.svar` store with an Ensembl-style FASTA (or vice
    versa) made `contig_name` absent from `c_map.contigs`, raising a bare
    `ValueError: 'chr1' is not in list` with no context -- even though
    `Reference.from_path` explicitly documents this pairing as supported, and the
    written path (`gvl.write` + `Dataset.open`) already handles it correctly.

    `svar1_mixed_naming_fixture` pairs a UCSC-style `.svar` store (`chr1`) with an
    Ensembl-style FASTA (`1`, no prefix) to reproduce the pairing; the comparison
    `gvl.Dataset` is opened with a same-sequence UCSC-style FASTA (`chr1`) to prove
    streamed output is still byte-identical despite the naming mismatch.
    """
    f = svar1_mixed_naming_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_ensembl_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(
        f.dataset_path, reference=f.reference_chr_path
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
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


def test_samples_property_matches_written_dataset(svar1_multicontig_fixture):
    """`StreamingDataset.samples` must mirror `Dataset.samples` -- lexicographically
    sorted sample names, the same order `gvl.write()` uses and the same order
    `sample_idx` indexes into."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)

    assert sds.samples == written.samples
    assert sds.samples == sorted(sds.samples), (
        "samples must be lexicographically sorted"
    )


def test_samples_property_indexes_actual_sample_identity(svar1_sample_order_fixture):
    """Non-tautological check for Finding 2: `sds.samples[i]` must genuinely be the
    NAME of the sample whose data arrives at `sample_idx == i` -- not merely "a
    sorted list of names" (which would pass even if `samples` and `sample_idx`
    disagreed on order, as long as `samples` itself happened to be sorted).

    `svar1_sample_order_fixture` declares samples in native VCF column order
    (S10, S2, S1) -- NOT lexicographic order -- with a diagnostic SNP that makes S2
    distinguishable from S1/S10 by reconstructed sequence content alone. If
    `sample_idx` and `samples` ever disagreed (e.g. one used native order, the
    other sorted order), this test would catch it by reading the wrong base at the
    diagnostic position for at least one sample.
    """
    f = svar1_sample_order_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")

    assert sorted(f.expected_base_by_sorted_name) == sds.samples

    seen_samples = set()
    for data, _r_idx, s_idx in sds.to_iter(batch_size=1):
        for i in range(len(s_idx)):
            s = int(s_idx[i])
            name = sds.samples[s]
            base = bytes(np.asarray(data[i][0])[f.diagnostic_pos])
            assert base == f.expected_base_by_sorted_name[name], (
                f"sds.samples[{s}] = {name!r}, but the haplotype actually "
                f"delivered at sample_idx={s} has base {base!r} at the "
                f"diagnostic position, not {f.expected_base_by_sorted_name[name]!r} "
                f"as expected for {name!r}"
            )
            seen_samples.add(name)

    assert seen_samples == set(sds.samples)
