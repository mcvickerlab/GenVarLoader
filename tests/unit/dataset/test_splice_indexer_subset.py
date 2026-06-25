"""Regression test for SpliceIndexer.parse_idx double-applying the sample-subset map.

Bug: when a Dataset has been sample-subsetted (non-identity ``sample_subset_idxs``)
AND output is spliced, ``SpliceIndexer.parse_idx`` applied the sample-subset map
twice:

1. Once inside the body of ``parse_idx`` via ``self.dsi._s_idx[s_idx]``.
2. Again inside the ``self.dsi.parse_idx((r_idx, s_idx))`` call at the end,
   which applies ``_s_idx`` a second time.

Concrete symptom: requesting sample at output position *p* fetches the data for
``full_sample_idxs[sample_subset_idxs[full_sample_idxs[sample_subset_idxs[p]]]]``
instead of the correct ``full_sample_idxs[sample_subset_idxs[p]]``.

With the MMRF consensus dataset, MMRF_2702 (svar pos 54, GVL sorted 625) received
MMRF_1395's NRAS G12D mutation because ``sample_subset_idxs[625]`` mapped to
MMRF_1395.

This test builds a tiny ``SpliceIndexer`` with a non-identity sample subset and
verifies that ``parse_idx`` resolves each requested (row, sample) pair to the
correct flat storage index — i.e. that the sample-subset map is applied exactly
once, and that each (row, sample) pair resolves to the *correct* sample's storage
position.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from hirola import HashTable
from seqpro.rag import Ragged

from genvarloader._dataset._indexing import DatasetIndexer, SpliceIndexer
from genvarloader._dataset._splice import SpliceMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ragged(exon_lists: list[list[int]]) -> Ragged:
    """Build a seqpro Ragged from a list-of-lists of integer region indices."""
    return Ragged(ak.Array(exon_lists, check_valid=True))


def _make_splice_indexer(
    *,
    n_full_regions: int,
    n_full_samples: int,
    # Which on-disk sample positions are in the subset (output pos -> on-disk pos)
    sample_subset_idxs: list[int],
    # Ragged list of per-transcript exon region indices (0-indexed into regions)
    exon_lists: list[list[int]],
    # On-disk region → storage row (identity here for simplicity)
    region_storage_idxs: list[int] | None = None,
    # On-disk sample → storage col (identity here for simplicity)
    sample_storage_idxs: list[int] | None = None,
) -> SpliceIndexer:
    """Construct a ``SpliceIndexer`` with explicit on-disk layouts and a
    non-trivial sample subset."""

    if region_storage_idxs is None:
        region_storage_idxs = list(range(n_full_regions))
    if sample_storage_idxs is None:
        sample_storage_idxs = list(range(n_full_samples))

    full_r = np.array(region_storage_idxs, dtype=np.intp)
    full_s = np.array(sample_storage_idxs, dtype=np.intp)
    sample_names = [f"s{i}" for i in range(n_full_samples)]
    _samples = np.array(sample_names, dtype=np.str_)
    s2i_map = HashTable(max=len(_samples) * 2, dtype=_samples.dtype)
    s2i_map.add(_samples)

    dsi = DatasetIndexer(
        full_region_idxs=full_r,
        full_sample_idxs=full_s,
        s2i_map=s2i_map,
        r2i_map=None,
        region_subset_idxs=None,
        sample_subset_idxs=np.array(sample_subset_idxs, dtype=np.intp),
    )

    # Build the SpliceMap from exon_lists.
    n_transcripts = len(exon_lists)
    transcript_names = np.array([f"T{i}" for i in range(n_transcripts)], dtype=np.str_)
    transcript_name_map = HashTable(max=n_transcripts * 2, dtype=transcript_names.dtype)
    transcript_name_map.add(transcript_names)

    splice_map_ragged = _make_ragged(exon_lists)

    sm = SpliceMap(
        names=transcript_name_map,
        splice_map=splice_map_ragged,
        full_splice_map=splice_map_ragged,
        row_idxs=np.arange(n_transcripts, dtype=np.intp),
        row_subset_idxs=None,
    )

    return SpliceIndexer(map=sm, dsi=dsi)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def splice_idxer_with_subset():
    """A SpliceIndexer with:

    * 4 full regions (region positions 0-3 map to storage rows 0-3, identity)
    * 5 full samples (sample positions 0-4 map to storage cols 0-4, identity)
    * sample subset: output positions [0, 1] map to on-disk positions [2, 4]
      (i.e. sample_subset_idxs = [2, 4])
    * 2 transcripts:
        T0 = exons at region positions [0, 1]
        T1 = exons at region positions [2, 3]
    """
    return _make_splice_indexer(
        n_full_regions=4,
        n_full_samples=5,
        sample_subset_idxs=[2, 4],
        exon_lists=[[0, 1], [2, 3]],
    )


# ---------------------------------------------------------------------------
# Core regression test — sample-subset map applied exactly once
# ---------------------------------------------------------------------------


def test_parse_idx_sample_subset_applied_once(splice_idxer_with_subset):
    """SpliceIndexer.parse_idx must apply the sample-subset map exactly once.

    When user requests all transcripts × all subset-samples (slice, slice), each
    (transcript, sample) pair should resolve to the correct flat storage index:

        storage_idx = ravel_multi_index(
            (full_region_idxs[exon_region_pos], full_sample_idxs[sample_subset_idxs[output_pos]]),
            full_shape,
        )

    With the bug, the second call to ``dsi.parse_idx`` re-applies _s_idx,
    turning ``full_sample_idxs[sample_subset_idxs[output_pos]]`` into
    ``full_sample_idxs[sample_subset_idxs[full_sample_idxs[sample_subset_idxs[output_pos]]]]``
    — fetching the wrong sample entirely (or causing an IndexError if the
    double-mapped value is out of bounds for the 2-element subset).
    """
    si = splice_idxer_with_subset

    # full_shape = (n_regions=4, n_full_samples=5)
    assert si.dsi.full_shape == (4, 5)
    # shape = (n_transcripts=2, n_subset_samples=2)
    assert si.shape == (2, 2)

    # Request all transcripts, all subset samples.
    ds_idx, squeeze, out_reshape, offsets, n_rows, n_samples = si.parse_idx(
        (slice(None), slice(None))
    )

    # Decode the returned flat storage indices.
    n_full_regions = 4
    n_full_samples = 5
    full_shape = (n_full_regions, n_full_samples)

    r_flat, s_flat = np.unravel_index(ds_idx, full_shape)

    # Every returned region index must be in {0, 1, 2, 3}.
    assert set(r_flat.tolist()).issubset({0, 1, 2, 3}), (
        f"Unexpected region storage indices: {r_flat.tolist()!r}"
    )

    # KEY ASSERTION: every returned on-disk sample index must be 2 or 4.
    # sample_subset_idxs = [2, 4] maps output pos 0 → on-disk 2, pos 1 → on-disk 4.
    # With the double-application bug, the code re-applies _s_idx (which is
    # [2, 4]) to the already-mapped values 2 and 4:
    #   _s_idx[2] is out-of-range for a 2-element array → IndexError or wrap.
    # Even if it doesn't crash, the on-disk indices would be wrong.
    unexpected_samples = set(s_flat.tolist()) - {2, 4}
    assert not unexpected_samples, (
        f"SpliceIndexer.parse_idx returned wrong on-disk sample indices {set(s_flat.tolist())}; "
        f"expected only {{2, 4}} (subset_idxs=[2,4] maps output positions [0,1] → on-disk [2,4]). "
        f"Got: {s_flat.tolist()!r}.  "
        f"This indicates the sample-subset map was applied more than once."
    )

    # Also check that BOTH on-disk samples appear (not just one).
    assert 2 in s_flat, "On-disk sample 2 missing from output"
    assert 4 in s_flat, "On-disk sample 4 missing from output"

    # Region correctness: T0 exons {0,1} and T1 exons {2,3} must appear.
    assert 0 in r_flat and 1 in r_flat, "T0 exon regions 0 and 1 must both appear"
    assert 2 in r_flat and 3 in r_flat, "T1 exon regions 2 and 3 must both appear"


def test_parse_idx_single_sample_from_subset(splice_idxer_with_subset):
    """Selecting one subset sample by output index returns that sample's storage index."""
    si = splice_idxer_with_subset

    # Request transcript T0 (idx 0), output sample 1 (→ on-disk sample 4).
    ds_idx, squeeze, out_reshape, offsets, n_rows, n_samples = si.parse_idx((0, 1))

    n_full_regions = 4
    n_full_samples = 5
    full_shape = (n_full_regions, n_full_samples)

    r_flat, s_flat = np.unravel_index(ds_idx, full_shape)

    # All returned on-disk sample indices must be 4 (output pos 1 → on-disk 4).
    assert (s_flat == 4).all(), (
        f"Expected all on-disk sample indices to be 4, got {s_flat.tolist()!r}.  "
        f"Double-applying the subset map would produce a wrong value or IndexError."
    )
    # T0 has exons at region positions 0 and 1.
    assert set(r_flat.tolist()) == {0, 1}, (
        f"Expected region indices {{0, 1}} for T0, got {r_flat.tolist()!r}"
    )


def test_parse_idx_no_subset_unaffected():
    """Without a sample subset the fix must not change existing behaviour.

    Uses an identity-subset (sample_subset_idxs = None effectively, achieved by
    constructing the DSI without subset) to confirm the non-subset path still works.
    """
    # Build a DSI *without* any sample_subset_idxs (the None path).
    n_full_regions = 3
    n_full_samples = 3
    full_r = np.arange(n_full_regions, dtype=np.intp)
    full_s = np.arange(n_full_samples, dtype=np.intp)
    sample_names = [f"s{i}" for i in range(n_full_samples)]
    _samples = np.array(sample_names, dtype=np.str_)
    s2i_map = HashTable(max=len(_samples) * 2, dtype=_samples.dtype)
    s2i_map.add(_samples)

    dsi = DatasetIndexer(
        full_region_idxs=full_r,
        full_sample_idxs=full_s,
        s2i_map=s2i_map,
        r2i_map=None,
        region_subset_idxs=None,
        sample_subset_idxs=None,  # no subset
    )

    exon_lists = [[0, 1], [1, 2]]
    n_transcripts = 2
    transcript_names = np.array([f"T{i}" for i in range(n_transcripts)], dtype=np.str_)
    transcript_name_map = HashTable(max=n_transcripts * 2, dtype=transcript_names.dtype)
    transcript_name_map.add(transcript_names)

    splice_map_ragged = _make_ragged(exon_lists)
    sm = SpliceMap(
        names=transcript_name_map,
        splice_map=splice_map_ragged,
        full_splice_map=splice_map_ragged,
        row_idxs=np.arange(n_transcripts, dtype=np.intp),
        row_subset_idxs=None,
    )
    si = SpliceIndexer(map=sm, dsi=dsi)

    ds_idx, squeeze, out_reshape, offsets, n_rows, n_samples = si.parse_idx(
        (slice(None), slice(None))
    )

    full_shape = (n_full_regions, n_full_samples)
    r_flat, s_flat = np.unravel_index(ds_idx, full_shape)

    # All samples must appear and only valid on-disk samples (0, 1, 2).
    unexpected = set(s_flat.tolist()) - {0, 1, 2}
    assert not unexpected, f"Unexpected on-disk sample indices: {unexpected}"
    # T0 exons {0,1}, T1 exons {1,2}
    assert set(r_flat.tolist()).issubset({0, 1, 2}), (
        f"Unexpected region indices: {r_flat.tolist()!r}"
    )
