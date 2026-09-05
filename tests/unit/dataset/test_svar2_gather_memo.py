"""Unit tests for ``Svar2Haps._gathered_groups``' one-batch memo (issue #349).

The memo exists because every SVAR2 read gathers the same read-bound FFI inputs
twice per batch -- once to size the output, once to fill it. These tests pin the
two properties that make it safe to reuse: it hits on an equal-valued *copy* of
the batch key (so the sizing and reconstruct passes share one gather), and it
misses whenever the key's contents change -- including when the caller refills
its own buffer in place, which an identity-keyed memo would silently miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._svar2_haps import Svar2Haps


class _CountingGather:
    """Stand-in for the slice of ``Svar2Haps`` that ``_gathered_groups`` uses.

    Binding the real methods onto a plain object keeps the test on the memo's
    logic; the gather itself needs a memmapped ``.svar2`` cache and is covered
    end-to-end by ``tests/dataset/test_svar2_dataset.py``.
    """

    def __init__(self, contigs: np.ndarray, n_regions: int, n_samples: int):
        self._contigs = contigs
        self.calls = 0
        self._gather_memo = None
        self.genotypes = type("_G", (), {"shape": (n_regions, n_samples, 2)})()

    _gathered_groups = Svar2Haps._gathered_groups
    _contig_groups = Svar2Haps._contig_groups

    def _gather_inputs(self, r_q, si_q, regions_grp, P):
        self.calls += 1
        return (np.asarray(r_q), np.asarray(si_q), np.asarray(regions_grp), P)


def _regions(contigs: list[int]) -> np.ndarray:
    n = len(contigs)
    out = np.zeros((n, 4), np.int32)
    out[:, 0] = contigs
    out[:, 1] = np.arange(n, dtype=np.int32) * 10
    out[:, 2] = out[:, 1] + 8
    return out


def test_memo_hits_on_an_equal_valued_copy():
    """The sizing pass and the reconstruct pass must share one gather.

    They are separate calls carrying separate-but-equal arrays (the splice-plan
    builder re-derives ``regions``), so an identity key would miss.
    """
    regions = _regions([0, 0, 0, 0])
    idx = np.arange(4, dtype=np.intp)
    h = _CountingGather(regions[:, 0], n_regions=4, n_samples=1)

    first = h._gathered_groups(idx, regions, 2)
    second = h._gathered_groups(idx.copy(), regions.copy(), 2)

    assert h.calls == 1  # one contig group, gathered once
    assert second is first


def test_memo_splits_per_contig_group():
    regions = _regions([0, 1, 0, 2])
    idx = np.arange(4, dtype=np.intp)
    h = _CountingGather(regions[:, 0], n_regions=4, n_samples=1)

    groups = h._gathered_groups(idx, regions, 2)
    h._gathered_groups(idx.copy(), regions.copy(), 2)

    assert [ci for ci, _, _ in groups] == [0, 1, 2]
    assert h.calls == 3  # once per contig, not once per pass


@pytest.mark.parametrize("field", ["idx", "regions", "ploidy"])
def test_memo_misses_when_the_batch_changes(field):
    regions = _regions([0, 0, 0, 0])
    idx = np.arange(4, dtype=np.intp)
    h = _CountingGather(regions[:, 0], n_regions=8, n_samples=1)

    h._gathered_groups(idx, regions, 2)
    if field == "idx":
        h._gathered_groups(idx + 4, regions, 2)
    elif field == "regions":
        h._gathered_groups(idx, _regions([0, 0, 0, 0]) + 1, 2)
    else:
        h._gathered_groups(idx, regions, 1)

    assert h.calls == 2


def test_memo_misses_when_the_caller_refills_a_buffer_in_place():
    """The memo keys on a private copy, so in-place reuse can't serve stale rows."""
    regions = _regions([0, 0, 0, 0])
    idx = np.arange(4, dtype=np.intp)
    h = _CountingGather(regions[:, 0], n_regions=8, n_samples=1)

    h._gathered_groups(idx, regions, 2)
    idx += 4  # same object, new batch
    groups = h._gathered_groups(idx, regions, 2)

    assert h.calls == 2
    assert np.array_equal(groups[0][2][0], idx)
