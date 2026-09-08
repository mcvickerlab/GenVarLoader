"""Tests for :class:`genvarloader.Reference` construction via ``from_path``.

Focus: the ``in_memory=False`` (memory-mapped) path cannot reorder or subset
contigs, because the mmap stays in full-FASTA order while ``offsets`` is a single
cumulative array. Regression coverage for issue #285, where the combination was
silently accepted and returned the wrong contig's bytes.
"""

import numpy as np
import pytest

import genvarloader as gvl

# synthetic.fa.bgz fixture (``ref_fasta``) has contigs chr1, chr2 in that order.
FULL_ORDER = ["chr1", "chr2"]


@pytest.mark.parametrize(
    "contigs",
    [
        ["chr2", "chr1"],  # reorder
        ["chr1"],  # strict subset (same relative order, still ambiguous vs mmap)
        ["chr2"],  # subset that isn't a prefix
    ],
)
def test_from_path_mmap_rejects_reorder_or_subset(ref_fasta, contigs):
    """``in_memory=False`` with a reordered/subset ``contigs`` must fail fast."""
    with pytest.raises(ValueError, match="in_memory=False"):
        gvl.Reference.from_path(ref_fasta, contigs, in_memory=False)


@pytest.mark.parametrize("contigs", [None, FULL_ORDER])
def test_from_path_mmap_allows_full_order(ref_fasta, contigs):
    """``in_memory=False`` with ``None`` or the full FASTA order stays valid."""
    ref = gvl.Reference.from_path(ref_fasta, contigs, in_memory=False)
    assert ref.contigs == FULL_ORDER
    # offsets are the full cumulative FASTA layout: (n_contigs + 1,)
    assert ref.offsets.shape == (len(FULL_ORDER) + 1,)


def test_from_path_in_memory_reorder_returns_correct_bytes(ref_fasta):
    """Lock-in: ``in_memory=True`` reorder puts the *right* contig at each index.

    Guards the copy path that already works so a future refactor cannot regress
    it into the #285 footgun.
    """
    full = gvl.Reference.from_path(ref_fasta, in_memory=True)
    reordered = gvl.Reference.from_path(ref_fasta, ["chr2", "chr1"], in_memory=True)

    assert reordered.contigs == ["chr2", "chr1"]

    # Integer contig index 0 on the reordered reference must be chr2's bytes,
    # matching chr2 fetched by name from the full-order reference.
    got = reordered.fetch(np.array([0]), [0], [100])
    want = full.fetch("chr2", 0, 100)
    np.testing.assert_array_equal(got.data, want.data)
