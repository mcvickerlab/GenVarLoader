"""Unit tests for gvl.concat's pure planning layer (no IO)."""

import numpy as np
import pytest

from genvarloader._dataset._concat_plan import (
    CONCAT_CHUNK_BYTES,
    Run,
    coalesce,
    provenance,
)


def test_chunk_bytes_is_16mib():
    assert CONCAT_CHUNK_BYTES == 16 << 20


def test_provenance_regions_appends_blocks():
    # two datasets, 2 samples each, ploidy 1; A has 2 regions, B has 1.
    prov = provenance("regions", [(2, 2), (1, 2)], ploidy=1)
    # merged order is (r, s): A(r0s0) A(r0s1) A(r1s0) A(r1s1) B(r0s0) B(r0s1)
    assert prov.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
    ]


def test_provenance_samples_interleaves_per_region():
    # two datasets, 2 regions each; A has 1 sample, B has 2. ploidy 1.
    prov = provenance("samples", [(2, 1), (2, 2)], ploidy=1)
    # merged S' = 3. Per region: A's sample, then B's two.
    assert prov.tolist() == [
        [0, 0],
        [1, 0],
        [1, 1],  # region 0
        [0, 1],
        [1, 2],
        [1, 3],  # region 1
    ]


def test_provenance_accounts_for_ploidy():
    prov = provenance("regions", [(1, 1), (1, 1)], ploidy=2)
    # each (r, s) contributes P consecutive slots
    assert prov.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_coalesce_regions_gives_one_run_per_dataset():
    prov = provenance("regions", [(2, 2), (1, 2)], ploidy=1)
    runs = coalesce(prov)
    assert runs == [
        Run(src=0, src_start=0, src_stop=4, dst_start=0),
        Run(src=1, src_start=0, src_stop=2, dst_start=4),
    ]


def test_coalesce_samples_gives_run_per_dataset_per_region():
    prov = provenance("samples", [(2, 1), (2, 2)], ploidy=1)
    runs = coalesce(prov)
    assert runs == [
        Run(src=0, src_start=0, src_stop=1, dst_start=0),
        Run(src=1, src_start=0, src_stop=2, dst_start=1),
        Run(src=0, src_start=1, src_stop=2, dst_start=3),
        Run(src=1, src_start=2, src_stop=4, dst_start=4),
    ]


def test_coalesce_covers_every_slot_exactly_once():
    prov = provenance("samples", [(3, 2), (3, 1), (3, 3)], ploidy=2)
    runs = coalesce(prov)
    covered = np.zeros(len(prov), dtype=np.int32)
    for r in runs:
        n = r.src_stop - r.src_start
        covered[r.dst_start : r.dst_start + n] += 1
    assert (covered == 1).all()


def test_coalesce_runs_are_monotonic_within_each_source():
    """Destination-ordered iteration must keep each source's reads sequential."""
    prov = provenance("samples", [(4, 2), (4, 3)], ploidy=1)
    runs = coalesce(prov)
    for src in (0, 1):
        starts = [r.src_start for r in runs if r.src == src]
        assert starts == sorted(starts)


def test_provenance_rejects_unknown_axis():
    with pytest.raises(ValueError, match="axis must be"):
        provenance("chromosomes", [(1, 1)], ploidy=1)


# --- `order` generalizes provenance to interleaved (non-block) merges --------
#
# On-disk GVL stores are sorted along both axes (samples.sort() in write(),
# _prep_bed sorts regions), so the true merged slot order is the sorted order,
# not necessarily dataset-0's-block-then-dataset-1's-block. Block-concatenation
# (the `order=None` default) only happens to coincide with the sorted order when
# the inputs' key ranges don't interleave. These tests use an `order` where they
# DO interleave, and assert the mapping block-concatenation could not produce.


def test_provenance_regions_with_interleaved_order():
    # ds0 has 2 regions, ds1 has 1; but the TRUE sorted merged order is
    # ds1's region, then ds0's two regions (e.g. ds1's region sorts first).
    shape_per_ds = [(2, 2), (1, 2)]
    order = np.array([[1, 0], [0, 0], [0, 1]])
    prov = provenance("regions", shape_per_ds, ploidy=1, order=order)
    assert prov.tolist() == [
        [1, 0],
        [1, 1],  # merged region 0 <- ds1's region 0
        [0, 0],
        [0, 1],  # merged region 1 <- ds0's region 0
        [0, 2],
        [0, 3],  # merged region 2 <- ds0's region 1
    ]
    # Block concatenation (order=None) cannot produce this: it always puts all
    # of ds0 first.
    block = provenance("regions", shape_per_ds, ploidy=1)
    assert prov.tolist() != block.tolist()


def test_provenance_samples_with_interleaved_order():
    # ds0 has 2 samples (w=0,1), ds1 has 1 (w=0); the TRUE sorted sample order
    # interleaves: ds0's sample 0, then ds1's sample, then ds0's sample 1.
    shape_per_ds = [(2, 2), (2, 1)]
    order = np.array([[0, 0], [1, 0], [0, 1]])
    prov = provenance("samples", shape_per_ds, ploidy=1, order=order)
    assert prov.tolist() == [
        [0, 0],
        [1, 0],
        [0, 1],  # region 0
        [0, 2],
        [1, 1],
        [0, 3],  # region 1
    ]
    block = provenance("samples", shape_per_ds, ploidy=1)
    assert prov.tolist() != block.tolist()


def test_provenance_regions_interleaved_order_coalesces_and_covers_all_slots():
    shape_per_ds = [(3, 2), (2, 2)]
    # ds1's regions interleave between ds0's: ds0 r0, ds1 r0, ds0 r1, ds1 r1, ds0 r2
    order = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2]])
    prov = provenance("regions", shape_per_ds, ploidy=2, order=order)
    runs = coalesce(prov)
    # Every merged region is independent (S*P contiguous both sides), so
    # interleaving must still yield exactly one run per merged region.
    assert len(runs) == len(order)
    covered = np.zeros(len(prov), dtype=np.int32)
    for r in runs:
        n = r.src_stop - r.src_start
        covered[r.dst_start : r.dst_start + n] += 1
    assert (covered == 1).all()


def test_provenance_samples_interleaved_order_covers_all_slots():
    shape_per_ds = [(2, 3), (2, 2)]
    # merged sample order interleaves ds0 and ds1's samples.
    order = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2]])
    prov = provenance("samples", shape_per_ds, ploidy=2, order=order)
    runs = coalesce(prov)
    covered = np.zeros(len(prov), dtype=np.int32)
    for r in runs:
        n = r.src_stop - r.src_start
        covered[r.dst_start : r.dst_start + n] += 1
    assert (covered == 1).all()
