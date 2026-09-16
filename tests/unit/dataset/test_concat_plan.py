"""Unit tests for gvl.concat's pure planning layer (no IO)."""

import numpy as np
import pytest

from genvarloader._dataset._concat_plan import (
    CONCAT_CHUNK_BYTES,
    Run,
    RunPlan,
    _default_order,
    as_plan,
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


def _rand_order(rng, counts):
    """A random valid merged order: every (ds, within) slot exactly once."""
    rows = [(d, i) for d, c in enumerate(counts) for i in range(c)]
    rng.shuffle(rows)
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def _sorted_interleave_order(rng, counts):
    """Each input's keys sorted, merged by global key -- the real two-cohort case."""
    keys = []
    for d, c in enumerate(counts):
        ks = sorted(rng.choice(10_000, size=c, replace=False))
        keys.extend((k, d, i) for i, k in enumerate(ks))
    keys.sort()
    return np.array([(d, i) for _, d, i in keys], dtype=np.int64).reshape(-1, 2)


def test_run_plan_matches_coalesce_provenance_exhaustively():
    """RunPlan must reproduce coalesce(provenance(...)) exactly, everywhere.

    This is the correctness proof for the whole change: `provenance` + `coalesce`
    are retained purely as this oracle. A plan that is wrong in a way this sweep
    misses produces a merged dataset whose slots point at the wrong samples --
    readable, and not obviously corrupt -- so the sweep deliberately includes the
    sorted key-interleave order that models the real two-cohort merge, not only
    shuffled and block-default orders.
    """
    import itertools

    rng = np.random.default_rng(0)
    n_checked = 0

    for axis, ploidy, n_ds in itertools.product(
        ("regions", "samples"), (1, 2, 3), (1, 2, 3)
    ):
        for _ in range(40):
            counts = [int(rng.integers(0, 5)) for _ in range(n_ds)]
            if axis == "regions":
                n_samples = int(rng.integers(0, 5))
                shapes = [(c, n_samples) for c in counts]
            else:
                n_regions = int(rng.integers(0, 5))
                shapes = [(n_regions, c) for c in counts]

            for mode in ("none", "default", "shuffled", "interleaved"):
                if mode == "none":
                    order = None
                elif mode == "default":
                    order = _default_order(len(counts), counts)
                elif mode == "shuffled":
                    order = _rand_order(rng, counts)
                else:
                    order = _sorted_interleave_order(rng, counts)

                want = coalesce(provenance(axis, shapes, ploidy, order=order))
                got = list(RunPlan(axis, shapes, ploidy, order=order))
                n_checked += 1
                assert got == want, (
                    f"axis={axis} ploidy={ploidy} shapes={shapes} mode={mode}\n"
                    f"want={want[:6]}\ngot={got[:6]}"
                )

    assert n_checked == 2880, "sweep shrank; the oracle coverage is the whole point"


def test_slot_batches_reproduce_the_provenance_map():
    """Concatenating the batches must rebuild `provenance` row for row.

    `copy_runs` gathers source lengths through these batches, so a batch whose
    slots are right but whose dataset column is wrong would read the correct
    offsets out of the wrong file -- a silent wrong merge, not a crash.
    """
    for axis, shapes, ploidy in (
        ("regions", [(3, 4), (2, 4)], 2),
        ("samples", [(3, 2), (3, 3)], 2),
        ("samples", [(2, 1), (2, 2)], 1),
    ):
        counts = [s for _, s in shapes] if axis == "samples" else [r for r, _ in shapes]
        interleaved = _sorted_interleave_order(np.random.default_rng(1), counts)
        for order in (None, interleaved):
            plan = RunPlan(axis, shapes, ploidy, order=order)
            want = provenance(axis, shapes, ploidy, order=order)
            got = np.zeros_like(want)
            seen = np.zeros(len(want), bool)
            for dst_start, ds_vec, slots in plan.slot_batches():
                n = len(slots)
                assert len(ds_vec) == n
                got[dst_start : dst_start + n, 0] = ds_vec
                got[dst_start : dst_start + n, 1] = slots
                seen[dst_start : dst_start + n] = True
            assert seen.all(), f"{axis} {shapes} left slots uncovered"
            np.testing.assert_array_equal(got, want)


def test_n_slots_matches_provenance_length():
    for axis, shapes, ploidy in (
        ("regions", [(3, 4), (2, 4)], 2),
        ("samples", [(3, 2), (3, 3)], 2),
        ("regions", [(0, 4), (2, 4)], 1),
        ("samples", [(0, 2), (0, 3)], 2),
    ):
        plan = RunPlan(axis, shapes, ploidy)
        assert plan.n_slots == len(provenance(axis, shapes, ploidy))


def test_slot_batches_chunk_large_region_runs(monkeypatch):
    """A regions-axis run can span the whole grid, so batches must be chunked.

    The production cap is 1Mi slots, far above anything a unit test can build,
    so this shrinks it and checks the chunking actually happens. Asserting only
    `max(sizes) <= cap` would pass vacuously on a single batch -- including
    against the unchunked implementation this test exists to rule out.
    """
    from genvarloader._dataset import _concat_plan

    monkeypatch.setattr(_concat_plan, "_SLOT_BATCH_SLOTS", 5)

    # One input, identity order -> exactly one run over 4 * 3 * 2 = 24 slots.
    plan = RunPlan("regions", [(4, 3)], 2)
    assert len(list(plan)) == 1

    batches = list(plan.slot_batches())
    assert [len(slots) for _, _, slots in batches] == [5, 5, 5, 5, 4]
    assert plan.n_slots == 24
    # Destination stays contiguous across every chunk boundary.
    assert [dst for dst, _, _ in batches] == [0, 5, 10, 15, 20]
    np.testing.assert_array_equal(
        np.concatenate([slots for _, _, slots in batches]), np.arange(24)
    )


def test_explicit_run_plan_round_trips_a_hand_built_list():
    runs = [Run(0, 0, 2, 0), Run(1, 5, 7, 2)]
    plan = as_plan(runs)
    assert list(plan) == runs
    assert plan.n_slots == 4
    assert as_plan(plan) is plan
    batches = list(plan.slot_batches())
    assert [b[0] for b in batches] == [0, 2]
    np.testing.assert_array_equal(batches[0][2], [0, 1])
    np.testing.assert_array_equal(batches[1][1], [1, 1])


def test_run_plan_is_re_iterable():
    """copy_runs iterates twice; a generator here would silently truncate output."""
    plan = RunPlan("samples", [(2, 2), (2, 1)], 2)
    assert list(plan) == list(plan)
    assert len(list(plan)) > 0


def test_run_plan_never_materializes_the_slot_space():
    """Peak allocation must stay O(R + S), not O(R * S * P).

    The old path on this grid allocates a (800_000, 2) int64 provenance map
    (12.8 MB) and coalesces it into 800_000 one-slot Runs (~147 MB at 184 bytes
    each). Both are invisible to every behavioural test, because RunPlan yields
    exactly the same runs -- so this bound is what stops a future edit from
    silently restoring them.
    """
    import tracemalloc

    shapes = [(200, 1000), (200, 1000)]
    # Interleaved samples: the worst case, one run per slot.
    order = np.array(
        [(d, i) for i in range(1000) for d in (0, 1)], dtype=np.int64
    ).reshape(-1, 2)

    tracemalloc.start()
    try:
        plan = RunPlan("samples", shapes, 2, order=order)
        n_runs = sum(1 for _ in plan)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert plan.n_slots == 200 * 2000 * 2
    assert n_runs == plan.n_slots // 2, "interleaved merge should give one run per cell"
    assert peak < (1 << 20), f"peak {peak} bytes: the slot space is being materialized"
