"""Unit tests for InsertionFill subclasses, the shift_and_realign_track_sparse
kernel, and Tracks.with_insertion_fill plumbing.

Originally lived in tests/integration/dataset/test_insertion_fill.py;
extracted to the unit tier because every test here constructs its inputs
in-memory and exercises only the in-memory reconstruction path — no
Dataset, no disk I/O. Three dataset-dependent tests (end-to-end + dummy
dataset + rejects-when-no-tracks) remain in the original file.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from genoray._svar import dense2sparse
from genvarloader._dataset._insertion_fill import (
    CONSTANT,
    FLANK_SAMPLE,
    INTERPOLATE,
    REPEAT_5P,
    REPEAT_5P_NORM,
    Constant,
    FlankSample,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
    lower,
)
from genvarloader._dataset._tracks import shift_and_realign_track_sparse

# Make tests/_builders/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _builders.reconstruct import make_tracks

# ---------------------------------------------------------------------------
# InsertionFill subclasses — pure unit tests of the dataclass / serializer
# ---------------------------------------------------------------------------


def test_lower_all_strategies():
    strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        Constant(value=0.5),
        FlankSample(flank_width=3),
        Interpolate(order=2),
    ]
    ids, params = lower(strategies)
    assert ids.dtype == np.int8
    assert params.dtype == np.float64
    assert ids.tolist() == [
        REPEAT_5P,
        REPEAT_5P_NORM,
        CONSTANT,
        FLANK_SAMPLE,
        INTERPOLATE,
    ]
    assert params[2, 0] == 0.5
    assert params[3, 0] == 3.0
    assert params[4, 0] == 2.0
    # Repeat5p (index 0) and Repeat5pNormalized (index 1) have no params: assert zeros.
    assert np.all(params[0] == 0)
    assert np.all(params[1] == 0)


def test_lower_empty():
    ids, params = lower([])
    assert ids.shape == (0,)
    assert params.shape == (0, 1)


def test_constant_default_is_nan():
    assert math.isnan(Constant().value)


def test_flank_sample_negative_width_rejected():
    with pytest.raises(ValueError, match="flank_width must be >= 0"):
        FlankSample(flank_width=-1)


def test_interpolate_order_capped():
    Interpolate(order=1)
    Interpolate(order=3)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=4)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=0)


def test_lower_unknown_class_raises():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="Unknown InsertionFill subclass"):
        lower([Bogus()])  # type: ignore[list-item]


def test_insertion_fill_base_not_instantiable():
    from genvarloader._dataset._insertion_fill import InsertionFill

    with pytest.raises(TypeError, match="abstract"):
        InsertionFill()


def _run_kernel(strategy_id, params, base_seed=np.uint64(0)):
    """Run the kernel on a single insertion at v_rel_pos=1, v_diff=3.

    Track is [0, 10, 20, 30, 40] (5 values). The variant at start=1 (v_rel_pos=1)
    with ilen=3 inserts 3 bases. Output length matches track length + 3 = 8.
    """
    v_starts = np.array([1], dtype=np.int32)
    ilens = np.array([3], dtype=np.int32)
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    out = np.zeros(8, dtype=np.float32)
    shift_and_realign_track_sparse(
        offset_idx=0,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=0,
        track=track,
        query_start=0,
        out=out,
        params=params,
        strategy_id=strategy_id,
        base_seed=base_seed,
    )
    return out, track


def test_kernel_repeat_5p_default():
    out, track = _run_kernel(REPEAT_5P, np.zeros(1, dtype=np.float64))
    # Positions 1..4 are the v_len=4 insertion stretch (anchor + 3 inserted bases).
    np.testing.assert_array_equal(
        out[1:5], np.array([10, 10, 10, 10], dtype=np.float32)
    )
    assert out[0] == 0.0
    np.testing.assert_array_equal(out[5:], track[2:])


def test_kernel_repeat_5p_normalized():
    out, _ = _run_kernel(REPEAT_5P_NORM, np.zeros(1, dtype=np.float64))
    # Sum across v_len=4 positions should equal track[v_rel_pos] = 10.
    assert math.isclose(out[1:5].sum(), 10.0, abs_tol=1e-6)
    assert np.allclose(out[1:5], out[1])


def test_kernel_constant_nan():
    params = np.zeros(1, dtype=np.float64)
    params[0] = float("nan")
    out, _ = _run_kernel(CONSTANT, params)
    assert np.all(np.isnan(out[1:5]))
    assert out[0] == 0.0
    assert out[5] == 20.0


def test_kernel_flank_sample_pool_membership():
    params = np.zeros(1, dtype=np.float64)
    params[0] = 2.0  # flank_width
    # pool = track[max(0, -1):min(4, 3)+1] = track[0:4] = [0, 10, 20, 30]
    out, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(42))
    pool = {0.0, 10.0, 20.0, 30.0}
    for v in out[1:5]:
        assert float(v) in pool


def test_kernel_flank_sample_deterministic():
    params = np.zeros(1, dtype=np.float64)
    params[0] = 2.0
    a, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(123))
    b, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(123))
    np.testing.assert_array_equal(a, b)


def test_kernel_interpolate_linear():
    params = np.zeros(1, dtype=np.float64)
    params[0] = 1.0  # order=1
    out, _ = _run_kernel(INTERPOLATE, params)
    # Linear between track[1]=10 (at x=0) and track[2]=20 (at x=v_len=4).
    # Evaluated at x=0,1,2,3. Slope = (20-10)/4 = 2.5.
    expected = np.array([10.0, 12.5, 15.0, 17.5], dtype=np.float32)
    np.testing.assert_allclose(out[1:5], expected, atol=1e-5)


def test_kernel_interpolate_cubic_passes_through_anchors():
    """Cubic Lagrange with 2 anchors per side must pass through all 4 anchor points.

    Track = [0, 10, 20, 30, 40], v_rel_pos=1, v_len=4.
    5' anchors: (xs=0, ys=10), (xs=-1, ys=0)
    3' anchors: (xs=4, ys=20), (xs=5, ys=30)
    Lagrange cubic through these 4 non-uniformly spaced points evaluated at
    x=0,1,2,3 gives [10, 14, 15, 16].
    """
    params = np.zeros(1, dtype=np.float64)
    params[0] = 3.0  # order=3 -> 2 anchors per side
    out, _ = _run_kernel(INTERPOLATE, params)
    expected = np.array([10.0, 14.0, 15.0, 16.0], dtype=np.float32)
    np.testing.assert_allclose(out[1:5], expected, atol=1e-4)


def test_kernel_flank_sample_edge_clamp():
    """Insertion at the very start of the track — pool clamps without crash."""
    v_starts = np.array([0], dtype=np.int32)
    ilens = np.array([2], dtype=np.int32)
    track = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    params = np.zeros(1, dtype=np.float64)
    params[0] = 10.0  # flank_width larger than track
    out = np.zeros(5, dtype=np.float32)
    shift_and_realign_track_sparse(
        offset_idx=0,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=0,
        track=track,
        query_start=0,
        out=out,
        params=params,
        strategy_id=FLANK_SAMPLE,
        base_seed=np.uint64(7),
    )
    pool = {5.0, 6.0, 7.0}
    for v in out[:3]:
        assert float(v) in pool


def test_kernel_flank_sample_query_hap_affects_hash():
    """Different (query, hap) seeds must drive different samples for the same base_seed."""
    v_starts = np.array([1], dtype=np.int32)
    ilens = np.array([3], dtype=np.int32)
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    params = np.zeros(1, dtype=np.float64)
    params[0] = 2.0

    def run(query, hap):
        out = np.zeros(8, dtype=np.float32)
        shift_and_realign_track_sparse(
            offset_idx=0,
            geno_v_idxs=sparse_genos.data,
            geno_offsets=sparse_genos.offsets,
            v_starts=v_starts,
            ilens=ilens,
            shift=0,
            track=track,
            query_start=0,
            out=out,
            params=params,
            strategy_id=FLANK_SAMPLE,
            base_seed=np.uint64(99),
            query=query,
            hap=hap,
        )
        return out[1:5].copy()

    a = run(0, 0)
    b = run(1, 0)
    c = run(0, 1)
    # Each pair should differ at least one position.
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Tracks reconstructor — insertion_fill plumbing (uses make_tracks builder)
# ---------------------------------------------------------------------------


def test_with_insertion_fill_single_applies_to_all():
    tracks = make_tracks(["a", "b"])
    new = tracks.with_insertion_fill(Constant(0.0))
    assert isinstance(new.insertion_fill["a"], Constant)
    assert isinstance(new.insertion_fill["b"], Constant)
    # original unchanged (evolve returns new instance)
    assert isinstance(tracks.insertion_fill["a"], Repeat5p)


def test_with_insertion_fill_dict_partial_falls_back():
    tracks = make_tracks(["a", "b"])
    new = tracks.with_insertion_fill({"a": FlankSample(flank_width=2)})
    assert isinstance(new.insertion_fill["a"], FlankSample)
    assert isinstance(new.insertion_fill["b"], Repeat5p)


def test_with_tracks_prunes_insertion_fill():
    tracks = make_tracks(["a", "b"]).with_insertion_fill({
        "a": Constant(0.0),
        "b": FlankSample(),
    })
    new = tracks.with_tracks("a")
    assert set(new.insertion_fill) == {"a"}
    assert isinstance(new.insertion_fill["a"], Constant)
