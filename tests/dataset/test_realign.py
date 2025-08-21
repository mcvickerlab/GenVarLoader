# %%
import numpy as np
from genoray import SparseGenotypes
from genoray._svar import dense2sparse
from genvarloader._dataset._tracks import shift_and_realign_track_sparse
from pytest_cases import parametrize_with_cases


# %%
def case_snps():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.zeros(2, dtype=np.int32)

    # (s p v)
    genos = np.array([[[0, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    track = np.arange(4, dtype=np.float32)
    desired = track.copy()
    query_start = 0

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        track,
        sparse_genos,
        desired,
        query_start,
    )


def case_indels():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.array([-1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    track = np.arange(4, dtype=np.float32)
    desired = np.array([0, 1, 3, 3], np.float32)
    query_start = 0

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        track,
        sparse_genos,
        desired,
        query_start,
    )


def case_spanning_del():
    v_starts = np.array([0], np.int32)
    ilens = np.array([-1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)

    shift = 0
    track = np.arange(1, 6, dtype=np.float32)
    # 0 (1 2 3 4 5) -> (2 3 4 5)
    # -  -
    desired = track[1:]
    query_start = 1

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        track,
        sparse_genos,
        desired,
        query_start,
    )


def case_shift_ins():
    v_starts = np.array([1, 3], np.int32)
    ilens = np.array([1, 1], dtype=np.int32)

    # (s p v) : (1 1 2)
    genos = np.array([[[1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 1
    track = np.arange(4, dtype=np.float32)
    # 0 1 2 3 -> shift 1 2 3 4 -> insert 1 1 2 3 4 -> truncate 1 1 2 3
    desired = np.array([1, 1, 2, 3], np.float32)
    query_start = 0

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        track,
        sparse_genos,
        desired,
        query_start,
    )


@parametrize_with_cases(
    "v_starts, ilens, shift, track, sparse_genos, desired, query_start",
    cases=".",
)
def test_sparse(
    v_starts,
    ilens,
    shift,
    track,
    sparse_genos: SparseGenotypes,
    desired,
    query_start,
):
    offset_idx = 0
    actual = np.empty(len(track) - query_start, np.float32)
    shift_and_realign_track_sparse(
        offset_idx=offset_idx,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=shift,
        track=track,
        query_start=query_start,
        out=actual,
    )

    np.testing.assert_equal(actual, desired)


# %%
# test_sparse(*case_snps())
# test_sparse(*case_indels())
# test_sparse(*case_spanning_del())
# test_sparse(*case_shift_ins())
