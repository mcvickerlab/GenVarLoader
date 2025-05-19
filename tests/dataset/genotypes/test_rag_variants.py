import numpy as np
from genoray._svar import POS_TYPE
from genvarloader._dataset._rag_variants import _infer_germline_ccfs
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro._ragged import OFFSET_TYPE


def ccfs_no_overlaps():
    ccfs = np.array([np.nan, 0.1, 0.2, 0.3], np.float32)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    np.nan_to_num(desired, False, max_ccf)

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_no_germs():
    ccfs = np.array([0.1, 0.1, 0.2, 0.3], np.float32)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_all_nonoverlap_germs():
    ccfs = np.full(4, np.nan, np.float32)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = np.full_like(ccfs, max_ccf)

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_overlap_som():
    ccfs = np.array([0.1, np.nan, 0.2, 0.3], np.float32)
    v_starts = np.array([0, 1, 1, 1], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    desired[1] = max_ccf - ccfs[2:].sum()

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_overlap_germ():
    ccfs = np.array([0.1, np.nan, np.nan, 0.3], np.float32)
    v_starts = np.array([0, 1, 1, 1], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    desired[1] = max_ccf - ccfs[3]
    desired[2] = 0.0

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_spanning_del():
    ccfs = np.array([0.1, np.nan, 0.2, 0.3], np.float32)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([-1, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    desired[1] = max_ccf - ccfs[0]

    return ccfs, v_starts, ilens, v_offsets, max_ccf, desired


@parametrize_with_cases(
    "ccfs, v_starts, ilens, v_offsets, max_ccf, desired", cases=".", prefix="ccfs_"
)
def test_infer_germ_ccfs(
    ccfs: NDArray[np.float32],
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    v_offsets: NDArray[OFFSET_TYPE],
    max_ccf: float,
    desired: NDArray[np.float32],
):
    _infer_germline_ccfs(
        ccfs=ccfs,
        v_offsets=v_offsets,
        v_starts=v_starts,
        ilens=ilens,
        max_ccf=max_ccf,
    )
    np.testing.assert_equal(ccfs, desired)
