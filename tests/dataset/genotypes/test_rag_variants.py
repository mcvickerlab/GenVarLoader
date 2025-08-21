import awkward as ak
import numpy as np
from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index
from genoray._svar import POS_TYPE
from genvarloader import Ragged, RaggedVariants
from genvarloader._dataset._rag_variants import _infer_germline_ccfs
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro.rag import OFFSET_TYPE, lengths_to_offsets


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


def _bpv(p: int, data: NDArray, offsets: NDArray) -> ak.Array:
    node = NumpyArray(data)  # type: ignore
    node = ListOffsetArray(Index(offsets), node)
    node = RegularArray(node, p)
    return ak.Array(node)


def _bpvl(p: int, data: NDArray, l_offsets: NDArray, v_offsets: NDArray) -> ak.Array:
    node = NumpyArray(
        data.view(np.uint8),  # type: ignore
        parameters={"__array__": "byte"},
    )
    node = ListOffsetArray(Index(l_offsets), node)
    node = ListOffsetArray(Index(v_offsets), node)
    node = RegularArray(node, p)
    return ak.Array(node)


def rc_no_rc():
    # (b, p, ~v, ~l)
    # (2, 1, [0, 2], [1, 2])
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([0, 2], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    alts = _bpvl(1, np.array([b"ACT"]).view(np.uint8), l_offsets, v_offsets)
    ilens = Ragged(_bpv(1, np.array([0, 1], np.int32), v_offsets))
    v_starts = Ragged(_bpv(1, np.array([0, 1], POS_TYPE), v_offsets))
    dosages = Ragged(_bpv(1, np.array([0.1, 0.2], np.float32), v_offsets))

    ragv = RaggedVariants(alts, v_starts, ilens, dosages)
    to_rc = np.zeros(2, np.bool_)[:, None]
    desired = RaggedVariants(alts, v_starts, ilens, dosages)

    return ragv, to_rc, desired


def rc_second_batch():
    # (b, p, ~v, ~l)
    # (2, 2, [0, 2], [1, 2])
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([0, 2], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    alts = _bpvl(1, np.array([b"ACT"]).view(np.uint8), l_offsets, v_offsets)
    rc_alts = _bpvl(1, np.array([b"TAG"]).view(np.uint8), l_offsets, v_offsets)
    ilens = Ragged(_bpv(1, np.array([0, 1], np.int32), v_offsets))
    v_starts = Ragged(_bpv(1, np.array([0, 1], POS_TYPE), v_offsets))
    dosages = Ragged(_bpv(1, np.array([0.1, 0.2], np.float32), v_offsets))

    ragv = RaggedVariants(alts, v_starts, ilens, dosages)
    to_rc = np.array([False, True])[:, None]
    desired = RaggedVariants(rc_alts, v_starts, ilens, dosages)

    return ragv, to_rc, desired


def rc_all():
    # (b, p, ~v, ~l)
    # (2, 2, [0, 2], [1, 2])
    l_lens = np.array([1, 2], np.int32)
    l_offsets = lengths_to_offsets(l_lens)
    v_lens = np.array([0, 2], np.int32)
    v_offsets = lengths_to_offsets(v_lens)
    alts = _bpvl(1, np.array([b"ACT"]).view(np.uint8), l_offsets, v_offsets)
    rc_alts = _bpvl(1, np.array([b"TAG"]).view(np.uint8), l_offsets, v_offsets)
    ilens = Ragged(_bpv(1, np.array([0, 1], np.int32), v_offsets))
    v_starts = Ragged(_bpv(1, np.array([0, 1], POS_TYPE), v_offsets))
    dosages = Ragged(_bpv(1, np.array([0.1, 0.2], np.float32), v_offsets))

    ragv = RaggedVariants(alts, v_starts, ilens, dosages)
    to_rc = None
    desired = RaggedVariants(rc_alts, v_starts, ilens, dosages)

    return ragv, to_rc, desired


@parametrize_with_cases("ragv, to_rc, desired", cases=".", prefix="rc_")
def test_rc(ragv: RaggedVariants, to_rc: NDArray[np.bool_], desired: RaggedVariants):
    rc_ragv = ragv.rc_(to_rc)

    actual_alts = rc_ragv.alts.layout
    while not isinstance(actual_alts, NumpyArray):
        actual_alts = actual_alts.content
    actual_alts = actual_alts.data

    desired_alts = desired.alts.layout
    while not isinstance(desired_alts, NumpyArray):
        desired_alts = desired_alts.content
    desired_alts = desired_alts.data

    np.testing.assert_equal(actual_alts, desired_alts)

    assert ak.all((rc_ragv.ilens == desired.ilens), None)
    assert ak.all((rc_ragv.v_starts == desired.v_starts), None)
    if ragv.dosages is not None:
        assert ak.all((rc_ragv.dosages == desired.dosages), None)
