import numpy as np
from genvarloader._dataset._intervals import intervals_to_tracks


def _known_case():
    # one query, region length 5, one interval [11,13) painted with value 2.0 at query_start=10
    offset_idxs = np.array([0], np.int64)
    starts = np.array([10], np.int32)
    itv_starts = np.array([11], np.int32)
    itv_ends = np.array([13], np.int32)
    itv_values = np.array([2.0], np.float32)
    itv_offsets = np.array([0, 1], np.int64)
    out_offsets = np.array([0, 5], np.int64)
    return (
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out_offsets,
    )


def test_wrapper_matches_known_result():
    (
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out_offsets,
    ) = _known_case()
    out = np.empty(5, np.float32)
    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_array_equal(out, np.array([0, 2, 2, 0, 0], np.float32))

