import numpy as np

from genvarloader.pgen import weighted_activity_selection


def test_no_early_exit():
    query_end = 2
    v_starts = np.array([1, 1, 1, 6, 3, 9])
    v_ends = np.array([3, 5, 6, 9, 9, 10]) - 1  # to make ends inclusive
    w = v_ends - v_starts + 1  # ends are inclusive, + 1 to get actual length

    was_ends = np.empty(len(v_ends) + 1, dtype=v_ends.dtype)
    was_ends[0] = 0
    was_ends[1:] = v_ends
    q = np.searchsorted(was_ends + 1, v_starts, side="right") - 1

    actual_max_end, actual_end_idx = weighted_activity_selection(
        v_ends, w, q, query_end
    )
    desired_max_end = query_end + 9
    desired_end_idx = len(v_starts)

    assert actual_max_end == desired_max_end
    assert actual_end_idx == desired_end_idx


def test_early_exit():
    query_end = 2
    v_starts = np.array([1, 11, 11, 11, 16, 13, 19])
    v_ends = np.array([3, 13, 15, 16, 19, 19, 20]) - 1  # to make ends inclusive
    w = v_ends - v_starts + 1  # ends are inclusive, + 1 to get actual length

    was_ends = np.empty(len(v_ends) + 1, dtype=v_ends.dtype)
    was_ends[0] = 0
    was_ends[1:] = v_ends
    q = np.searchsorted(was_ends + 1, v_starts, side="right") - 1

    actual_max_end, actual_end_idx = weighted_activity_selection(
        v_ends, w, q, query_end
    )
    desired_max_end = query_end + 2
    desired_end_idx = 1

    assert actual_max_end == desired_max_end
    assert actual_end_idx == desired_end_idx


def test_del_shift():
    query_end = 1010696
    v_starts = np.array([110, 1010694, 1010695, 1010695, 1110695, 1110695])
    v_ends = np.array([110, 1010700, 1010698, 1010705, 1110695, 1110695])
    sorter = np.argsort(v_ends)
    v_starts = v_starts[sorter]
    v_ends = v_ends[sorter]
    w = v_ends - v_starts

    was_ends = np.empty(len(v_ends) + 1, dtype=v_ends.dtype)
    was_ends[0] = 0
    was_ends[1:] = v_ends
    q = np.searchsorted(was_ends + 1, v_starts, side="right") - 1

    s = 1
    q = (q[s:] - s).clip(0)
    actual_max_end, actual_end_idx = weighted_activity_selection(
        v_ends[s:], w[s:], q, query_end
    )
    desired_max_end = query_end + 10
    desired_end_idx = 4 - s

    assert actual_max_end == desired_max_end
    assert actual_end_idx == desired_end_idx


def test_ins_shift():
    query_end = 1110696
    v_starts = np.array([110, 1010694, 1010695, 1010695, 1110695, 1110695])
    v_ends = np.array([110, 1010700, 1010698, 1010705, 1110695, 1110695])
    ilen = np.array([0, -6, -3, -10, 75, 0])
    sorter = np.argsort(v_ends)
    v_starts = v_starts[sorter]
    v_ends = v_ends[sorter]
    w = -ilen[sorter]

    was_ends = np.empty(len(v_ends) + 1, dtype=v_ends.dtype)
    was_ends[0] = 0
    was_ends[1:] = v_ends
    q = np.searchsorted(was_ends + 1, v_starts, side="right") - 1

    s = 4
    q = (q[s:] - s).clip(0)
    actual_max_end, actual_end_idx = weighted_activity_selection(
        v_ends[s:], w[s:], q, query_end
    )
    desired_max_end = query_end
    desired_end_idx = 6 - s

    assert actual_max_end == desired_max_end
    assert actual_end_idx == desired_end_idx


if __name__ == "__main__":
    test_del_shift()
