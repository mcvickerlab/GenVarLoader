import numpy as np
import pytest
from genoray import SparseGenotypes
from genvarloader._dataset._genotypes import filter_af
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases


@pytest.mark.skip
@parametrize_with_cases(
    "min_af, max_af, offset_idx, genos, afs, desired_keep, desired_offsets",
    cases=".",
    prefix="case_filter_af",
)
def test_filter_af(
    min_af: float,
    max_af: float,
    offset_idx: NDArray[np.intp],
    genos: SparseGenotypes,
    afs: NDArray[np.float32],
    desired_keep: NDArray[np.bool_],
    desired_offsets: NDArray[np.int64],
):
    keep, keep_offsets = filter_af(
        offset_idx,
        genos.offsets,
        genos.data,
        afs,
        min_af,
        max_af,
    )
    assert np.allclose(keep, desired_keep)
    assert np.allclose(keep_offsets, desired_offsets)
