import numpy as np
import pytest
from seqpro.rag import Ragged
from seqpro.rag import to_padded as sp_to_padded
from genvarloader._ragged import to_padded as gvl_to_padded


@pytest.mark.parametrize("dtype,pad", [("S1", b"N"), ("i4", -1), ("f4", 0.0)])
@pytest.mark.parametrize("rows", [[0, 1, 3, 2], [5], [0, 0, 4], [0, 0, 0]])
def test_gvl_to_padded_matches_seqpro(dtype, pad, rows):
    offsets = np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)
    n = int(offsets[-1])
    data = (
        (np.arange(n, dtype=np.int64) % 4).astype(dtype)
        if dtype != "S1"
        else np.frombuffer(b"ACGT" * (n // 4 + 1), dtype="S1")[:n]
    )
    # S1 dtype uses (n_rows,) shape (rag_dim=0 via str_offsets path);
    # numeric dtypes require an explicit None ragged dimension: (n_rows, None).
    shape = (len(rows),) if dtype == "S1" else (len(rows), None)
    rag = Ragged.from_offsets(np.ascontiguousarray(data), shape, offsets)
    np.testing.assert_array_equal(gvl_to_padded(rag, pad), sp_to_padded(rag, pad))
