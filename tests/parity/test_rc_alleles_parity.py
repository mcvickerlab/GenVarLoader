import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from genvarloader._dataset import _flat_variants  # noqa: F401  (registers rc_alleles)
from genvarloader import _dispatch

_ACGTN = np.frombuffer(b"ACGTN", np.uint8)


@st.composite
def _allele_batch(draw):
    n_rows = draw(st.integers(1, 4))
    alleles_per_row = [draw(st.integers(0, 3)) for _ in range(n_rows)]
    var_offsets = np.concatenate([[0], np.cumsum(alleles_per_row)]).astype(np.int64)
    n_alleles = int(var_offsets[-1])
    lens = [draw(st.integers(0, 5)) for _ in range(n_alleles)]
    seq_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(seq_offsets[-1])
    data = _ACGTN[draw(st.lists(st.integers(0, 4), min_size=total, max_size=total))] \
        if total else np.zeros(0, np.uint8)
    data = np.ascontiguousarray(data, np.uint8)
    mask = np.array([draw(st.booleans()) for _ in range(n_rows)], np.bool_)
    return data, seq_offsets, var_offsets, mask


@settings(max_examples=200, deadline=None)
@given(batch=_allele_batch())
def test_rc_alleles_rust_matches_reference(batch):
    data, seq_offsets, var_offsets, mask = batch
    numba_fn, rust_fn = _dispatch.backends("rc_alleles")
    a = data.copy()
    b = data.copy()
    numba_fn(a, seq_offsets, var_offsets, mask)
    rust_fn(b, seq_offsets, var_offsets, mask)
    assert a.tobytes() == b.tobytes()
