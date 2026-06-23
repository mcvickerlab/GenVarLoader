import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader import RaggedVariants


def _rv():
    """A (2, 2, ~v) RaggedVariants: 2 batch × 2 ploidy, ragged variants."""
    alt = Ragged.from_offsets(
        np.frombuffer(b"ACGTACGT", dtype="S1").copy(),
        (2, 2, None),
        np.array([0, 1, 2, 3, 4], np.int64),  # 4 groups (b*p), 1-2 vars each
        str_offsets=np.array([0, 1, 3, 5, 8], np.int64),
    ).to_strings()
    start = Ragged.from_offsets(
        np.arange(4, dtype=np.int32), (2, 2, None), np.array([0, 1, 2, 3, 4], np.int64)
    )
    ilen = Ragged.from_offsets(
        np.zeros(4, np.int32), (2, 2, None), np.array([0, 1, 2, 3, 4], np.int64)
    )
    return RaggedVariants(alt=alt, start=start, ilen=ilen)


def test_raggedvariants_is_ragged_subclass():
    rv = _rv()
    assert isinstance(rv, Ragged)
    assert type(rv).__mro__[1] is Ragged
    assert rv.__slots__ == ()


def test_no_rag_composition_attribute():
    rv = _rv()
    assert not hasattr(rv, "_rag")  # composition field is gone
    assert rv._layout is not None  # holds the record layout directly


@pytest.mark.parametrize(
    "key", [0, slice(0, 2), np.array([1, 0])], ids=["int", "slice", "fancy"]
)
def test_positional_indexing_preserves_subclass(key):
    rv = _rv()
    out = rv[key]
    assert type(out) is RaggedVariants


def test_int_index_collapses_leading_axis():
    rv = _rv()  # (2, 2, ~v)
    assert rv[0].shape == (2, None)  # int collapses batch -> (ploidy, ~v)
    assert rv[0:2].shape == (2, 2, None)  # slice keeps batch (ploidy preserved)


def test_string_key_returns_base_ragged():
    rv = _rv()
    field = rv["start"]
    assert isinstance(field, Ragged)
    assert type(field) is Ragged  # NOT RaggedVariants


def test_inherited_structural_transforms_preserve_subclass():
    rv = _rv()
    assert type(rv.reshape(1, 2, 2, None)) is RaggedVariants  # base *shape signature
    assert type(rv.to_packed()) is RaggedVariants
    sq = rv.reshape(1, 2, 2, None).squeeze(0)
    assert type(sq) is RaggedVariants
    assert sq.shape == (2, 2, None)


def test_squeeze_axis0_equals_index0_on_singleton():
    rv = _rv().reshape(1, 2, 2, None)  # (1, 2, 2, ~v)
    np.testing.assert_array_equal(
        np.asarray(rv.squeeze(0)["start"].data), np.asarray(rv[0]["start"].data)
    )


def test_extra_field_via_getattr():
    alt = _rv()["alt"]
    start = _rv()["start"]
    af = Ragged.from_offsets(
        np.arange(4, dtype=np.float32),
        (2, 2, None),
        np.array([0, 1, 2, 3, 4], np.int64),
    )
    rv = RaggedVariants(alt=alt.to_strings(), start=start, ilen=_rv()["ilen"], AF=af)
    assert "AF" in rv.fields
    np.testing.assert_array_equal(
        np.asarray(rv.AF.data), np.arange(4, dtype=np.float32)
    )
    with pytest.raises(AttributeError):
        _ = rv.not_a_field
