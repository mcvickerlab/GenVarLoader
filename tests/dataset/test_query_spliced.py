import inspect

from genvarloader._dataset import _query


def test_spliced_has_no_dead_variant_guard():
    src = inspect.getsource(_query._getitem_spliced)
    assert "_VARIANT_TYPES_S" not in src, (
        "spliced variant RC guard is unreachable (spliced variants are rejected "
        "upstream) and must be removed"
    )
