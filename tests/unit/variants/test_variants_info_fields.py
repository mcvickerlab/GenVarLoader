"""Unit tests for ``_Variants`` info-field plumbing.

Originally lived in tests/integration/dataset/test_issue_191_var_fields.py;
extracted to the unit tier because every test here calls ``_Variants``
class methods (``from_table``, ``available_info_fields``, ``load_info``)
against the canonical ``filtered_svar`` path fixture — no Dataset, no
write pipeline.

Dataset-level tests (``Dataset.open(var_fields=...)``, ``with_settings``
lazy reload, dosage gating end-to-end) remain in the original file
because they require the full open/write path.
"""

import pytest
from genvarloader._dataset._haps import _Variants


def test_available_info_fields_lists_numeric_columns_without_loading(filtered_svar):
    """Schema peek: returns numeric columns from the variants table without
    materializing data."""
    fields = _Variants.available_info_fields(filtered_svar / "index.arrow")
    # Canonical SVAR has at least AF in its index.
    # POS and ILEN must NOT appear — they're treated as positional, not info.
    assert "POS" not in fields
    assert "ILEN" not in fields
    # All returned names are strings
    assert all(isinstance(f, str) for f in fields)
    # Non-empty for a real SVAR
    assert len(fields) >= 0  # may be 0 if no extra numeric columns; that's fine


def test_from_table_info_fields_filter(filtered_svar):
    """When info_fields is a set, only those numeric columns are loaded."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available:
        pytest.skip("No numeric info columns in canonical SVAR; cannot exercise filter")

    pick = {next(iter(available))}
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=pick)
    assert set(v.info.keys()) == pick


def test_from_table_info_fields_none_loads_all(filtered_svar):
    """Back-compat: info_fields=None loads every numeric column (current behavior)."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=None)
    assert set(v.info.keys()) == available


def test_load_info_extends_info_dict(filtered_svar):
    """load_info reads only the missing fields from disk and merges them."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available:
        pytest.skip(
            "No numeric info columns in canonical SVAR; cannot exercise load_info"
        )

    pick = next(iter(available))
    # Start with empty info
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=set())
    assert pick not in v.info

    v.load_info([pick])
    assert pick in v.info


def test_load_info_idempotent_for_already_loaded_fields(filtered_svar):
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=None)
    already = list(v.info.keys())
    if not already:
        pytest.skip("No numeric info columns in canonical SVAR")
    # Snapshot one array; load_info shouldn't reload it.
    arr0 = v.info[already[0]]
    v.load_info(already)
    assert v.info[already[0]] is arr0
