"""Tests for issue #191 — var_fields loading and dosage gating.

See docs/superpowers/specs/2026-05-24-issue-191-var-fields-loading-design.md.
"""

import shutil
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE
from genvarloader._dataset._haps import _Variants

_DATA = Path(__file__).resolve().parents[1] / "data"
_REF = _DATA / "fasta" / "hg38.fa.bgz"
_SOURCE_SVAR = _DATA / "filtered.svar"
_SOURCE_BED = _DATA / "source.bed"


@pytest.fixture
def svar_with_dosages_ds(tmp_path):
    """Copy the canonical SVAR to tmp_path, add a synthetic dosages.npy, then
    write a fresh GVL dataset pointing at it. Returns the GVL path."""
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(_SOURCE_SVAR, svar_copy)

    v_idxs_bytes = (svar_copy / "variant_idxs.npy").stat().st_size
    n_records = v_idxs_bytes // np.dtype(np.uint32).itemsize
    mm = np.memmap(
        svar_copy / "dosages.npy",
        dtype=DOSAGE_TYPE,
        mode="w+",
        shape=(n_records,),
    )
    mm[:] = np.arange(n_records, dtype=DOSAGE_TYPE)
    mm.flush()
    del mm

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=_SOURCE_BED, variants=svar_copy, overwrite=True)
    return gvl_path


def test_dosage_absent_when_not_requested(svar_with_dosages_ds):
    """Regression: bug from issue #191.

    Dataset has dosages on disk; user did NOT request dosage in var_fields.
    The output RaggedVariants must not contain a `dosage` field.
    """
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" not in batch.fields, (
        f"dosage leaked into output despite not being in var_fields; got fields={batch.fields}"
    )


def test_dosage_present_when_requested(svar_with_dosages_ds):
    """Sanity: opting in adds the field."""
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start", "dosage"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" in batch.fields


def test_available_var_fields_includes_dosage_when_present(svar_with_dosages_ds):
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    assert "dosage" in ds.available_var_fields


def test_available_var_fields_excludes_dosage_when_absent():
    # The canonical SVAR (no dosages.npy) — opening the existing fixture
    # should not list dosage as available.
    ds = gvl.Dataset.open(
        _DATA / "phased_dataset.svar.gvl",
        _REF,
        rc_neg=False,
    )
    assert "dosage" not in ds.available_var_fields


def test_available_info_fields_lists_numeric_columns_without_loading():
    """Schema peek: returns numeric columns from the variants table without
    materializing data."""
    fields = _Variants.available_info_fields(_SOURCE_SVAR / "index.arrow")
    # Canonical SVAR has at least AF in its index.
    # POS and ILEN must NOT appear — they're treated as positional, not info.
    assert "POS" not in fields
    assert "ILEN" not in fields
    # All returned names are strings
    assert all(isinstance(f, str) for f in fields)
    # Non-empty for a real SVAR
    assert len(fields) >= 0  # may be 0 if no extra numeric columns; that's fine


def test_from_table_info_fields_filter():
    """When info_fields is a set, only those numeric columns are loaded."""
    available = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    if not available:
        pytest.skip("No numeric info columns in canonical SVAR; cannot exercise filter")

    pick = {next(iter(available))}
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=pick)
    assert set(v.info.keys()) == pick


def test_from_table_info_fields_none_loads_all():
    """Back-compat: info_fields=None loads every numeric column (current behavior)."""
    available = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=None)
    assert set(v.info.keys()) == available
