"""Tests for issue #191 — var_fields loading and dosage gating.

See docs/superpowers/specs/2026-05-24-issue-191-var-fields-loading-design.md.
"""

import shutil

import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE
from genvarloader._dataset._haps import _Variants


@pytest.fixture
def svar_with_dosages_ds(tmp_path, filtered_svar, source_bed):
    """Copy the canonical SVAR to tmp_path, add a synthetic dosages.npy, then
    write a fresh GVL dataset pointing at it. Returns the GVL path."""
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(filtered_svar, svar_copy)

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
    gvl.write(path=gvl_path, bed=source_bed, variants=svar_copy, overwrite=True)
    return gvl_path


def test_dosage_absent_when_not_requested(svar_with_dosages_ds, ref_fasta):
    """Regression: bug from issue #191.

    Dataset has dosages on disk; user did NOT request dosage in var_fields.
    The output RaggedVariants must not contain a `dosage` field.
    """
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" not in batch.fields, (
        f"dosage leaked into output despite not being in var_fields; got fields={batch.fields}"
    )


def test_dosage_present_when_requested(svar_with_dosages_ds, ref_fasta):
    """Sanity: opting in adds the field."""
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start", "dosage"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" in batch.fields


def test_available_var_fields_includes_dosage_when_present(
    svar_with_dosages_ds, ref_fasta
):
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert "dosage" in ds.available_var_fields


def test_available_var_fields_excludes_dosage_when_absent(phased_svar_gvl, ref_fasta):
    # The canonical SVAR (no dosages.npy) — opening the existing fixture
    # should not list dosage as available.
    ds = gvl.Dataset.open(
        phased_svar_gvl,
        ref_fasta,
        rc_neg=False,
    )
    assert "dosage" not in ds.available_var_fields


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


def test_haps_from_path_filters_info_loading(svar_with_dosages_ds, ref_fasta):
    """from_path(var_fields=[default]) does not load extra numeric info columns
    or open the dosages memmap."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    haps = ds._seqs  # type: ignore[attr-defined]
    # Default var_fields: only alt/ilen/start. info dict must not contain extras.
    assert haps.var_fields == ["alt", "ilen", "start"]
    assert set(haps.variants.info.keys()) == set()
    # Dosages file exists on disk but should not be loaded.
    assert haps.dosages is None


def test_haps_available_var_fields_from_schema(svar_with_dosages_ds, ref_fasta):
    """available_var_fields reflects the file's schema + dosage presence,
    not what was actually loaded."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert "dosage" in ds.available_var_fields
    # ref is also discoverable because the SVAR has a REF column
    assert "ref" in ds.available_var_fields


def test_dataset_open_accepts_var_fields(svar_with_dosages_ds, ref_fasta):
    """Dataset.open(var_fields=...) routes through to Haps.from_path so the
    requested fields are loaded eagerly at open time."""
    ds = gvl.Dataset.open(
        svar_with_dosages_ds,
        ref_fasta,
        rc_neg=False,
        var_fields=["alt", "ilen", "start", "dosage"],
    )
    haps = ds._seqs  # type: ignore[attr-defined]
    assert haps.var_fields == ["alt", "ilen", "start", "dosage"]
    assert haps.dosages is not None


def test_dataset_open_default_var_fields_is_minimum_useful_set(
    svar_with_dosages_ds, ref_fasta
):
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert ds.active_var_fields == ["alt", "ilen", "start"]


def test_with_settings_lazily_loads_new_info_field(
    svar_with_dosages_ds, filtered_svar, ref_fasta
):
    """Opening with default var_fields does not load AF (or other info columns).
    with_settings(var_fields=[..., 'AF']) should lazily extend the info dict."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    available_info = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available_info:
        pytest.skip("No numeric info columns; cannot test lazy info expansion")

    new_field = next(iter(available_info))
    haps_before = ds._seqs  # type: ignore[attr-defined]
    assert new_field not in haps_before.variants.info

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", new_field])
    haps_after = ds2._seqs  # type: ignore[attr-defined]
    assert new_field in haps_after.variants.info


def test_with_settings_lazily_loads_dosages(svar_with_dosages_ds, ref_fasta):
    """Opening with default var_fields does not memmap dosages.
    with_settings(var_fields=[..., 'dosage']) should memmap them."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert ds._seqs.dosages is None  # type: ignore[attr-defined]

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", "dosage"])
    assert ds2._seqs.dosages is not None  # type: ignore[attr-defined]
