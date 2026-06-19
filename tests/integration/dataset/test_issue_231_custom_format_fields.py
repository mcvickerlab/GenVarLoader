"""Tests for issue #231 — surfacing genoray custom per-call FORMAT fields.

A .svar can register arbitrary Number=G FORMAT fields in
<svar>/metadata.json["fields"] (name -> numpy dtype) stored as <svar>/<name>.npy,
sharing the genotype offsets (byte-identical layout to dosages.npy). These tests
exercise the full Dataset.open path with a hand-crafted custom field.
"""

import json
import shutil

import awkward as ak
import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE

FIELD_NAME = "mutcat"
FIELD_DTYPE = "int16"


@pytest.fixture
def custom_field_ds(tmp_path, filtered_svar, source_bed):
    """Copy the canonical SVAR to tmp_path, register a custom int16 FORMAT field
    `mutcat` (values = arange over the genotype calls) AND a parallel dosages.npy
    with identical arange values, then write a fresh GVL dataset linking to it.

    Returns (gvl_path, field_name, n_records).
    """
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(filtered_svar, svar_copy)

    # One value per genotype call: length == n variant_idxs (V_IDX_TYPE = uint32).
    v_idxs_bytes = (svar_copy / "variant_idxs.npy").stat().st_size
    n_records = v_idxs_bytes // np.dtype(np.uint32).itemsize

    # Custom int16 field, values 0..n-1.
    mm = np.memmap(
        svar_copy / f"{FIELD_NAME}.npy", dtype=FIELD_DTYPE, mode="w+", shape=(n_records,)
    )
    mm[:] = np.arange(n_records, dtype=FIELD_DTYPE)
    mm.flush()
    del mm

    # Parallel dosages with the same arange values, to prove gather-equivalence.
    dmm = np.memmap(
        svar_copy / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=(n_records,)
    )
    dmm[:] = np.arange(n_records, dtype=DOSAGE_TYPE)
    dmm.flush()
    del dmm

    # Register the custom field in the SVAR metadata.
    meta_path = svar_copy / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["fields"] = {FIELD_NAME: FIELD_DTYPE}
    meta_path.write_text(json.dumps(meta))

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=source_bed, variants=svar_copy, overwrite=True)
    return gvl_path, FIELD_NAME, n_records


def test_available_var_fields_lists_custom_field(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
    assert field_name in ds.available_var_fields
    # Listed exactly once.
    assert ds.available_var_fields.count(field_name) == 1


def test_var_field_data_loaded_only_when_requested(custom_field_ds, ref_fasta):
    gvl_path, field_name, n_records = custom_field_ds

    # Not requested (default var_fields) -> not memmapped, not in info dict.
    ds = gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
    haps = ds._seqs  # type: ignore[attr-defined]
    assert field_name not in haps.var_field_data
    assert field_name not in haps.variants.info  # not loaded as an INFO column

    # Requested -> memmapped into var_field_data with the registered dtype.
    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", field_name])
    haps2 = ds2._seqs  # type: ignore[attr-defined]
    assert field_name in haps2.var_field_data
    rag = haps2.var_field_data[field_name]
    assert np.asarray(rag.data).dtype == np.dtype(FIELD_DTYPE)
    assert np.asarray(rag.data).shape[0] == n_records
    # Still must not have been loaded as an INFO column.
    assert field_name not in haps2.variants.info


def _open_variants(gvl_path, ref_fasta, field_name, **settings):
    # "AF" must be in var_fields so it is eagerly loaded for AF-filter tests.
    return (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", "dosage", "AF", field_name],
                       **settings)
    )


def test_custom_field_present_in_ragged_variants(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name)
    batch = ds[0, ds.samples[0]]
    assert field_name in batch.fields
    # dtype is the registered int16, not coerced.
    flat = ak.to_numpy(ak.flatten(batch[field_name], axis=None))
    assert flat.dtype == np.dtype(FIELD_DTYPE)
    # Per-cell variant counts equal `start`'s (call-aligned with the genotypes).
    assert ak.num(batch[field_name], -1).to_list() == ak.num(batch["start"], -1).to_list()


def test_custom_field_matches_dosage_gather(custom_field_ds, ref_fasta):
    """Custom field and dosages were written with identical arange values and
    share the genotype offsets, so the gathered output must be elementwise equal."""
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name)
    batch = ds[0, ds.samples[0]]
    custom = ak.to_numpy(ak.flatten(batch[field_name], axis=None)).astype(np.float64)
    dosage = ak.to_numpy(ak.flatten(batch["dosage"], axis=None)).astype(np.float64)
    np.testing.assert_array_equal(custom, dosage)


def test_custom_field_af_compaction_matches_dosage(custom_field_ds, ref_fasta):
    """Under AF filtering the custom field is compacted with the SAME keep mask
    as dosage, so they stay elementwise equal."""
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name, max_af=0.5)
    batch = ds[0, ds.samples[0]]
    custom = ak.to_numpy(ak.flatten(batch[field_name], axis=None)).astype(np.float64)
    dosage = ak.to_numpy(ak.flatten(batch["dosage"], axis=None)).astype(np.float64)
    np.testing.assert_array_equal(custom, dosage)


def test_custom_field_present_in_flat_mode(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", field_name])
        .with_output_format("flat")
    )
    flat = ds[0, ds.samples[0]]  # _FlatVariants
    assert field_name in flat.fields
    # Re-wrapping to ragged keeps the field.
    assert field_name in flat.to_ragged().fields


def test_custom_field_present_in_variant_windows(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    opt = gvl.VarWindowOpt(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
    ds = (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variant-windows", opt)
        .with_settings(var_fields=["alt", "ilen", "start", field_name])
        .with_output_format("flat")
    )
    batch = ds[0, ds.samples[0]]
    assert field_name in batch.fields
