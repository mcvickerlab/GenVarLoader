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
