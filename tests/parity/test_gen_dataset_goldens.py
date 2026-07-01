"""Dataset-level golden generator for the parity suite.

Run with GVL_GEN_GOLDENS=1 to regenerate all dataset goldens:

    GVL_GEN_GOLDENS=1 pixi run -e dev pytest tests/parity/test_gen_dataset_goldens.py -q --basetemp=$(pwd)/.pytest_tmp

Each test:
  1. Builds the SAME dataset the corresponding parity test uses (identical fixtures).
  2. Reads ds[idx] under numba then rust (GVL_BACKEND env flip — gen time only).
  3. HARD-FAILS on any numba != rust mismatch (oracle cross-check).
  4. Saves the rust output as a frozen golden.

Normal test runs skip all tests in this file.

*** DANGER (post-W5): numba was DELETED in W5, so the GVL_BACKEND flip + oracle
cross-check (steps 2-3) no longer fire. Regenerating now would freeze rust == rust
with no oracle — meaningless goldens. Only regenerate on a numba-PRESENT checkout
(at or before the Stage-A snapshot). ***
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest
from dataclasses import replace

import genvarloader as gvl
import genvarloader._dataset._genotypes  # noqa: F401 — trigger register()
import genvarloader._dataset._flat_variants  # noqa: F401
import genvarloader._dataset._reference  # noqa: F401
import genvarloader._dataset._tracks  # noqa: F401
from genvarloader import VarWindowOpt

from tests.parity import _golden
from tests.parity._fixtures import (
    build_haps_tracks_dataset,
    build_strand_mixed_dataset,
    build_track_dataset,
    build_track_dataset_jittered,
)

pytestmark = pytest.mark.parity

GEN = os.environ.get("GVL_GEN_GOLDENS") == "1"
skip_unless_gen = pytest.mark.skipif(
    not GEN, reason="set GVL_GEN_GOLDENS=1 to generate"
)


def _oracle_check(out_numba, out_rust, name: str) -> None:
    """HARD-FAIL if numba output differs from rust output. No suppression."""
    flat_n = _golden.flatten_output(out_numba)
    flat_r = _golden.flatten_output(out_rust)
    _golden._assert_flat_eq(flat_n, flat_r, f"oracle/{name}")


def _gen(name: str, monkeypatch, build_fn):
    """Build dataset, read under numba then rust, oracle-check, save golden."""
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = build_fn()
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = build_fn()
    _oracle_check(out_numba, out_rust, name)
    _golden.save_flat_golden(name, out_rust)


# ---------------------------------------------------------------------------
# Haplotypes-mode (non-splice) and fused-haps — share ds_haplotypes_mode
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_haplotypes_mode(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_haplotypes_mode: phased_svar_gvl + reference, haplotypes mode."""
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference).with_seqs("haplotypes")
    _gen("ds_haplotypes_mode", monkeypatch, lambda: ds[:, :])


@skip_unless_gen
def test_gen_annotated_mode(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_annotated_mode: annotated mode."""
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference).with_seqs("annotated")
    _gen("ds_annotated_mode", monkeypatch, lambda: ds[:, :])


@skip_unless_gen
def test_gen_haps_fixed_len(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_haps_fixed_len: haplotypes mode with with_len(15)."""
    FIXED_LEN = 15
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("haplotypes")
        .with_len(FIXED_LEN)
    )
    _gen("ds_haps_fixed_len", monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Spliced haplotypes
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_spliced_haps(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_spliced_haps: haplotypes + splice (T1=[0,1], T2=[2,3])."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("haplotypes")
        .with_tracks(False)
    )
    n = 4
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"])
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced
    _gen("ds_spliced_haps", monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Annotated spliced haplotypes
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_annotated_spliced(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_annotated_spliced: annotated + spliced with mixed strands."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("annotated")
        .with_tracks(False)
    )
    n = 4
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"]),
        pl.Series("strand", ["+", "+", "-", "-"]),
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced
    _gen("ds_annotated_spliced", monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Track-only datasets
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_tracks(tmp_path, monkeypatch):
    """Generates ds_tracks: track-only dataset, signal track."""
    ds_dir = build_track_dataset(tmp_path)
    ds = gvl.Dataset.open(ds_dir).with_tracks("signal")
    _gen("ds_tracks", monkeypatch, lambda: ds[slice(None), slice(None)])


@skip_unless_gen
def test_gen_tracks_jitter(tmp_path, monkeypatch):
    """Generates ds_tracks_jitter: jittered track dataset (max_jitter=4)."""
    MAX_JITTER = 4
    ds_dir = build_track_dataset_jittered(tmp_path, max_jitter=MAX_JITTER)
    ds = gvl.Dataset.open(ds_dir).with_tracks("signal")
    _gen("ds_tracks_jitter", monkeypatch, lambda: ds[slice(None), slice(None)])


# ---------------------------------------------------------------------------
# Haps+tracks (5 fill strategies) — shared by test_dataset_parity and test_fused_tracks_parity
# ---------------------------------------------------------------------------


@skip_unless_gen
@pytest.mark.parametrize(
    "strategy_name",
    [
        "Repeat5p",
        "Repeat5pNormalized",
        "Constant",
        "FlankSample",
        "Interpolate",
    ],
)
def test_gen_haps_tracks(strategy_name, tmp_path, synthetic_case, monkeypatch):
    """Generates ds_haps_tracks_{strategy}: haps+tracks with each fill strategy."""
    from genvarloader._dataset._insertion_fill import (
        Constant,
        FlankSample,
        Interpolate,
        Repeat5p,
        Repeat5pNormalized,
    )

    strat_map = {
        "Repeat5p": Repeat5p(),
        "Repeat5pNormalized": Repeat5pNormalized(),
        "Constant": Constant(0.0),
        "FlankSample": FlankSample(flank_width=5),
        "Interpolate": Interpolate(order=1),
    }
    fill = strat_map[strategy_name]
    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref)
        .with_seqs("haplotypes")
        .with_tracks("signal")
        .with_insertion_fill(fill)
    )
    golden_name = f"ds_haps_tracks_{strategy_name}"
    _gen(golden_name, monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Reference mode
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_reference_mode(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_reference_mode: reference mode on phased_svar_gvl."""
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference).with_seqs("reference")
    _gen("ds_reference_mode", monkeypatch, lambda: ds[:, :])


@skip_unless_gen
def test_gen_reference_fetch(reference, monkeypatch):
    """Generates ds_reference_fetch: Reference.fetch(contigs[:1], [0], [50])."""
    contigs = reference.contigs[:1]
    starts = np.array([0], dtype=np.int64)
    ends = np.array([50], dtype=np.int64)
    _gen(
        "ds_reference_fetch",
        monkeypatch,
        lambda: reference.fetch(contigs, starts, ends),
    )


# ---------------------------------------------------------------------------
# Variants mode
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_variants(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_variants: variants mode (RaggedVariants)."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_tracks(False)
        .with_seqs("variants")
    )
    _gen("ds_variants", monkeypatch, lambda: ds[:, :])


@skip_unless_gen
def test_gen_variants_af(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_variants_af: variants with AF filter (skips if AF unavailable)."""
    ds_base = gvl.Dataset.open(phased_svar_gvl, reference=reference).with_tracks(False)
    try:
        ds = ds_base.with_seqs("variants").with_settings(min_af=0.1, max_af=0.9)
    except Exception as e:
        pytest.skip(f"AF filtering unavailable: {e}")
        raise  # unreachable (skip raises); tells pyrefly this branch is NoReturn
    try:
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]
    except KeyError as e:
        pytest.skip(f"AF key missing: {e}")
        raise  # unreachable; NoReturn marker for pyrefly
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    _oracle_check(out_numba, out_rust, "ds_variants_af")
    _golden.save_flat_golden("ds_variants_af", out_rust)


@skip_unless_gen
def test_gen_variant_windows(phased_svar_gvl, reference, monkeypatch):
    """Generates ds_variant_windows: variant-windows mode (_FlatVariantWindows)."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    _gen("ds_variant_windows", monkeypatch, lambda: ds[[0, 1], [0, 1]])


# ---------------------------------------------------------------------------
# Neg-strand parity (6 kinds, unspliced)
# ---------------------------------------------------------------------------

_NEG_STRAND_KINDS = [
    "reference",
    "haplotypes",
    "annotated",
    "tracks",
    "tracks-seqs",
    "haps-tracks",
]


@skip_unless_gen
@pytest.mark.parametrize("kind", _NEG_STRAND_KINDS)
def test_gen_neg_strand(kind, tmp_path, synthetic_case, monkeypatch):
    """Generates ds_neg_strand_{kind}: mixed +/- strand regions."""
    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)

    if kind == "tracks":
        ds = gvl.Dataset.open(ds_dir).with_seqs(None).with_tracks("signal")
    elif kind == "tracks-seqs":
        ds = (
            gvl.Dataset.open(ds_dir, reference=ref)
            .with_seqs("reference")
            .with_tracks("signal")
        )
    elif kind == "haps-tracks":
        ds = (
            gvl.Dataset.open(ds_dir, reference=ref)
            .with_seqs("haplotypes")
            .with_tracks("signal")
        )
    else:
        ds = gvl.Dataset.open(ds_dir, reference=ref).with_seqs(kind).with_tracks(False)

    safe_kind = kind.replace("-", "_")
    _gen(f"ds_neg_strand_{safe_kind}", monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Neg-strand SPLICED parity (4 kinds)
# ---------------------------------------------------------------------------

_SPLICE_TRANSCRIPT_IDS = ["T1", "T2", "T3", "T3", "T4"]
_NEG_SPLICED_KINDS = ["reference", "haplotypes", "annotated", "tracks"]


def _open_strand_spliced(ds_dir, ref, kind: str):
    if kind == "tracks":
        ds = gvl.Dataset.open(ds_dir).with_seqs(None).with_tracks("signal")
    else:
        ds = gvl.Dataset.open(ds_dir, reference=ref).with_seqs(kind).with_tracks(False)
    sub_bed = ds._full_bed.with_columns(
        pl.Series("transcript_id", _SPLICE_TRANSCRIPT_IDS)
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced
    return ds


@skip_unless_gen
@pytest.mark.parametrize("kind", _NEG_SPLICED_KINDS)
def test_gen_neg_strand_spliced(kind, tmp_path, synthetic_case, monkeypatch):
    """Generates ds_neg_strand_spliced_{kind}: spliced mixed +/- strand."""
    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = _open_strand_spliced(ds_dir, ref, kind)
    _gen(f"ds_neg_strand_spliced_{kind}", monkeypatch, lambda: ds[:, :])


# ---------------------------------------------------------------------------
# Neg-strand variants
# ---------------------------------------------------------------------------


@skip_unless_gen
def test_gen_neg_strand_variants(tmp_path, synthetic_case, monkeypatch):
    """Generates ds_neg_strand_variants: variants on mixed-strand dataset."""
    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref).with_tracks(False).with_seqs("variants")
    )
    _gen("ds_neg_strand_variants", monkeypatch, lambda: ds[:, :])


@skip_unless_gen
def test_gen_neg_strand_variants_dummy(tmp_path, synthetic_case, monkeypatch):
    """Generates ds_neg_strand_variants_dummy: variants with custom DummyVariant."""
    from genvarloader._dataset._flat_variants import DummyVariant

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref)
        .with_tracks(False)
        .with_seqs("variants")
        .with_settings(dummy_variant=DummyVariant(alt=b"AC", ref=b"AC"))
    )
    _gen("ds_neg_strand_variants_dummy", monkeypatch, lambda: ds[:, :])
