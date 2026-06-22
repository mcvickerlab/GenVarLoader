"""Filter composition through ``Dataset.__getitem__`` / ``_query.getitem``.

``_query.py`` is only reachable via the dataset query path, so these tests
exercise filter combinations end-to-end:

- AF range on ``with_seqs("variants")`` mode (min_af / max_af / both)
- Empty-result AF range (degenerate output)
- ``var_filter="exonic"`` on haplotype mode (smoke through ``_query``)
- Sample subset via ``subset_to``
- Composition: subset + AF filter
- ``rc_neg=True`` toggling exercising ``reverse_complement_ragged``
"""

from __future__ import annotations

import awkward as ak
import genvarloader as gvl
import numpy as np
import pytest


@pytest.fixture(scope="module")
def svar_ds_path(tmp_path_factory, filtered_svar, source_bed):
    """Build a small SVAR-backed GVL store under a per-module tmp dir.

    SVAR is required for AF filtering (info["AF"] is populated).
    """
    assert filtered_svar.is_dir(), (
        f"missing fixture {filtered_svar}; run pixi run -e dev gen"
    )
    out = tmp_path_factory.mktemp("query_filters") / "ds.gvl"
    gvl.write(path=out, bed=source_bed, variants=filtered_svar, overwrite=True)
    return out


@pytest.fixture(scope="module")
def vars_ds(svar_ds_path, ref_fasta):
    """Variants-mode dataset (the AF-filter path uses this output mode)."""
    return (
        gvl.Dataset.open(svar_ds_path, reference=ref_fasta)
        .with_seqs("variants")
        .with_len("ragged")
        # AF info isn't loaded eagerly; AF filtering reads variants.info["AF"].
        .with_settings(var_fields=["alt", "ilen", "start", "AF"])
    )


@pytest.fixture(scope="module")
def haps_ds(svar_ds_path, ref_fasta):
    return (
        gvl.Dataset.open(svar_ds_path, reference=ref_fasta)
        .with_seqs("haplotypes")
        .with_len("ragged")
    )


def _n_variants(rv) -> int:
    """Total scalar variant count across a RaggedVariants result."""
    return int(ak.sum(ak.count(rv.alt, axis=None)))


def test_no_filter_yields_variants(vars_ds):
    """Sanity: unfiltered query returns the full variant set."""
    rv = vars_ds[:, :]
    assert _n_variants(rv) > 0


def test_af_min_only_reduces_or_preserves(vars_ds):
    """min_af filter never increases variant count vs unfiltered."""
    n_full = _n_variants(vars_ds[:, :])
    rv = vars_ds.with_settings(min_af=0.3)[:, :]
    assert _n_variants(rv) <= n_full


def test_af_range_excludes_outside_band(vars_ds):
    """A narrow AF band yields fewer variants than the full set."""
    n_full = _n_variants(vars_ds[:, :])
    rv = vars_ds.with_settings(min_af=0.3, max_af=0.6)[:, :]
    n_band = _n_variants(rv)
    # Test fixture AFs are in {0.167, 0.333, 0.5, 0.667}; band {0.333, 0.5}
    # strictly drops 0.167 and 0.667.
    assert 0 < n_band < n_full


def test_af_empty_result_degenerates(vars_ds):
    """An AF range that excludes every variant yields zero variants per cell.

    Exercises the empty-filter-result path: ``_get_variants`` runs the
    ``keep`` mask, the resulting Ragged has empty inner lists, and
    ``_query.getitem`` still returns a shape-correct RaggedVariants.
    """
    rv = vars_ds.with_settings(min_af=0.99, max_af=1.0)[:, :]
    assert _n_variants(rv) == 0
    # Outer shape preserved: (n_regions, n_samples, ploidy, *var).
    assert rv.alt.ndim == 4


def test_exonic_filter_haps_runs_through_query(haps_ds):
    """``var_filter='exonic'`` reaches the haplotype reconstructor via _query.

    The toy fixture has no exonic variants, so per-haplotype lengths must
    match the unfiltered output (filter has no effect on lengths here).
    The point is to exercise the filter wiring through ``getitem``.
    """
    h0 = haps_ds[:, :]
    h1 = haps_ds.with_settings(var_filter="exonic")[:, :]
    assert np.array_equal(
        np.asarray(ak.num(h0.to_ak(), axis=-1)),
        np.asarray(ak.num(h1.to_ak(), axis=-1)),
    )


def test_subset_to_samples_reduces_outer_shape(vars_ds):
    """``subset_to(samples=...)`` changes the outer (n_samples) axis."""
    one_sample = [vars_ds.samples[0]]
    sub = vars_ds.subset_to(samples=one_sample)
    rv = sub[:, :]
    # Outer shape: (n_regions, n_samples_sub, ploidy, *var)
    assert rv.alt.type.length == vars_ds.n_regions
    # Inner sample axis collapsed to 1.
    counts = ak.num(rv.alt, axis=1)
    assert ak.all(counts == 1)


def test_compose_subset_and_af_filter(vars_ds):
    """subset_to(samples) and min_af compose: count is bounded by both."""
    one_sample = [vars_ds.samples[0]]
    n_one_sample = _n_variants(vars_ds.subset_to(samples=one_sample)[:, :])
    n_af_only = _n_variants(vars_ds.with_settings(min_af=0.3)[:, :])

    composed = vars_ds.subset_to(samples=one_sample).with_settings(min_af=0.3)[:, :]
    n_composed = _n_variants(composed)
    # Composition is at most as large as either filter applied alone.
    assert n_composed <= n_one_sample
    assert n_composed <= n_af_only


def test_rc_neg_preserves_variant_count(vars_ds):
    """rc_neg toggles reverse_complement_ragged in _query without losing data."""
    n_no_rc = _n_variants(vars_ds.with_settings(rc_neg=False)[:, :])
    n_rc = _n_variants(vars_ds.with_settings(rc_neg=True)[:, :])
    assert n_no_rc == n_rc
