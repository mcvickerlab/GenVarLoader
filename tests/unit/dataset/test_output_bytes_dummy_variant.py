"""Direct byte-accounting bound tests for dummy_variant.

Pins the fix in ``Dataset._output_bytes_per_instance`` for the case where a
(region, sample, ploid) group gets dummy-filled by
``_FlatVariants.fill_empty_groups`` / ``_FlatVariantWindows.fill_empty_groups``
-- either because it has 0 real on-disk variants, or (the regression this
file targets) because ``min_af``/``max_af`` filtering removes all of that
group's real variants even though it had some on-disk. In the latter case,
the group is NOT raw-empty (``Dataset.n_variants()`` reports > 0), so any
accounting that decides "does this group need dummy-row bytes?" purely from
raw on-disk counts under-counts it.

The invariant under test mirrors exactly what sizes a ``double_buffered``
shared-memory slot (see ``_chunked.ChunkPlanner`` / ``_shm_layout.write_chunk``):

    _output_bytes_per_instance(..., include_offsets=True).sum() + HEADER_RESERVED
    >= actual bytes write_chunk(...) serializes

This is checked directly (no subprocess, no producer) against the exact same
``write_chunk`` the ``double_buffered`` producer calls, so it fails loudly and
fast if the estimate ever under-shoots -- independent of whether a given
fixture's real/estimated gap happens to be smaller than the shm slot's fixed
slack (which is why the ``mode="double_buffered"`` slot-fit tests in
``tests/unit/test_double_buffered_loader.py`` cannot pin this on their own).
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np

import genvarloader as gvl
from genvarloader._shm_layout import HEADER_RESERVED, write_chunk

# "start" (variant position) is unconditionally emitted by the flat builder
# regardless of the requested var_fields, so every ds in this file includes it
# explicitly -- omitting it here would not remove it from the real output,
# only from what _output_bytes_per_instance's per-field loop charges for it
# (a separate, pre-existing gap unrelated to dummy_variant; see the
# AF-filter tests' docstrings for why that matters for test design here).
_VAR_FIELDS = ["alt", "start", "AF"]


def _dummy_variant(long: bool = False) -> "gvl.DummyVariant":
    allele = b"N" * 20 if long else b"N"
    return gvl.DummyVariant(start=-1, ilen=0, ref=allele, alt=allele)


def _assert_estimate_covers_actual(ds: "gvl.Dataset") -> None:
    """``_output_bytes_per_instance(include_offsets=True)`` must upper-bound the real serialized chunk size."""
    n_inst = ds.shape[0] * ds.shape[1]
    r_idx, s_idx = np.unravel_index(np.arange(n_inst), ds.shape)

    bpi_off = ds._output_bytes_per_instance(include_offsets=True)
    estimated = int(np.asarray(bpi_off).sum()) + HEADER_RESERVED

    chunk = ds[r_idx, s_idx]
    arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]

    # Generously oversized so write_chunk itself never fails for lack of
    # room -- the assertion below is what actually pins the byte budget, not
    # the shm slot size.
    shm = SharedMemory(create=True, size=(1 << 22))
    try:
        actual = write_chunk(shm.buf, arrays, n_instances=n_inst)
    finally:
        shm.close()
        shm.unlink()

    assert estimated >= actual, (
        f"_output_bytes_per_instance under-estimates the double_buffered slot: "
        f"estimated={estimated} < actual={actual} (would raise ProducerError "
        f"'buffer is smaller than requested size' in mode='double_buffered')"
    )


def test_variant_windows_dummy_variant_bound(phased_svar_gvl, reference):
    """dummy_variant + variant-windows, no AF filter (some groups are raw-empty)."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(dummy_variant=_dummy_variant())
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    _assert_estimate_covers_actual(ds)


def test_variants_flank_tokens_dummy_variant_bound(phased_svar_gvl, reference):
    """dummy_variant + flat variants + ride-along flank_tokens, no AF filter."""
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(
            dummy_variant=_dummy_variant(),
            flank_length=2,
            token_alphabet=b"ACGT",
            unknown_token=4,
        )
        .with_output_format("flat")
    )
    _assert_estimate_covers_actual(ds)


def test_variant_windows_af_filter_dummy_variant_bound(phased_svar_gvl, reference):
    """Regression: AF filtering can dummy-fill a group that had real on-disk variants.

    ``max_af=0.001`` is below every variant's AF in this fixture (min observed
    AF is ~0.167), so AF filtering empties every originally-nonzero
    (region, sample, ploid) group -- 33 of them in this fixture -- none of
    which a raw-on-disk-count-based accounting (``n_variants() == 0``) would
    recognize as needing dummy-row bytes, since their *raw* count is nonzero.

    A large ``flank_length`` and a long dummy allele are used deliberately:
    ``_output_bytes_per_instance``'s scalar-field terms (``start``, ``AF``)
    are charged from the *raw* on-disk variant count regardless of AF
    filtering -- an unrelated, pre-existing imprecision that happens to
    *over*-estimate under AF filtering (it still charges for variants that
    got filtered out). With a small/default dummy allele and flank_length,
    that pre-existing over-charge is large enough to mask the (smaller)
    dummy-fill deficit this test targets, so a naive version of this test
    does not actually go red on pre-fix code. Scaling up the dummy
    row's own footprint (long allele, large flank) pushes its deficit past
    that masking margin and reproduces the failure this test pins: without
    the ploidy-upper-bound fix in ``_output_bytes_per_instance``, this
    configuration under-estimates by several hundred bytes for this fixture.
    """
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(
            var_fields=_VAR_FIELDS,
            max_af=0.001,
            dummy_variant=_dummy_variant(long=True),
        )
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=20, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    _assert_estimate_covers_actual(ds)


def test_variants_af_filter_dummy_variant_bound(phased_svar_gvl, reference):
    """Same AF-filter + dummy_variant regression, for flat variants.

    See ``test_variant_windows_af_filter_dummy_variant_bound`` for why a long
    dummy allele is used: it pushes the allele-byte dummy-fill deficit past
    the unrelated, pre-existing raw-count-based over-charge on the scalar
    fields (``start``/``AF``) that would otherwise mask it.
    """
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(
            var_fields=_VAR_FIELDS,
            max_af=0.001,
            dummy_variant=_dummy_variant(long=True),
        )
        .with_output_format("flat")
    )
    _assert_estimate_covers_actual(ds)


def test_variants_af_filter_flank_tokens_dummy_variant_bound(
    phased_svar_gvl, reference
):
    """AF-filter + dummy_variant regression with flank_tokens also active.

    Same isolating long-dummy-allele config as the other AF-filter tests,
    with ride-along ``flank_tokens`` (Config B's realistic shape) layered on
    top.
    """
    ds = (
        gvl.Dataset.open(phased_svar_gvl, reference=reference)
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(
            var_fields=_VAR_FIELDS,
            max_af=0.001,
            dummy_variant=_dummy_variant(long=True),
            flank_length=10,
            token_alphabet=b"ACGT",
            unknown_token=4,
        )
        .with_output_format("flat")
    )
    _assert_estimate_covers_actual(ds)
