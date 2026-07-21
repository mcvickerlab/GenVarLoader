"""_apply_schema reconstructs variant-windows and flank configs in the child.

Covers both the ``_apply_schema`` reconstruction side (schema dict -> Dataset)
and the ``_build_producer_schema`` emission side (Dataset -> schema dict),
including the full round trip through both for a mixed builder chain that
leaves a stale ``Haps.window_opt`` alongside an active flank config.
"""

import genvarloader as gvl
from genvarloader._dataset._flat_flanks import build_token_lut
from genvarloader._dataset._flat_variants import VarWindowOpt
from genvarloader._double_buffered_loader import _build_producer_schema
from genvarloader._producer import _apply_schema


def test_apply_schema_rebuilds_variant_windows():
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    schema = {
        "with_seqs": "variant-windows",
        "output_format": "flat",
        "window_opt": {
            "flank_length": 3,
            "token_alphabet": b"ACGT",
            "unknown_token": 4,
            "ref": "window",
            "alt": "allele",
        },
    }
    ds = _apply_schema(dummy, schema)
    assert ds.sequence_type == "variant-windows"
    opt = ds._seqs.window_opt
    assert opt.flank_length == 3
    assert opt.token_alphabet == b"ACGT"
    assert opt.unknown_token == 4
    assert opt.ref == "window"
    assert opt.alt == "allele"


def test_apply_schema_rebuilds_flank_tokens():
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    schema = {
        "with_seqs": "variants",
        "output_format": "flat",
        "flank_length": 2,
        "token_alphabet": b"ACGT",
        "unknown_token": 4,
    }
    ds = _apply_schema(dummy, schema)
    expected_lut, expected_dtype = build_token_lut(b"ACGT", 4)
    assert ds._seqs.flank_length == 2
    assert ds._seqs.unknown_token == 4
    assert ds._seqs.token_lut is not None
    assert ds._seqs.token_dtype == expected_dtype
    assert (ds._seqs.token_lut == expected_lut).all()


def test_build_producer_schema_variant_windows():
    """Config A: with_seqs('variant-windows', opt) emits a window_opt schema entry."""
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    opt = VarWindowOpt(
        flank_length=3,
        token_alphabet=b"ACGT",
        unknown_token=4,
        ref="window",
        alt="allele",
    )
    ds = dummy.with_seqs("variant-windows", opt).with_output_format("flat")

    schema = _build_producer_schema(ds)

    assert schema["with_seqs"] == "variant-windows"
    assert schema["window_opt"] == {
        "flank_length": 3,
        "token_alphabet": b"ACGT",
        "unknown_token": 4,
        "ref": "window",
        "alt": "allele",
    }
    assert "flank_length" not in schema

    rebuilt = _apply_schema(dummy, schema)
    assert rebuilt._seqs.window_opt == ds._seqs.window_opt


def test_build_producer_schema_flank_tokens():
    """Config B: with_seqs('variants') + with_settings(flank_length=...) emits flank fields."""
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    ds = (
        dummy.with_seqs("variants")
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=4)
        .with_output_format("flat")
    )

    schema = _build_producer_schema(ds)

    assert "window_opt" not in schema
    assert schema["flank_length"] == 2
    assert schema["token_alphabet"] == b"ACGT"
    assert schema["unknown_token"] == 4

    rebuilt = _apply_schema(dummy, schema)
    assert rebuilt._seqs.flank_length == ds._seqs.flank_length
    assert rebuilt._seqs.unknown_token == ds._seqs.unknown_token
    assert (rebuilt._seqs.token_lut == ds._seqs.token_lut).all()


def test_build_producer_schema_mixed_chain_prefers_active_config():
    """Regression: a stale Haps.window_opt must not shadow an active flank config.

    ``Haps.window_opt`` is never cleared when the builder chain switches away
    from ``with_seqs("variant-windows", ...)`` (see ``_impl.py``'s
    ``with_seqs``), so after ``.with_seqs("variant-windows", opt)
    .with_seqs("variants").with_settings(flank_length=..., ...)`` the dataset's
    *active* output is plain variants with flank tokens, but ``seqs.window_opt``
    is still non-None from the earlier call. The schema must reflect the
    active config (flank_length/token_alphabet/unknown_token), not the stale
    window_opt -- otherwise the producer subprocess silently drops the flank
    settings and reconstructs the wrong dataset.
    """
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    opt = VarWindowOpt(flank_length=3, token_alphabet=b"ACGT", unknown_token=5)
    ds = (
        dummy.with_seqs("variant-windows", opt)
        .with_seqs("variants")
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=4)
        .with_output_format("flat")
    )
    # Sanity check the premise: window_opt is indeed still set (stale) while
    # flank_length/token_lut reflect the later, active with_settings call.
    assert ds._seqs.window_opt is not None
    assert ds._seqs.flank_length == 2

    schema = _build_producer_schema(ds)

    assert schema["with_seqs"] == "variants"
    assert "window_opt" not in schema, (
        "stale Haps.window_opt leaked into the schema and will shadow the "
        "active flank config in the child"
    )
    assert schema["flank_length"] == 2
    assert schema["token_alphabet"] == b"ACGT"
    assert schema["unknown_token"] == 4

    rebuilt = _apply_schema(dummy, schema)
    assert rebuilt.sequence_type == "variants"
    assert rebuilt._seqs.flank_length == 2
    assert rebuilt._seqs.unknown_token == 4
    assert (rebuilt._seqs.token_lut == ds._seqs.token_lut).all()


def test_build_producer_schema_flank_tokens_roundtrips_noncolliding_unknown_token():
    """Config B, unknown_token outside the alphabet's own token range: must round-trip exactly."""
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    ds = (
        dummy.with_seqs("variants")
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=4)
        .with_output_format("flat")
    )

    schema = _build_producer_schema(ds)

    assert schema["token_alphabet"] == b"ACGT"
    assert schema["unknown_token"] == 4

    rebuilt = _apply_schema(dummy, schema)
    assert rebuilt._seqs.flank_length == ds._seqs.flank_length
    assert rebuilt._seqs.unknown_token == ds._seqs.unknown_token
    assert (rebuilt._seqs.token_lut == ds._seqs.token_lut).all()


def test_build_producer_schema_flank_tokens_roundtrips_colliding_unknown_token():
    """Config B, unknown_token=0 collides with "A"'s own token id (0) in b"ACGT".

    This used to be recovered by inverting ``Haps.token_lut``
    (``_token_alphabet_from_lut``), which is provably lossy for exactly this
    collision and raised ``ValueError`` -- silently rejecting a legal,
    common config (``unknown_token=0`` as a natural pad/unknown id) that
    ``mode='buffered'``/``mode=None`` accept. Now that ``Haps`` stores the
    original ``token_alphabet`` bytes directly (no LUT inversion), this must
    round-trip exactly.
    """
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    ds = (
        dummy.with_seqs("variants")
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )

    schema = _build_producer_schema(ds)

    assert schema["token_alphabet"] == b"ACGT"
    assert schema["unknown_token"] == 0

    rebuilt = _apply_schema(dummy, schema)
    assert rebuilt._seqs.flank_length == ds._seqs.flank_length
    assert rebuilt._seqs.unknown_token == ds._seqs.unknown_token
    assert (rebuilt._seqs.token_lut == ds._seqs.token_lut).all()


def test_build_producer_schema_dummy_variant_roundtrips():
    """Regression: dummy_variant must be replayed by the double_buffered producer.

    ``_build_producer_schema``/``_apply_schema`` previously had no
    ``dummy_variant`` handling at all: the producer subprocess reopened the
    dataset and replayed every other setting, but silently dropped
    ``dummy_variant``, so double_buffered's output diverged from
    ``buffered``/``mode=None`` for any dataset with empty (region, sample,
    ploid) groups -- not a crash, a silent data-correctness bug. Must
    round-trip exactly for both ``variants`` and ``variant-windows``.
    """
    dummy_ds = gvl.get_dummy_dataset().with_tracks(False)
    dv = gvl.DummyVariant(start=-1, ilen=0, dosage=0.5, ref=b"N", alt=b"NN")
    ds = (
        dummy_ds.with_seqs("variants")
        .with_settings(dummy_variant=dv)
        .with_output_format("flat")
    )

    schema = _build_producer_schema(ds)

    assert schema["dummy_variant"] == {
        "start": -1,
        "ilen": 0,
        "dosage": 0.5,
        "ref": b"N",
        "alt": b"NN",
        "info": {},
    }

    rebuilt = _apply_schema(dummy_ds, schema)
    assert rebuilt._seqs.dummy_variant == dv


def test_build_producer_schema_no_dummy_variant_omits_key():
    """No dummy_variant set -> no "dummy_variant" key (mirrors the other optional fields)."""
    dummy_ds = gvl.get_dummy_dataset().with_tracks(False)
    ds = dummy_ds.with_seqs("variants").with_output_format("flat")

    schema = _build_producer_schema(ds)

    assert "dummy_variant" not in schema

    rebuilt = _apply_schema(dummy_ds, schema)
    assert rebuilt._seqs.dummy_variant is None


def test_build_producer_schema_unphased_union_roundtrips():
    """Regression: unphased_union must be replayed by the double_buffered producer.

    ``_build_producer_schema`` previously had no ``unphased_union`` handling at
    all: the parent process sizes shm slots from the folded ploidy-1 shape
    (see ``Dataset.n_variants``/``ploidy`` under the flag), but the producer
    subprocess reopened the dataset and replayed every other setting while
    silently dropping ``unphased_union`` -- so the child decoded at the
    on-disk ploidy and emitted mismatched rows. ``unphased_union`` is valid
    for both "variants" and "variant-windows" output, so this is checked for
    both.
    """
    dummy_ds = gvl.get_dummy_dataset().with_tracks(False)

    ds = (
        dummy_ds.with_seqs("variants")
        .with_settings(unphased_union=True)
        .with_output_format("flat")
    )
    schema = _build_producer_schema(ds)
    assert schema["unphased_union"] is True
    rebuilt = _apply_schema(dummy_ds, schema)
    assert rebuilt._seqs.unphased_union is True

    opt = VarWindowOpt(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
    ds_vw = (
        dummy_ds.with_seqs("variant-windows", opt)
        .with_settings(unphased_union=True)
        .with_output_format("flat")
    )
    schema_vw = _build_producer_schema(ds_vw)
    assert schema_vw["unphased_union"] is True
    rebuilt_vw = _apply_schema(dummy_ds, schema_vw)
    assert rebuilt_vw._seqs.unphased_union is True


def test_build_producer_schema_unphased_union_default_omits_key():
    """Default (unphased_union not opted in) -> no schema key, matching the
    other optional-field pattern; the child's default (False) is correct
    either way.
    """
    dummy_ds = gvl.get_dummy_dataset().with_tracks(False)
    ds = dummy_ds.with_seqs("variants").with_output_format("flat")

    schema = _build_producer_schema(ds)

    assert "unphased_union" not in schema

    rebuilt = _apply_schema(dummy_ds, schema)
    assert rebuilt._seqs.unphased_union is False
