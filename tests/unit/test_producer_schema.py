"""_apply_schema reconstructs variant-windows and flank configs in the child.

Covers both the ``_apply_schema`` reconstruction side (schema dict -> Dataset)
and the ``_build_producer_schema`` emission side (Dataset -> schema dict),
including the full round trip through both for a mixed builder chain that
leaves a stale ``Haps.window_opt`` alongside an active flank config.
"""

import pytest

import genvarloader as gvl
from genvarloader._dataset._flat_flanks import build_token_lut
from genvarloader._dataset._flat_variants import VarWindowOpt
from genvarloader._double_buffered_loader import (
    _build_producer_schema,
    _token_alphabet_from_lut,
)
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


def test_token_alphabet_from_lut_roundtrips_for_normal_config():
    lut, _ = build_token_lut(b"ACGT", 4)
    assert _token_alphabet_from_lut(lut, 4) == b"ACGT"


def test_token_alphabet_from_lut_accepts_boundary_unknown_token():
    """unknown_token == len(alphabet) - 1 collides with the last symbol's own
    token id (e.g. "T"'s id 3 in b"ACGT"), so the recovered alphabet text
    ("ACG") differs from the original -- but a byte assigned that token id
    behaves identically whether treated as "in the alphabet at that position"
    or "unknown", so the rebuilt LUT is byte-identical and this must NOT
    raise (this is the case a naive contiguity check gets wrong).
    """
    lut, _ = build_token_lut(b"ACGT", 3)
    alphabet = _token_alphabet_from_lut(lut, 3)
    rebuilt_lut, _ = build_token_lut(alphabet, 3)
    assert (rebuilt_lut == lut).all()


def test_token_alphabet_from_lut_raises_on_colliding_unknown_token():
    # unknown_token=0 collides with "A"'s token id (0) in b"ACGT" in a way
    # that loses information: both "A" and true out-of-alphabet bytes map to
    # 0, and no alphabet reconstructed from the LUT rebuilds it faithfully.
    lut, _ = build_token_lut(b"ACGT", 0)
    with pytest.raises(ValueError, match="collides"):
        _token_alphabet_from_lut(lut, 0)
