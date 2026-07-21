"""_apply_schema reconstructs variant-windows and flank configs in the child."""

import genvarloader as gvl
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
    assert ds._seqs.window_opt.flank_length == 3
    assert ds._seqs.window_opt.alt == "allele"


def test_apply_schema_rebuilds_flank_tokens():
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    schema = {
        "with_seqs": "variants",
        "output_format": "flat",
        "flank_length": 2,
        "token_alphabet": b"ACGT",
        "unknown_token": 0,
    }
    ds = _apply_schema(dummy, schema)
    assert ds._seqs.flank_length == 2 and ds._seqs.token_lut is not None
