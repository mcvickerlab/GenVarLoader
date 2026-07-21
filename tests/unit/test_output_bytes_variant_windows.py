"""Byte-accounting must handle variant-windows and variants+flank_tokens."""

import numpy as np
import pytest

import genvarloader as gvl


def _vw_ds(ref="window", alt="window"):
    base = gvl.get_dummy_dataset().with_tracks(False).with_output_format("flat")
    if ref == "allele" and base._seqs.variants.ref is None:
        # Matches the skip convention in test_output_bytes_per_instance.py's
        # test_variants_with_ref_exact: the dummy dataset has no REF alleles.
        pytest.skip("dummy dataset does not have ref allele")
    return base.with_seqs(
        "variant-windows",
        gvl.VarWindowOpt(
            flank_length=3, token_alphabet=b"ACGT", unknown_token=4, ref=ref, alt=alt
        ),
    )


@pytest.mark.parametrize(
    "ref,alt", [("window", "window"), ("window", "allele"), ("allele", "allele")]
)
def test_variant_windows_bytes_positive(ref, alt):
    ds = _vw_ds(ref, alt)
    bpi = ds._output_bytes_per_instance(None, None)
    assert bpi.shape == ds.shape and bpi.dtype == np.int64
    assert (bpi >= 0).all() and bpi.sum() > 0
    # include_offsets must be >= payload-only (adds offset overhead).
    bpi_off = ds._output_bytes_per_instance(None, None, include_offsets=True)
    assert (bpi_off >= bpi).all() and bpi_off.sum() > bpi.sum()


def test_flank_tokens_adds_bytes():
    base = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
    )
    with_flank = base.with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=0
    )
    b0 = base._output_bytes_per_instance(None, None).sum()
    b1 = with_flank._output_bytes_per_instance(None, None).sum()
    assert b1 > b0  # flank tokens are extra payload
