import polars as pl
import pytest

from genvarloader._dataset._write import _reject_unsupported_variants


def _index(alts, refs=None):
    """Build a minimal genoray-style index frame from a list of ALT lists."""
    n = len(alts)
    refs = refs if refs is not None else ["A"] * n
    return pl.DataFrame(
        {
            "CHROM": ["chr1"] * n,
            "POS": list(range(1, n + 1)),
            "REF": refs,
            "ALT": alts,
            "ILEN": [[0]] * n,
        },
        schema_overrides={"ALT": pl.List(pl.Utf8), "ILEN": pl.List(pl.Int32)},
    )


def test_clean_index_passes():
    # all bi-allelic SNPs: no raise
    _reject_unsupported_variants(_index([["T"], ["C"], ["G"]]), "VCF")


def test_symbolic_is_rejected():
    idx = _index([["T"], ["<DEL>"]])
    with pytest.raises(ValueError, match="symbolic"):
        _reject_unsupported_variants(idx, "VCF")


def test_breakend_is_rejected():
    idx = _index([["T"], ["C[chr1:600["]])
    with pytest.raises(ValueError, match="breakend"):
        _reject_unsupported_variants(idx, "PGEN")


def test_multiallelic_is_rejected():
    idx = _index([["T", "C"]])
    with pytest.raises(ValueError, match="multi-allelic"):
        _reject_unsupported_variants(idx, "SVAR")


def test_error_reports_source_and_counts():
    idx = _index([["<DEL>"], ["<INS>"], ["C[chr1:600["]])
    with pytest.raises(ValueError) as exc:
        _reject_unsupported_variants(idx, "SVAR")
    msg = str(exc.value)
    assert "SVAR" in msg
    assert "2 symbolic" in msg
    assert "1 breakend" in msg
