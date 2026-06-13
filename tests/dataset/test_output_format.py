"""Unit tests for Dataset.output_format field and with_output_format method."""

import pytest


def test_with_output_format_sets_field(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    assert ds.output_format == "ragged"
    flat = ds.with_output_format("flat")
    assert flat.output_format == "flat"
    # original is unchanged (frozen dataclass / replace semantics)
    assert ds.output_format == "ragged"


def test_with_output_format_rejects_bad_value(snap_dataset):
    with pytest.raises(ValueError):
        snap_dataset.with_output_format("nope")
