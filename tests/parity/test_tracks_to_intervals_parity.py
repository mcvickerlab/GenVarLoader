"""Parity tests for tracks_to_intervals (RLE encoder, batch kernel)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from genvarloader._dataset import _intervals  # noqa: F401 — triggers register()
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import tracks_to_intervals_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None, max_examples=500)
@given(tracks_to_intervals_inputs())
def test_tracks_to_intervals_parity(args):
    """Numba and Rust produce byte-identical (starts, ends, values, offsets)."""
    regions, tracks, track_offsets = args
    assert_kernel_parity_tuple("tracks_to_intervals", regions, tracks, track_offsets)
