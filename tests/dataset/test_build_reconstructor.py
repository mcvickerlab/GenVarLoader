"""Unit tests for `_build_reconstructor` factory.

These exercise the factory in isolation, without spinning up a full Dataset.
Synthetic sources are constructed via Mocks. The goal is parity with the
construction logic that previously lived inline in `Dataset.open`, `with_seqs`,
etc.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from genvarloader._dataset._reconstruct import (
    Haps,
    HapsTracks,
    Ref,
    RefTracks,
    Tracks,
    _build_reconstructor,
)


def _haps_mock() -> Haps:
    return Mock(spec=Haps)


def _ref_mock() -> Ref:
    return Mock(spec=Ref)


def _tracks_mock() -> Tracks:
    return Mock(spec=Tracks)


def test_haps_only_returns_haps():
    seqs = _haps_mock()
    result = _build_reconstructor(seqs, None)
    assert result is seqs


def test_ref_only_returns_ref():
    seqs = _ref_mock()
    result = _build_reconstructor(seqs, None)
    assert result is seqs


def test_tracks_only_returns_tracks():
    tracks = _tracks_mock()
    result = _build_reconstructor(None, tracks)
    assert result is tracks


def test_haps_and_tracks_returns_haps_tracks():
    seqs = _haps_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks)
    assert isinstance(result, HapsTracks)
    assert result.haps is seqs
    assert result.tracks is tracks


def test_ref_and_tracks_returns_ref_tracks():
    seqs = _ref_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks)
    assert isinstance(result, RefTracks)
    assert result.seqs is seqs
    assert result.tracks is tracks


def test_neither_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        _build_reconstructor(None, None)
