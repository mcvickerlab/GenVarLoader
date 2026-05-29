"""Unit tests for `_build_reconstructor` factory.

These exercise the factory in isolation, without spinning up a full Dataset.
Synthetic sources are constructed via Mocks. The goal is parity with the
construction logic that previously lived inline in `Dataset.open`, `with_seqs`,
etc.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from genvarloader._dataset._reconstruct import (
    Haps,
    HapsTracks,
    Ref,
    RefTracks,
    Tracks,
    _build_reconstructor,
)


def _haps_mock(with_reference: bool = True) -> Haps:
    m = Mock(spec=Haps)
    m.reference = Mock() if with_reference else None
    m.to_kind = Mock(return_value=m)  # to_kind returns same mock for routing tests
    return m


def _ref_mock() -> Ref:
    return Mock(spec=Ref)


def _tracks_mock() -> Tracks:
    m = Mock(spec=Tracks)
    m.active_tracks = ["dummy_track"]  # non-None signals "active"
    return m


def test_haps_only_returns_haps():
    seqs = _haps_mock()
    result = _build_reconstructor(seqs, None, "haplotypes")
    assert result is seqs


def test_ref_only_returns_ref():
    seqs = _ref_mock()
    result = _build_reconstructor(seqs, None, "reference")
    assert result is seqs


def test_tracks_only_returns_tracks():
    tracks = _tracks_mock()
    result = _build_reconstructor(None, tracks, None)
    assert result is tracks


def test_haps_and_tracks_returns_haps_tracks():
    seqs = _haps_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks, "haplotypes")
    assert isinstance(result, HapsTracks)
    assert result.haps is seqs
    assert result.tracks is tracks


def test_ref_and_tracks_returns_ref_tracks():
    seqs = _ref_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks, "reference")
    assert isinstance(result, RefTracks)
    assert result.seqs is seqs
    assert result.tracks is tracks


def test_neither_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        _build_reconstructor(None, None, None)


def test_haps_with_kind_reference_returns_ref():
    """When user wants reference view but storage is Haps, factory builds Ref."""
    seqs = _haps_mock(with_reference=True)
    result = _build_reconstructor(seqs, None, "reference")
    assert isinstance(result, Ref)


def test_haps_with_kind_reference_no_reference_raises():
    """When storage Haps has no reference genome, kind='reference' raises."""
    seqs = _haps_mock(with_reference=False)
    with pytest.raises(ValueError, match="no reference"):
        _build_reconstructor(seqs, None, "reference")


def test_ref_with_haps_kind_raises():
    """Cannot view Ref-only storage as haplotypes."""
    seqs = _ref_mock()
    with pytest.raises(ValueError, match="no haplotypes"):
        _build_reconstructor(seqs, None, "haplotypes")


def test_seqs_kind_none_with_haps_storage_returns_tracks_only():
    """User said with_seqs(None); tracks are active → tracks-only."""
    seqs = _haps_mock()
    tracks = _tracks_mock()  # active by default
    result = _build_reconstructor(seqs, tracks, None)
    assert result is tracks


def test_tracks_inactive_with_seqs_returns_seqs_only():
    """User said with_tracks(False) — active_tracks=None — and seqs are active."""
    seqs = _haps_mock()
    tracks = Mock(spec=Tracks)
    tracks.active_tracks = None  # user-deactivated
    result = _build_reconstructor(seqs, tracks, "haplotypes")
    # Result should be the haps view (kind-resolved), not RefTracks/HapsTracks.
    assert isinstance(result, Haps)


def test_both_inactive_raises():
    """seqs_kind=None and tracks inactive — meaningless state."""
    seqs = _haps_mock()
    tracks = Mock(spec=Tracks)
    tracks.active_tracks = None
    with pytest.raises(ValueError, match="at least one"):
        _build_reconstructor(seqs, tracks, None)
