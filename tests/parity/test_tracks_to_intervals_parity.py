"""tracks_to_intervals: rust vs frozen golden (oracle frozen Phase 5 W5)."""
from __future__ import annotations

import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_tracks_to_intervals_golden():
    cases = _golden.load_golden("tracks_to_intervals")
    assert cases, "empty golden"
    _golden.replay_tuple("tracks_to_intervals", cases)
