"""intervals_to_tracks: rust vs frozen golden (oracle frozen Phase 5 W5)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_intervals_to_tracks_golden():
    cases = _golden.load_golden("intervals_to_tracks")
    assert cases, "empty golden"
    _golden.replay_inplace(
        "intervals_to_tracks",
        cases,
        out_factory=lambda inputs: np.zeros(int(np.asarray(inputs[-1])[-1]), np.float32),
        out_index=6,
    )
