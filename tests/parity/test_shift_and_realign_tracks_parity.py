"""shift_and_realign_tracks_sparse: rust vs frozen golden (oracle frozen Phase 5 W5)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_shift_and_realign_tracks_sparse_golden():
    cases = _golden.load_golden("shift_and_realign_tracks_sparse")
    assert cases, "empty golden"
    _golden.replay_inplace(
        "shift_and_realign_tracks_sparse",
        cases,
        out_factory=lambda inputs: np.zeros(int(np.asarray(inputs[0])[-1]), np.float32),
        out_index=0,
    )
