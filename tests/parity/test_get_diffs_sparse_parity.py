"""get_diffs_sparse: rust vs frozen golden (oracle frozen Phase 5 W5)."""

from __future__ import annotations

import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_get_diffs_sparse_golden():
    cases = _golden.load_golden("get_diffs_sparse")
    assert cases, "empty golden"
    _golden.replay_tuple("get_diffs_sparse", cases)
