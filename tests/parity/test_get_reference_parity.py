"""get_reference: rust vs frozen golden (oracle frozen Phase 5 W5)."""

from __future__ import annotations

import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_get_reference_golden():
    cases = _golden.load_golden("get_reference")
    assert cases, "empty golden"
    _golden.replay_return("get_reference", cases)
