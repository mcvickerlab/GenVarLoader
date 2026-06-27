"""choose_exonic_variants: rust vs frozen golden (oracle frozen Phase 5 W5)."""

from __future__ import annotations

import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_choose_exonic_variants_golden():
    cases = _golden.load_golden("choose_exonic_variants")
    assert cases, "empty golden"
    _golden.replay_tuple("choose_exonic_variants", cases)
