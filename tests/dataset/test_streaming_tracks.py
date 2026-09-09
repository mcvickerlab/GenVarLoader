"""Smoke test for the interval-streaming parity fixture (issue #279 Task 2).

Confirms `streaming_tracks_fixture` (see `conftest.py`) actually constructs and
that its two tracks -- passed to `gvl.write` in non-alphabetical order
(``[zeta, alpha]``) -- land on a name-sorted track axis, since every later
parity test (Tasks 5-8) depends on that ordering.
"""

from __future__ import annotations

import genvarloader as gvl


def test_fixture_builds(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    ds = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
    assert list(ds.available_tracks) == ["alpha", "zeta"], (
        "written track axis must be name-sorted, not tracks= argument order"
    )
