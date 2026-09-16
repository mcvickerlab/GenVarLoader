"""Unit tests for ``Svar2Link`` resolution + fingerprint integrity.

Mirrors ``test_svar_link_models.py`` but for the ``.svar2`` back-reference
(``_svar2_link.py``). Three pure/tmp_path tests exercise the override/no-op
error paths; one integration-flavored test builds a real ``.svar2`` store
(via genoray's conversion pipeline, same fixture recipe as
``tests/test_svar2_reconstruct.py``) to prove the fingerprint actually
detects a mutated store.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from genvarloader._dataset._svar2_link import (
    Svar2Fingerprint,
    Svar2Link,
    _resolve_svar2,
    _verify_svar2_fingerprint,
    make_svar2_link,
)


def test_resolve_prefers_override(tmp_path: Path):
    real = tmp_path / "cohort.svar2"
    real.mkdir()
    link = Svar2Link(
        relative_path="nope.svar2",
        absolute_path="/nope.svar2",
        fingerprint=Svar2Fingerprint(n_files=1, store_bytes=1),
    )
    assert _resolve_svar2(tmp_path, link, real) == real


def test_resolve_missing_override_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _resolve_svar2(tmp_path, None, tmp_path / "absent.svar2")


def test_verify_none_link_is_noop(tmp_path: Path):
    _verify_svar2_fingerprint(tmp_path, None)  # must not raise


def test_fingerprint_detects_mutated_store(svar2_store_2s: Path, tmp_path: Path):
    # `svar2_store_2s` is a session-scoped fixture shared with five other test
    # modules -- mutate a private copy, never the shared store itself. Mutating
    # it in place corrupts every later consumer across the whole session.
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")

    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    link = make_svar2_link(gvl_path, store)
    _verify_svar2_fingerprint(store, link)  # must not raise

    bin_files = sorted(store.rglob("*.bin"))
    assert bin_files, "expected at least one .bin file in a real .svar2 store"
    with open(bin_files[0], "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError):
        _verify_svar2_fingerprint(store, link)
