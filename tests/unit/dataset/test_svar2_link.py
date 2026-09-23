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


def _add_annotation_layer(store: Path, name: str = "mutcat") -> list[Path]:
    """Write an additive annotation layer into every contig of `store`.

    Reproduces the shape of the layer in issue #419: a separate pipeline writes
    per-contig annotations beside the payload, touching nothing the dataset
    reads. Returns the files it created.
    """
    written: list[Path] = []
    for contig in sorted(p for p in store.iterdir() if p.is_dir()):
        layer = contig / name
        layer.mkdir()
        for fname in ("categories.npy", "offsets.npy", "codes.bin", "meta.bin"):
            f = layer / fname
            f.write_bytes(b"\x01" * 32)
            written.append(f)
    return written


def test_additive_annotation_layer_does_not_break_the_fingerprint(
    svar2_store_2s: Path, tmp_path: Path
):
    """Issue #419. A layer the dataset never reads must be invisible here.

    The fingerprint's stated contract is "changes iff the store's data files
    change". A sibling layer is not a data file, and counting it takes a shared
    store offline for every consumer while the variant data is byte-for-byte
    unchanged.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    link = make_svar2_link(gvl_path, store)
    _verify_svar2_fingerprint(store, link)  # baseline: clean

    added = _add_annotation_layer(store)
    assert added, "the layer fixture must actually write files"

    # The store on disk now has strictly more .npy/.bin files than when the
    # link was recorded, and none of them are payload.
    _verify_svar2_fingerprint(store, link)


def test_removing_an_annotation_layer_also_leaves_the_fingerprint_alone(
    svar2_store_2s: Path, tmp_path: Path
):
    """The same bug in the other direction, which is how aster hit it.

    There the link was recorded while the layer was present and the store
    later reached a consumer without it, so the recorded value referenced
    bytes that no longer existed. A payload-scoped fingerprint is symmetric:
    neither adding nor removing a sibling layer can move it.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    added = _add_annotation_layer(store)
    link = make_svar2_link(gvl_path, store)  # recorded WITH the layer
    _verify_svar2_fingerprint(store, link)

    for f in added:
        f.unlink()
    for contig in sorted(p for p in store.iterdir() if p.is_dir()):
        (contig / "mutcat").rmdir()

    _verify_svar2_fingerprint(store, link)


def test_a_new_payload_file_is_still_detected(svar2_store_2s: Path, tmp_path: Path):
    """Scoping must not become a hole: payload additions still count.

    Without this, "ignore what we don't read" could quietly widen into
    "ignore additions", and a store that gained a contig would verify clean.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()
    link = make_svar2_link(gvl_path, store)

    contig = next(p for p in sorted(store.iterdir()) if p.is_dir())
    (contig / "var_key" / "snp" / "extra.bin").write_bytes(b"\x00" * 16)

    with pytest.raises(ValueError):
        _verify_svar2_fingerprint(store, link)


def test_a_real_payload_mismatch_says_so_and_rules_out_a_sibling_layer(
    svar2_store_2s: Path, tmp_path: Path
):
    """Issue #419 asked the error to name appeared/disappeared files.

    It cannot name the specific file: the record is two integers, so there is
    no baseline to diff against, and inventing one would mean writing a
    per-file manifest into every dataset. What the message CAN do without
    storing anything more is state the scope it used and separate the two
    cases a reader has to tell apart -- and that is the part that cost a
    manual bisection. Here nothing but payload exists, so the verdict is
    unambiguous.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()
    link = make_svar2_link(gvl_path, store)

    target = sorted(store.rglob("var_key/snp/offsets.npy"))[0]
    with open(target, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError) as exc:
        _verify_svar2_fingerprint(store, link)

    msg = str(exc.value)
    assert "scope: payload only" in msg
    assert "var_key" in msg, "the message must say what the scope covers"
    assert "treat the store as modified" in msg
    assert "re-recorded" not in msg, (
        "with no sibling layer present the message must NOT suggest the "
        "recorded value is merely stale -- that would excuse real corruption"
    )


def test_a_record_larger_than_the_payload_is_not_called_corruption(
    svar2_store_2s: Path, tmp_path: Path
):
    """The direction aster hit, seen from a store that no longer has the layer.

    Once the sibling layer is gone there is nothing left to point at, so the
    store alone cannot prove which happened. The message must therefore say
    both readings are open rather than assert corruption -- a record written
    under the old unscoped walk is strictly larger than the payload on both
    axes, and so is a truncated store.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    payload = make_svar2_link(gvl_path, store).fingerprint
    stale = Svar2Link(
        relative_path="store.svar2",
        absolute_path=str(store),
        fingerprint=Svar2Fingerprint(
            n_files=payload.n_files + 4,
            store_bytes=payload.store_bytes + 128,
        ),
    )

    with pytest.raises(ValueError) as exc:
        _verify_svar2_fingerprint(store, stale)

    msg = str(exc.value)
    assert "before the fingerprint was scoped" in msg
    assert "Compare against a replica" in msg
    assert "treat the store as modified" not in msg, (
        "must not assert corruption when a stale broad-scope record explains "
        "the mismatch equally well"
    )


def test_a_link_recorded_before_scoping_still_verifies(
    svar2_store_2s: Path, tmp_path: Path
):
    """Narrowing the scope must not invalidate records that pass today.

    Datasets in the wild were linked while the walk covered the whole store,
    so their recorded value includes any sibling layer present at write time.
    Those stores are healthy and open fine; if scoping started rejecting them
    the fix would break more than the bug did.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    _add_annotation_layer(store)
    broad = [
        q for q in store.rglob("*") if q.is_file() and q.suffix in {".bin", ".npy"}
    ]
    legacy = Svar2Link(
        relative_path="store.svar2",
        absolute_path=str(store),
        fingerprint=Svar2Fingerprint(
            n_files=len(broad),
            store_bytes=sum(q.stat().st_size for q in broad),
        ),
    )

    _verify_svar2_fingerprint(store, legacy)  # must not raise


def test_legacy_acceptance_does_not_excuse_a_modified_payload(
    svar2_store_2s: Path, tmp_path: Path
):
    """The legacy path is a second exact match, never a loosening.

    If it degraded into "close enough", a store whose variant data changed
    would sail through on the strength of an old record -- the one outcome
    this fingerprint exists to prevent.
    """
    store = shutil.copytree(svar2_store_2s, tmp_path / "store.svar2")
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    _add_annotation_layer(store)
    broad = [
        q for q in store.rglob("*") if q.is_file() and q.suffix in {".bin", ".npy"}
    ]
    legacy = Svar2Link(
        relative_path="store.svar2",
        absolute_path=str(store),
        fingerprint=Svar2Fingerprint(
            n_files=len(broad),
            store_bytes=sum(q.stat().st_size for q in broad),
        ),
    )
    _verify_svar2_fingerprint(store, legacy)

    payload_file = sorted(store.rglob("var_key/snp/offsets.npy"))[0]
    with open(payload_file, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError):
        _verify_svar2_fingerprint(store, legacy)
