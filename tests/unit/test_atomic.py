import os
from pathlib import Path

import pytest

from genvarloader import _atomic
from genvarloader._atomic import SkipPublish, atomic_dir


def test_publishes_on_clean_exit(tmp_path):
    dest = tmp_path / "artifact"
    with atomic_dir(dest) as tmp:
        assert tmp.is_dir()
        assert tmp != dest
        (tmp / "data.bin").write_bytes(b"hello")
    assert dest.is_dir()
    assert (dest / "data.bin").read_bytes() == b"hello"
    # temp sibling cleaned up
    leftovers = list(tmp_path.glob("artifact.tmp.*"))
    assert leftovers == []


def test_temp_is_sibling_same_parent(tmp_path):
    dest = tmp_path / "sub" / "artifact"
    dest.parent.mkdir()
    seen = {}
    with atomic_dir(dest) as tmp:
        seen["tmp"] = tmp
        (tmp / "x").write_text("1")
    assert seen["tmp"].parent == dest.parent


def test_removes_temp_and_leaves_no_dest_on_exception(tmp_path):
    dest = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_dir(dest) as tmp:
            (tmp / "x").write_text("1")
            raise RuntimeError("boom")
    assert not dest.exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []


def test_existing_dest_no_overwrite_raises(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        with atomic_dir(dest, overwrite=False) as tmp:
            (tmp / "x").write_text("1")


def test_overwrite_replaces_existing_dir(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"old")
    with atomic_dir(dest, overwrite=True) as tmp:
        (tmp / "new.bin").write_bytes(b"new")
    assert (dest / "new.bin").read_bytes() == b"new"
    assert not (dest / "old.bin").exists()
    assert list(tmp_path.glob("artifact.old.*")) == []


def test_overwrite_retains_old_tree_during_unavoidable_publish_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"old")
    real_replace = os.replace
    observed_handoff: list[tuple[bool, bytes]] = []

    def inspect_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        src_path, dst_path = Path(src), Path(dst)
        if src_path.name.startswith("artifact.tmp.") and dst_path == dest:
            backups = list(tmp_path.glob("artifact.old.*"))
            assert len(backups) == 1
            observed_handoff.append(
                (dest.exists(), (backups[0] / "old.bin").read_bytes())
            )
        real_replace(src, dst)

    monkeypatch.setattr(_atomic.os, "replace", inspect_replace)

    with atomic_dir(dest, overwrite=True, lock=False) as tmp:
        (tmp / "new.bin").write_bytes(b"new")

    assert observed_handoff == [(False, b"old")]
    assert (dest / "new.bin").read_bytes() == b"new"
    assert list(tmp_path.glob("artifact.old.*")) == []


def test_overwrite_publish_failure_restores_old_dest_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"old")
    real_replace = os.replace
    replace_count = 0

    def fail_new_publish(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("publish failed")
        real_replace(src, dst)

    monkeypatch.setattr(_atomic.os, "replace", fail_new_publish)

    with pytest.raises(OSError, match="publish failed"):
        with atomic_dir(dest, overwrite=True, lock=False) as tmp:
            (tmp / "new.bin").write_bytes(b"new")

    assert (dest / "old.bin").read_bytes() == b"old"
    assert list(tmp_path.glob("artifact.tmp.*")) == []
    assert list(tmp_path.glob("artifact.old.*")) == []


def test_overwrite_rollback_failure_preserves_backup_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"old")
    real_replace = os.replace
    replace_count = 0

    def fail_publish_and_rollback(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("publish failed")
        if replace_count == 3:
            raise OSError("rollback failed")
        real_replace(src, dst)

    monkeypatch.setattr(_atomic.os, "replace", fail_publish_and_rollback)

    with pytest.raises(OSError, match="publish failed") as exc_info:
        with atomic_dir(dest, overwrite=True, lock=False) as tmp:
            (tmp / "new.bin").write_bytes(b"new")

    backups = list(tmp_path.glob("artifact.old.*"))
    assert replace_count == 3
    assert not dest.exists()
    assert len(backups) == 1
    assert (backups[0] / "old.bin").read_bytes() == b"old"
    assert list(tmp_path.glob("artifact.tmp.*")) == []
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "rollback failed"


def test_skip_publish_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "keep.bin").write_bytes(b"keep")
    with atomic_dir(dest, overwrite=True) as tmp:
        (tmp / "ignored.bin").write_bytes(b"ignored")
        raise SkipPublish
    assert (dest / "keep.bin").read_bytes() == b"keep"
    assert not (dest / "ignored.bin").exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []


def test_lock_file_sibling_created_and_reused(tmp_path):
    dest = tmp_path / "artifact"
    with atomic_dir(dest, lock=True) as tmp:
        (tmp / "x").write_text("1")
    # filelock leaves an empty <dest>.lock sibling
    assert (tmp_path / "artifact.lock").exists()


def test_concurrent_publish_loser_discarded(tmp_path):
    # Simulate: a racing builder publishes dest while we are mid-build and we
    # are overwrite=False. Our publish must discard our temp, not clobber.
    dest = tmp_path / "artifact"
    with atomic_dir(dest, overwrite=False, lock=False) as tmp:
        (tmp / "ours.bin").write_bytes(b"ours")
        # racing builder finishes first:
        dest.mkdir()
        (dest / "theirs.bin").write_bytes(b"theirs")
    assert (dest / "theirs.bin").read_bytes() == b"theirs"
    assert not (dest / "ours.bin").exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []
