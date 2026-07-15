import pytest

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
