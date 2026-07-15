import builtins
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pysam
import pytest

import genvarloader._atomic as atomic
import genvarloader._fasta_cache as fc


@contextmanager
def _capture_warnings():
    messages = []
    sink_id = fc.logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        fc.logger.remove(sink_id)


def _sharing_violation(path: Path) -> PermissionError:
    error = PermissionError(13, "sharing violation", str(path))
    error.winerror = 32  # type: ignore[attr-defined]
    return error


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_fingerprint_records_size_and_window(tmp_path):
    src = _write(tmp_path / "a.bin", b"ACGT" * 10)
    fp = fc.fingerprint(src)
    assert fp.algorithm == "blake2b"
    assert fp.size_bytes == 40
    assert fp.n_bytes_hashed == 40
    assert isinstance(fp.digest, str) and len(fp.digest) > 0


def test_fingerprint_window_caps_at_1mib(tmp_path):
    src = _write(tmp_path / "big.bin", b"N" * (fc.FINGERPRINT_WINDOW + 100))
    fp = fc.fingerprint(src)
    assert fp.size_bytes == fc.FINGERPRINT_WINDOW + 100
    assert fp.n_bytes_hashed == fc.FINGERPRINT_WINDOW


def test_fingerprint_identical_bytes_match(tmp_path):
    a = _write(tmp_path / "a.bin", b"ACGTACGT")
    b = _write(tmp_path / "b.bin", b"ACGTACGT")
    assert fc.fingerprint(a).digest == fc.fingerprint(b).digest


def test_fingerprint_flipped_first_byte_differs(tmp_path):
    a = _write(tmp_path / "a.bin", b"ACGTACGT")
    b = _write(tmp_path / "b.bin", b"TCGTACGT")
    assert fc.fingerprint(a).digest != fc.fingerprint(b).digest


def _make_meta(tmp_path, source_fa, gvlfa_dir) -> fc.FastaCache:
    return fc.FastaCache(
        format_version=fc.FORMAT_VERSION,
        genvarloader_version=fc.FORMAT_VERSION,
        contigs={"chr1": 4},
        source=fc._source_hints(source_fa, gvlfa_dir),
        fingerprint=fc.fingerprint(source_fa),
    )


def test_resolve_source_sibling(tmp_path):
    src = _write(tmp_path / "ref.fa", b"ACGT")
    gvlfa = tmp_path / "ref.fa.gvlfa"
    gvlfa.mkdir()
    meta = _make_meta(tmp_path, src, gvlfa)
    assert fc.resolve_source(gvlfa, meta) == src.resolve()


def test_resolve_source_absolute_when_moved(tmp_path):
    # Source elsewhere; cache dir has no sibling and a non-resolving relative.
    src_dir = tmp_path / "elsewhere"
    src_dir.mkdir()
    src = _write(src_dir / "ref.fa", b"ACGT")
    gvlfa = tmp_path / "cache" / "ref.fa.gvlfa"
    gvlfa.mkdir(parents=True)
    meta = _make_meta(tmp_path, src, gvlfa)
    # No sibling ref.fa next to the cache dir, but absolute_path still points home.
    assert fc.resolve_source(gvlfa, meta) == src.resolve()


def test_resolve_source_missing_returns_none(tmp_path):
    src = _write(tmp_path / "ref.fa", b"ACGT")
    gvlfa = tmp_path / "ref.fa.gvlfa"
    gvlfa.mkdir()
    meta = _make_meta(tmp_path, src, gvlfa)
    src.unlink()
    assert fc.resolve_source(gvlfa, meta) is None


@pytest.fixture
def local_fa(tmp_path, ref_fasta):
    """Copy the shared bgzipped FASTA (and its .fai/.gzi if present) into tmp_path."""
    src = Path(ref_fasta)
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    for ext in (".fai", ".gzi"):
        side = Path(str(src) + ext)
        if side.exists():
            shutil.copy(side, tmp_path / side.name)
    return dst


def test_build_then_load_round_trip(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    meta = fc.build(local_fa, gvlfa)
    assert (gvlfa / fc.METADATA_FILENAME).exists()
    assert (gvlfa / fc.DATA_FILENAME).exists()

    loaded, source, status = fc.load(gvlfa)
    assert loaded.contigs == meta.contigs
    assert source == local_fa.resolve()
    assert status == "fresh"


def test_build_sequence_matches_pysam(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    meta = fc.build(local_fa, gvlfa)
    data = np.memmap(gvlfa / fc.DATA_FILENAME, dtype="S1", mode="r")
    offset = 0
    with pysam.FastaFile(str(local_fa)) as f:
        for contig, length in meta.contigs.items():
            expected = np.frombuffer(f.fetch(contig).encode("ascii").upper(), "S1")
            np.testing.assert_array_equal(data[offset : offset + length], expected)
            offset += length


def test_load_stale_when_source_changes(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    fc.build(local_fa, gvlfa)
    # Truncate the source so size_bytes (and thus the fingerprint) changes.
    with open(local_fa, "r+b") as f:
        f.truncate(10)
    _, source, status = fc.load(gvlfa)
    assert source == local_fa.resolve()
    assert status == "stale"


def test_load_unvalidated_when_source_missing(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    fc.build(local_fa, gvlfa)
    local_fa.unlink()
    _, source, status = fc.load(gvlfa)
    assert source is None
    assert status == "unvalidated"


def test_load_rejects_newer_major_format(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    meta = fc.build(local_fa, gvlfa)
    meta.format_version = type(meta.format_version)(
        major=fc.FORMAT_VERSION.major + 1, minor=0, patch=0
    )
    (gvlfa / fc.METADATA_FILENAME).write_text(meta.model_dump_json())
    with pytest.raises(ValueError, match="newer than supported"):
        fc.load(gvlfa)


def test_data_size_ok_detects_corruption(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    meta = fc.build(local_fa, gvlfa)
    assert fc._data_size_ok(gvlfa, meta) is True
    (gvlfa / fc.DATA_FILENAME).write_bytes(b"too short")
    assert fc._data_size_ok(gvlfa, meta) is False


def test_migrate_legacy_reuses_bytes_and_removes_old(tmp_path, local_fa):
    # Fabricate a legacy flat cache: concatenated contigs, exactly as the old
    # _write_to_cache produced. Build via the new code, then rename to legacy.
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()

    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    meta = fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert not legacy.exists()  # old file removed (moved)
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes  # bytes reused
    assert meta.contigs == fc._contig_lengths(local_fa)
    _, source, status = fc.load(gvlfa)
    assert status == "fresh" and source == local_fa.resolve()


def test_migrate_legacy_publish_failure_preserves_source_and_previous_cache(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()

    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    gvlfa.mkdir()
    previous = gvlfa / "previous-cache"
    previous.write_text("keep me")
    real_replace = atomic.os.replace

    def fail_new_publish(src, dst):
        src_path, dst_path = Path(src), Path(dst)
        if dst_path == gvlfa and ".tmp." in src_path.name:
            raise OSError("publish failed")
        real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", fail_new_publish)

    with pytest.raises(OSError, match="publish failed"):
        fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert legacy.read_bytes() == legacy_bytes
    assert previous.read_text() == "keep me"
    assert list(tmp_path.glob(f"{gvlfa.name}.tmp.*")) == []
    assert list(tmp_path.glob(f"{gvlfa.name}.old.*")) == []


def test_concurrent_legacy_migrations_reuse_published_destination(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)

    real_fingerprint = fc.fingerprint
    both_ready = threading.Barrier(2)

    def synchronized_fingerprint(path):
        result = real_fingerprint(path)
        both_ready.wait(timeout=5)
        return result

    real_copyfileobj = fc.shutil.copyfileobj

    def slow_legacy_copy(src, dst, *args, **kwargs):
        time.sleep(0.25)
        return real_copyfileobj(src, dst, *args, **kwargs)

    monkeypatch.setattr(fc, "fingerprint", synchronized_fingerprint)
    monkeypatch.setattr(fc.shutil, "copyfileobj", slow_legacy_copy)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(fc.migrate_legacy, local_fa, legacy, gvlfa) for _ in range(2)
        ]
        metadata = [future.result(timeout=5) for future in futures]

    assert metadata[0] == metadata[1]
    assert not legacy.exists()
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes


def test_queued_migrator_reuses_publication_and_leaves_replacement(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    original_bytes = legacy.read_bytes()
    replacement_bytes = b"R" * len(original_bytes)
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)

    real_source_lock = fc.FileLock
    source_lock_entries = 0
    source_lock_entries_guard = threading.Lock()
    waiter_queued = threading.Event()

    class ObservedSourceLock:
        def __init__(self, path):
            self._lock = real_source_lock(path)

        def __enter__(self):
            nonlocal source_lock_entries
            with source_lock_entries_guard:
                source_lock_entries += 1
                if source_lock_entries == 2:
                    waiter_queued.set()
            return self._lock.__enter__()

        def __exit__(self, *args):
            return self._lock.__exit__(*args)

    real_atomic_dir = fc.atomic_dir
    publication_calls = 0
    publication_calls_guard = threading.Lock()

    @contextmanager
    def replace_source_after_first_publish(*args, **kwargs):
        nonlocal publication_calls
        with publication_calls_guard:
            publication_calls += 1
            is_first_publication = publication_calls == 1
        with real_atomic_dir(*args, **kwargs) as tmp:
            if is_first_publication:
                assert waiter_queued.wait(timeout=5)
            yield tmp
        if is_first_publication:
            replacement = tmp_path / "queued-replacement.gvl"
            replacement.write_bytes(replacement_bytes)
            replacement.replace(legacy)

    monkeypatch.setattr(fc, "FileLock", ObservedSourceLock)
    monkeypatch.setattr(fc, "atomic_dir", replace_source_after_first_publish)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(fc.migrate_legacy, local_fa, legacy, gvlfa) for _ in range(2)
        ]
        metadata = [future.result(timeout=5) for future in futures]

    assert metadata[0] == metadata[1]
    assert publication_calls == 1
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == original_bytes
    assert legacy.read_bytes() == replacement_bytes


def test_migrate_legacy_does_not_unlink_post_publish_replacement(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_atomic_dir = fc.atomic_dir
    replacement_bytes = b"replacement legacy cache"

    @contextmanager
    def replace_source_after_publish(*args, **kwargs):
        with real_atomic_dir(*args, **kwargs) as tmp:
            yield tmp
        replacement = tmp_path / "replacement.gvl"
        replacement.write_bytes(replacement_bytes)
        replacement.replace(legacy)

    monkeypatch.setattr(fc, "atomic_dir", replace_source_after_publish)

    fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert legacy.read_bytes() == replacement_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes


def test_migrate_legacy_rejects_replacement_immediately_before_copy_open(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    replacement = tmp_path / "replacement-before-open.gvl"
    replacement_bytes = b"R" * len(legacy_bytes)
    replacement.write_bytes(replacement_bytes)
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_open = builtins.open
    replaced = False

    def replace_before_copy_open(file, mode="r", *args, **kwargs):
        nonlocal replaced
        if file == legacy and mode == "rb" and not replaced:
            replacement.replace(legacy)
            replaced = True
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", replace_before_copy_open)

    with pytest.raises(RuntimeError, match="changed before it could be opened"):
        fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert replaced
    assert legacy.read_bytes() == replacement_bytes
    assert not gvlfa.exists()


def test_migrate_legacy_cleanup_does_not_unlink_replacement_after_identity_check(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    replacement = tmp_path / "replacement-before-unlink.gvl"
    replacement_bytes = b"replacement after identity check"
    replacement.write_bytes(replacement_bytes)
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_unlink = Path.unlink
    replaced = False

    def replace_before_cleanup_unlink(path, *args, **kwargs):
        nonlocal replaced
        is_cleanup_candidate = path.name.startswith(f"{legacy.name}.cleanup.")
        if not replaced and (path == legacy or is_cleanup_candidate):
            replacement.replace(legacy)
            replaced = True
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", replace_before_cleanup_unlink)

    fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert replaced
    assert legacy.read_bytes() == replacement_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes


def test_migrate_legacy_cleanup_claim_failure_is_nonfatal(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_replace = fc.os.replace

    def fail_cleanup_claim(src, dst):
        if Path(src) == legacy and Path(dst).name.startswith(f"{legacy.name}.cleanup."):
            raise PermissionError("cleanup claim denied")
        real_replace(src, dst)

    monkeypatch.setattr(fc.os, "replace", fail_cleanup_claim)

    with _capture_warnings() as warnings:
        meta = fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert meta.contigs == fc._contig_lengths(local_fa)
    assert legacy.read_bytes() == legacy_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes
    assert any("claim legacy cache for cleanup" in warning for warning in warnings)


def test_migrate_legacy_cleanup_candidate_stat_failure_is_nonfatal(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_stat = Path.stat

    def fail_cleanup_stat(path, *args, **kwargs):
        if path.name.startswith(f"{legacy.name}.cleanup."):
            raise PermissionError("cleanup candidate is busy")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_cleanup_stat)

    with _capture_warnings() as warnings:
        fc.migrate_legacy(local_fa, legacy, gvlfa)

    candidates = list(tmp_path.glob(f"{legacy.name}.cleanup.*"))
    assert len(candidates) == 1
    assert candidates[0].read_bytes() == legacy_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes
    assert any("inspect cleanup candidate" in warning for warning in warnings)


def test_migrate_legacy_cleanup_unlink_sharing_violation_is_nonfatal(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_unlink = Path.unlink

    def fail_cleanup_unlink(path, *args, **kwargs):
        if path.name.startswith(f"{legacy.name}.cleanup."):
            raise _sharing_violation(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup_unlink)

    with _capture_warnings() as warnings:
        fc.migrate_legacy(local_fa, legacy, gvlfa)

    candidates = list(tmp_path.glob(f"{legacy.name}.cleanup.*"))
    assert len(candidates) == 1
    assert candidates[0].read_bytes() == legacy_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes
    assert any("remove cleanup candidate" in warning for warning in warnings)


def test_migrate_legacy_restore_unlink_failure_preserves_both_links(
    tmp_path, local_fa, monkeypatch
):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    legacy_bytes = legacy.read_bytes()
    replacement_bytes = b"replacement blocked by a Windows-style sharing violation"
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    real_atomic_dir = fc.atomic_dir
    real_unlink = Path.unlink

    @contextmanager
    def replace_source_after_publish(*args, **kwargs):
        with real_atomic_dir(*args, **kwargs) as tmp:
            yield tmp
        replacement = tmp_path / "replacement-for-restore.gvl"
        replacement.write_bytes(replacement_bytes)
        replacement.replace(legacy)

    def fail_cleanup_unlink(path, *args, **kwargs):
        if path.name.startswith(f"{legacy.name}.cleanup."):
            raise _sharing_violation(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(fc, "atomic_dir", replace_source_after_publish)
    monkeypatch.setattr(Path, "unlink", fail_cleanup_unlink)

    with _capture_warnings() as warnings:
        fc.migrate_legacy(local_fa, legacy, gvlfa)

    candidates = list(tmp_path.glob(f"{legacy.name}.cleanup.*"))
    assert legacy.read_bytes() == replacement_bytes
    assert len(candidates) == 1
    assert candidates[0].read_bytes() == replacement_bytes
    assert (gvlfa / fc.DATA_FILENAME).read_bytes() == legacy_bytes
    assert any("remove restored cleanup candidate" in warning for warning in warnings)


def test_migrate_legacy_aborts_before_move_if_source_unreadable(tmp_path):
    legacy = tmp_path / "ghost.fa.gvl"
    legacy.write_bytes(b"data")
    missing_src = tmp_path / "ghost.fa"  # does not exist
    gvlfa = tmp_path / "ghost.fa.gvlfa"
    with pytest.raises(Exception):
        fc.migrate_legacy(missing_src, legacy, gvlfa)
    assert legacy.exists()  # untouched on failure


def test_migrate_legacy_truncated_bytes_builds_fresh(tmp_path, local_fa):
    # A stale/truncated legacy cache must NOT be trusted: migrate should build
    # fresh from the source and leave the legacy file untouched.
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    legacy.write_bytes(b"truncated")  # wrong size vs the real source
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    meta = fc.migrate_legacy(local_fa, legacy, gvlfa)

    assert legacy.exists()  # stale legacy left untouched, not consumed
    assert (gvlfa / fc.DATA_FILENAME).stat().st_size == sum(meta.contigs.values())
    # The rebuilt cache is valid and fresh against the source.
    _, source, status = fc.load(gvlfa)
    assert status == "fresh" and source == local_fa.resolve()


def test_is_gvlfa_dir(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    fc.build(local_fa, gvlfa)
    assert fc.is_gvlfa(gvlfa) is True
    assert fc.is_gvlfa(local_fa) is False


def test_ensure_cache_from_fasta_builds_then_loads(tmp_path, local_fa):
    meta, data_path = fc.ensure_cache(local_fa)
    expected_dir = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    assert data_path == expected_dir / fc.DATA_FILENAME
    assert data_path.exists()
    # Second call loads the existing fresh cache (no error, same contigs).
    meta2, _ = fc.ensure_cache(local_fa)
    assert meta2.contigs == meta.contigs


def test_ensure_cache_from_fasta_rebuilds_when_stale(tmp_path, local_fa):
    fc.ensure_cache(local_fa)
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    # Corrupt the data file; ensure_cache should rebuild it to the right size.
    (gvlfa / fc.DATA_FILENAME).write_bytes(b"short")
    meta, data_path = fc.ensure_cache(local_fa)
    assert data_path.stat().st_size == sum(meta.contigs.values())


def test_ensure_cache_from_fasta_migrates_legacy(tmp_path, local_fa):
    staging = tmp_path / "staging.gvlfa"
    fc.build(local_fa, staging)
    legacy = local_fa.with_name(local_fa.name + fc.LEGACY_SUFFIX)
    shutil.move(str(staging / fc.DATA_FILENAME), str(legacy))
    shutil.rmtree(staging)

    meta, data_path = fc.ensure_cache(local_fa)
    assert not legacy.exists()
    assert data_path.exists()
    assert meta.contigs == fc._contig_lengths(local_fa)


def test_ensure_cache_from_gvlfa_missing_source_warns(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    fc.build(local_fa, gvlfa)
    local_fa.unlink()
    with pytest.warns(UserWarning, match="Could not locate source FASTA"):
        meta, data_path = fc.ensure_cache(gvlfa)
    assert data_path == gvlfa / fc.DATA_FILENAME


def test_ensure_cache_from_gvlfa_stale_rebuilds(tmp_path):
    # Use a plain-text FASTA: rebuilding needs a valid re-readable source, and
    # truncating a bgzipped file would corrupt it for pysam.
    src = tmp_path / "tiny.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    gvlfa = tmp_path / "tiny.fa.gvlfa"
    fc.build(src, gvlfa)
    # Change the source content (same path), invalidating the fingerprint.
    src.write_text(">chr1\nTTTTTTTT\n")
    pysam.faidx(str(src))
    meta, data_path = fc.ensure_cache(gvlfa)
    data = np.memmap(data_path, dtype="S1", mode="r")
    np.testing.assert_array_equal(data, np.frombuffer(b"TTTTTTTT", "S1"))


def test_ensure_cache_from_gvlfa_corrupt_data_no_source_raises(tmp_path, local_fa):
    gvlfa = tmp_path / "out.gvlfa"
    fc.build(local_fa, gvlfa)
    # Corrupt the data and remove the source so it cannot be rebuilt.
    (gvlfa / fc.DATA_FILENAME).write_bytes(b"short")
    local_fa.unlink()
    with pytest.raises(ValueError, match="could not be located"):
        fc.ensure_cache(gvlfa)


def test_ensure_cache_format_too_new_raises_from_both_entry_points(tmp_path, local_fa):
    gvlfa = local_fa.with_name(local_fa.name + fc.GVLFA_SUFFIX)
    meta = fc.build(local_fa, gvlfa)
    meta.format_version = type(meta.format_version)(
        major=fc.FORMAT_VERSION.major + 1, minor=0, patch=0
    )
    (gvlfa / fc.METADATA_FILENAME).write_text(meta.model_dump_json())
    # Direct .gvlfa entry point raises.
    with pytest.raises(ValueError, match="newer than supported"):
        fc.ensure_cache(gvlfa)
    # .fa entry point (sibling cache) must ALSO raise, not silently rebuild/downgrade.
    with pytest.raises(ValueError, match="newer than supported"):
        fc.ensure_cache(local_fa)


def test_build_is_atomic_no_temp_left(tmp_path):
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    gvlfa = tmp_path / "ref.fa.gvlfa"
    fc.build(src, gvlfa)
    assert (gvlfa / fc.DATA_FILENAME).exists()
    assert (gvlfa / fc.METADATA_FILENAME).exists()
    # no orphan temp / lock-leak that looks like a cache dir
    assert list(tmp_path.glob("ref.fa.gvlfa.tmp.*")) == []
    assert list(tmp_path.glob("ref.fa.gvlfa.old.*")) == []


def test_build_failure_leaves_no_partial_cache(tmp_path, monkeypatch):
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    gvlfa = tmp_path / "ref.fa.gvlfa"

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fc, "_write_sequence", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        fc.build(src, gvlfa)
    assert not gvlfa.exists()
    assert list(tmp_path.glob("ref.fa.gvlfa.tmp.*")) == []


def test_ensure_cache_double_check_reuses_fresh(tmp_path):
    # When dest is already a fresh cache, ensure_cache must not rebuild it.
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    meta1, data1 = fc.ensure_cache(src)
    mtime1 = (data1).stat().st_mtime_ns
    meta2, data2 = fc.ensure_cache(src)
    assert data2 == data1
    # data file was not rewritten (fresh cache reused)
    assert data2.stat().st_mtime_ns == mtime1
