import shutil
from pathlib import Path

import numpy as np
import pysam
import pytest

import genvarloader._fasta_cache as fc


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


def test_migrate_legacy_aborts_before_move_if_source_unreadable(tmp_path):
    legacy = tmp_path / "ghost.fa.gvl"
    legacy.write_bytes(b"data")
    missing_src = tmp_path / "ghost.fa"  # does not exist
    gvlfa = tmp_path / "ghost.fa.gvlfa"
    with pytest.raises(Exception):
        fc.migrate_legacy(missing_src, legacy, gvlfa)
    assert legacy.exists()  # untouched on failure


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
