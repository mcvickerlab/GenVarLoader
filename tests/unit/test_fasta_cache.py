from pathlib import Path

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
