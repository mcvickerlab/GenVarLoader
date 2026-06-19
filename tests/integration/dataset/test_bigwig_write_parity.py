# tests/integration/dataset/test_bigwig_write_parity.py
from pathlib import Path

import pytest

from genvarloader import BigWigs
from genvarloader._dataset import _write
from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions, make_synthetic_bigwigs


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("bw_corpus")
    paths = make_synthetic_bigwigs(d, n_samples=3, density=0.02, seed=11)
    regions = make_regions(DEFAULT_CONTIGS, n_per_contig=20, width=5000, seed=3)
    return paths, regions


def _assert_byte_identical(a: Path, b: Path):
    assert (a / "intervals.npy").read_bytes() == (b / "intervals.npy").read_bytes()
    assert (a / "offsets.npy").read_bytes() == (b / "offsets.npy").read_bytes()


def test_per_sample_parity(corpus, tmp_path):
    paths, regions = corpus
    samples = [f"sample_{i}" for i in range(len(paths))]
    track = BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})

    legacy = tmp_path / "legacy"
    rust = tmp_path / "rust"
    legacy.mkdir()
    rust.mkdir()
    _write._write_track_legacy(legacy, regions, track, samples, 1 << 30)
    _write._write_track_rust(rust, regions, track, samples, 1 << 30)
    _assert_byte_identical(legacy, rust)


def test_annotation_parity(corpus, tmp_path):
    paths, regions = corpus
    legacy = tmp_path / "legacy"
    rust = tmp_path / "rust"
    legacy.mkdir()
    rust.mkdir()
    itvs = _write._annot_intervals(regions, paths[0], max_mem=1 << 30)
    _write._write_ragged_intervals(legacy, itvs)
    _write._write_annot_track_rust(rust, regions, paths[0], 1 << 30)
    _assert_byte_identical(legacy, rust)
