import numpy as np
import pytest

from genvarloader._dataset._validate import validate_dataset
from genvarloader._dataset._write import DATASET_FORMAT_VERSION, Metadata


def _minimal_valid_dataset(path):
    path.mkdir()
    meta = Metadata(
        samples=["s0", "s1"],
        contigs=["chr1"],
        n_regions=2,
        format_version=DATASET_FORMAT_VERSION,
    )
    (path / "metadata.json").write_text(meta.model_dump_json())
    # input_regions.arrow: presence-only check, write a stub
    (path / "input_regions.arrow").write_bytes(b"stub")
    np.save(path / "regions.npy", np.zeros((2, 4), dtype=np.int32))
    return meta


def test_valid_dataset_passes(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    validate_dataset(meta, path)  # no raise


def test_missing_format_version_loads_as_1_0_0(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.format_version = None
    validate_dataset(meta, path)  # no raise, no warning


def test_format_version_too_new_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    from pydantic_extra_types.semantic_version import SemanticVersion

    meta.format_version = SemanticVersion.parse(
        f"{DATASET_FORMAT_VERSION.major + 1}.0.0"
    )
    with pytest.raises(ValueError, match="format version"):
        validate_dataset(meta, path)


def test_missing_regions_npy_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    (path / "regions.npy").unlink()
    with pytest.raises(ValueError, match="regions.npy"):
        validate_dataset(meta, path)


def test_regions_npy_wrong_length_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    np.save(path / "regions.npy", np.zeros((5, 4), dtype=np.int32))  # n_regions=2
    with pytest.raises(ValueError, match="regions.npy"):
        validate_dataset(meta, path)


def test_genotype_offsets_wrong_length_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.ploidy = 2
    geno = path / "genotypes"
    geno.mkdir()
    # correct length would be n_regions*ploidy*n_samples + 1 = 2*2*2 + 1 = 9
    # offsets.npy is a raw memmap (no numpy header), so use tofile not np.save
    np.zeros(4, dtype=np.int64).tofile(geno / "offsets.npy")
    with pytest.raises(ValueError, match="offsets.npy"):
        validate_dataset(meta, path)


def test_genotype_offsets_correct_length_passes(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.ploidy = 2
    geno = path / "genotypes"
    geno.mkdir()
    # offsets.npy is a raw memmap (no numpy header), so use tofile not np.save
    np.zeros(2 * 2 * 2 + 1, dtype=np.int64).tofile(geno / "offsets.npy")
    validate_dataset(meta, path)  # no raise
