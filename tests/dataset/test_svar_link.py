import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_extra_types.semantic_version import SemanticVersion

from genvarloader._dataset._svar_link import SvarFingerprint, SvarLink
from genvarloader._dataset._write import Metadata


def test_svar_link_roundtrip():
    link = SvarLink(
        relative_path="../foo.svar",
        absolute_path="/abs/path/foo.svar",
        fingerprint=SvarFingerprint(n_variants=10, variant_idxs_bytes=42),
    )
    payload = link.model_dump_json()
    parsed = SvarLink.model_validate_json(payload)
    assert parsed == link


def test_svar_link_rejects_malformed_fingerprint():
    bad = (
        '{"relative_path":"a","absolute_path":"b",'
        '"fingerprint":{"n_variants":"not_an_int","variant_idxs_bytes":1}}'
    )
    with pytest.raises(ValidationError):
        SvarLink.model_validate_json(bad)


def test_metadata_version_parses_existing_strings():
    payload = json.dumps(
        {
            "samples": ["s1"],
            "contigs": ["1"],
            "n_regions": 1,
            "version": "0.18.0",
        }
    )
    m = Metadata.model_validate_json(payload)
    assert isinstance(m.version, SemanticVersion)
    assert m.version == SemanticVersion.parse("0.18.0")


def test_metadata_version_serializes_back_to_string():
    m = Metadata(
        samples=["s1"],
        contigs=["1"],
        n_regions=1,
        version=SemanticVersion.parse("0.18.0"),
    )
    dumped = json.loads(m.model_dump_json())
    assert dumped["version"] == "0.18.0"


def test_metadata_svar_link_defaults_to_none():
    m = Metadata(samples=["s1"], contigs=["1"], n_regions=1)
    assert m.svar_link is None


def test_semantic_version_ordering_for_one_based_dispatch():
    """The legacy comparison '>= 0.18.0' must still work under SemanticVersion."""
    assert SemanticVersion.parse("0.18.0") >= SemanticVersion.parse("0.18.0")
    assert SemanticVersion.parse("0.20.0") >= SemanticVersion.parse("0.18.0")
    assert not (SemanticVersion.parse("0.17.5") >= SemanticVersion.parse("0.18.0"))


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "tests" / "data"


@pytest.fixture
def svar_dataset_paths(tmp_path):
    """Produce a fresh GVL dataset built from the canonical test svar."""
    import genvarloader as gvl

    svar_path = _DATA_DIR / "filtered.svar"
    bed_path = _DATA_DIR / "source.bed"
    assert svar_path.is_dir(), f"missing fixture {svar_path}; run pixi run -e dev gen"
    assert bed_path.exists(), f"missing fixture {bed_path}"

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=bed_path, variants=svar_path, overwrite=True)
    return gvl_path, svar_path


def test_write_from_svar_records_svar_link_and_no_symlink(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths

    link_path = gvl_path / "genotypes" / "link.svar"
    assert not link_path.exists() and not link_path.is_symlink()

    metadata = Metadata.model_validate_json(
        (gvl_path / "metadata.json").read_text()
    )
    assert metadata.svar_link is not None
    assert Path(metadata.svar_link.absolute_path) == svar_path.resolve()
    assert (gvl_path / metadata.svar_link.relative_path).resolve() == svar_path.resolve()
    expected_bytes = (svar_path / "variant_idxs.npy").stat().st_size
    assert metadata.svar_link.fingerprint.variant_idxs_bytes == expected_bytes
    assert metadata.svar_link.fingerprint.n_variants > 0
