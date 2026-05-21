import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_extra_types.semantic_version import SemanticVersion

from genvarloader._dataset._svar_link import (
    SvarFingerprint,
    SvarLink,
    _resolve_svar,
    _verify_fingerprint,
)
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

    metadata = Metadata.model_validate_json((gvl_path / "metadata.json").read_text())
    assert metadata.svar_link is not None
    assert Path(metadata.svar_link.absolute_path) == svar_path.resolve()
    assert (
        gvl_path / metadata.svar_link.relative_path
    ).resolve() == svar_path.resolve()
    expected_bytes = (svar_path / "variant_idxs.npy").stat().st_size
    assert metadata.svar_link.fingerprint.variant_idxs_bytes == expected_bytes
    assert metadata.svar_link.fingerprint.n_variants > 0


def test_resolve_svar_prefers_override(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    link = SvarLink(
        relative_path="does/not/exist",
        absolute_path="/does/not/exist",
        fingerprint=SvarFingerprint(
            n_variants=1,
            variant_idxs_bytes=(svar_path / "variant_idxs.npy").stat().st_size,
        ),
    )
    assert _resolve_svar(gvl_path, link, override=svar_path) == svar_path


def test_resolve_svar_uses_relative_path(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    metadata = Metadata.model_validate_json((gvl_path / "metadata.json").read_text())
    resolved = _resolve_svar(gvl_path, metadata.svar_link, override=None)
    assert resolved.resolve() == svar_path.resolve()


def test_resolve_svar_falls_back_to_sibling(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    sibling = gvl_path.parent / "sibling.svar"
    shutil.copytree(svar_path, sibling)
    link = SvarLink(
        relative_path="nowhere",
        absolute_path="/nowhere",
        fingerprint=SvarFingerprint(
            n_variants=1,
            variant_idxs_bytes=(sibling / "variant_idxs.npy").stat().st_size,
        ),
    )
    resolved = _resolve_svar(gvl_path, link, override=None)
    assert resolved.resolve() == sibling.resolve()


def test_resolve_svar_raises_when_not_found(tmp_path):
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()
    link = SvarLink(
        relative_path="nowhere",
        absolute_path="/nowhere",
        fingerprint=SvarFingerprint(n_variants=1, variant_idxs_bytes=1),
    )
    with pytest.raises(FileNotFoundError, match="svar="):
        _resolve_svar(gvl_path, link, override=None)


def test_verify_fingerprint_mismatch_raises(svar_dataset_paths):
    _, svar_path = svar_dataset_paths
    bogus = SvarLink(
        relative_path=str(svar_path),
        absolute_path=str(svar_path),
        fingerprint=SvarFingerprint(n_variants=999_999, variant_idxs_bytes=1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        _verify_fingerprint(svar_path, bogus)


def test_verify_fingerprint_ok(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    metadata = Metadata.model_validate_json((gvl_path / "metadata.json").read_text())
    _verify_fingerprint(svar_path, metadata.svar_link)


def test_open_dataset_via_recorded_svar_link(svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, _ = svar_dataset_paths
    ref = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset.open(gvl_path, reference=ref)
        .with_seqs("haplotypes")
        .with_tracks(False)
    )
    _ = ds[0, 0]


def test_open_dataset_after_relocation_via_override(tmp_path, svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, svar_path = svar_dataset_paths
    moved = tmp_path / "moved.svar"
    shutil.copytree(svar_path, moved)

    # Break the stored paths by relocating the dataset (so relative & absolute fail).
    moved_gvl = tmp_path / "elsewhere" / "ds.gvl"
    moved_gvl.parent.mkdir()
    shutil.copytree(gvl_path, moved_gvl)

    ref = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset.open(moved_gvl, reference=ref, svar=moved)
        .with_seqs("haplotypes")
        .with_tracks(False)
    )
    _ = ds[0, 0]


def test_open_dataset_mismatched_svar_raises(tmp_path, svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, svar_path = svar_dataset_paths
    fake = tmp_path / "fake.svar"
    shutil.copytree(svar_path, fake)
    target = fake / "variant_idxs.npy"
    target.write_bytes(target.read_bytes()[:-8])
    with pytest.raises(ValueError, match="fingerprint"):
        gvl.Dataset.open(gvl_path, svar=fake)


def test_open_dataset_legacy_symlink_layout(tmp_path, svar_dataset_paths):
    import warnings as _warnings

    import genvarloader as gvl

    gvl_path, svar_path = svar_dataset_paths
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        svar_path.resolve(), target_is_directory=True
    )

    ref = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ds = (
            gvl.Dataset.open(gvl_path, reference=ref)
            .with_seqs("haplotypes")
            .with_tracks(False)
        )
        _ = ds[0, 0]
        assert any(
            issubclass(w.category, DeprecationWarning) and "link.svar" in str(w.message)
            for w in caught
        )


def test_migrate_svar_link_upgrades_legacy_dataset(tmp_path, svar_dataset_paths):
    import warnings as _warnings

    import genvarloader as gvl

    gvl_path, svar_path = svar_dataset_paths
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        svar_path.resolve(), target_is_directory=True
    )

    gvl.migrate_svar_link(gvl_path)

    upgraded = json.loads(meta_path.read_text())
    assert upgraded.get("svar_link") is not None
    assert not (gvl_path / "genotypes" / "link.svar").exists()

    ref = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ds = (
            gvl.Dataset.open(gvl_path, reference=ref)
            .with_seqs("haplotypes")
            .with_tracks(False)
        )
        _ = ds[0, 0]
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_migrate_svar_link_is_idempotent(svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, _ = svar_dataset_paths
    before = (gvl_path / "metadata.json").read_text()
    gvl.migrate_svar_link(gvl_path)
    after = (gvl_path / "metadata.json").read_text()
    assert before == after


def test_open_after_joint_relocation_preserves_relative(tmp_path, svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, svar_path = svar_dataset_paths
    new_parent = tmp_path / "relocated"
    new_parent.mkdir()
    new_gvl = new_parent / gvl_path.name
    new_svar = new_parent / svar_path.name
    shutil.copytree(gvl_path, new_gvl)
    shutil.copytree(svar_path, new_svar)

    ref = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset.open(new_gvl, reference=ref)
        .with_seqs("haplotypes")
        .with_tracks(False)
    )
    _ = ds[0, 0]


def test_migrate_svar_link_refuses_dangling_symlink(tmp_path, svar_dataset_paths):
    import genvarloader as gvl

    gvl_path, _ = svar_dataset_paths
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        tmp_path / "does_not_exist.svar", target_is_directory=True
    )
    with pytest.raises(FileNotFoundError):
        gvl.migrate_svar_link(gvl_path)
