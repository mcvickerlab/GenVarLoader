"""Unit tests for gvl.concat preconditions."""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from genvarloader._dataset._concat_validate import (
    ConcatInput,
    load_inputs,
    validate_concat,
    variants_fingerprint,
)
from genvarloader._dataset._svar2_link import Svar2Fingerprint, Svar2Link
from genvarloader._dataset._svar_link import SvarFingerprint, SvarLink
from genvarloader._dataset._write import Metadata
from genvarloader._fasta_cache import Fingerprint


def _mk(
    samples,
    n_regions,
    *,
    backend="pgen_vcf",
    ploidy=2,
    tracks=(),
    chroms=None,
    fingerprint=None,
):
    n = n_regions
    bed = pl.DataFrame(
        {
            "chrom": chroms or ["chr1"] * n,
            "chromStart": list(range(0, 100 * n, 100)),
            "chromEnd": list(range(50, 100 * n + 50, 100)),
        }
    )
    meta = Metadata(
        samples=list(samples),
        contigs=["chr1", "chr2"],
        n_regions=n,
        ploidy=ploidy,
    )
    return ConcatInput(
        path=None,
        meta=meta,
        bed=bed,
        n_regions=n,
        n_samples=len(samples),
        backend=backend,
        tracks=list(tracks),
        annot_tracks=[],
        has_dosages=False,
        fingerprint=fingerprint,
    )


def _write_dataset_dir(
    tmp_path: Path,
    name: str,
    *,
    samples=("a",),
    n_regions=1,
    svar_link=None,
    svar2_link=None,
    with_genotypes_dir=True,
    extra_genotypes_files=(),
) -> Path:
    """Build a minimal on-disk dataset dir matching what `load_inputs` reads."""
    ds_dir = tmp_path / name
    (ds_dir / "intervals").mkdir(parents=True)
    (ds_dir / "annot_intervals").mkdir()
    if with_genotypes_dir:
        geno_dir = ds_dir / "genotypes"
        geno_dir.mkdir()
        for fname, contents in extra_genotypes_files:
            (geno_dir / fname).write_bytes(contents)

    meta = Metadata(
        samples=list(samples),
        contigs=["chr1"],
        n_regions=n_regions,
        ploidy=2,
        svar_link=svar_link,
        svar2_link=svar2_link,
    )
    (ds_dir / "metadata.json").write_text(meta.model_dump_json())

    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * n_regions,
            "chromStart": list(range(0, 100 * n_regions, 100)),
            "chromEnd": list(range(50, 100 * n_regions + 50, 100)),
        }
    )
    bed.write_ipc(ds_dir / "input_regions.arrow")
    return ds_dir


def test_accepts_disjoint_samples_on_sample_axis():
    validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "samples")


def test_rejects_overlapping_samples_on_sample_axis():
    with pytest.raises(ValueError, match="overlapping samples"):
        validate_concat([_mk(["a", "b"], 2), _mk(["b"], 2)], "samples")


def test_rejects_mismatched_regions_on_sample_axis():
    with pytest.raises(ValueError, match="identical regions"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 3)], "samples")


def test_rejects_mismatched_samples_on_region_axis():
    with pytest.raises(ValueError, match="identical samples"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "regions")


def test_rejects_mixed_backends():
    with pytest.raises(ValueError, match="same variant source"):
        validate_concat(
            [_mk(["a"], 2, backend="pgen_vcf"), _mk(["a"], 2, backend="svar2")],
            "regions",
        )


def test_rejects_mismatched_ploidy():
    with pytest.raises(ValueError, match="ploidy"):
        validate_concat([_mk(["a"], 2, ploidy=2), _mk(["a"], 2, ploidy=1)], "regions")


def test_rejects_mismatched_track_sets():
    with pytest.raises(ValueError, match="track"):
        validate_concat(
            [_mk(["a"], 2, tracks=("atac",)), _mk(["a"], 2, tracks=("dnase",))],
            "regions",
        )


def test_rejects_single_input():
    with pytest.raises(ValueError, match="at least two"):
        validate_concat([_mk(["a"], 2)], "regions")


def test_rejects_duplicate_region_names_on_region_axis():
    a = _mk(["a"], 2)
    b = _mk(["a"], 2)  # identical coordinates -> identical derived names
    a = replace(a, bed=a.bed.with_columns(name=pl.Series(["r0", "r1"])))
    b = replace(b, bed=b.bed.with_columns(name=pl.Series(["r1", "r2"])))
    with pytest.raises(ValueError, match="duplicate region name"):
        validate_concat([a, b], "regions")


def test_identical_coordinates_without_names_are_allowed():
    """Coordinate collisions are fine; only *name* collisions raise."""
    validate_concat([_mk(["a"], 2), _mk(["a"], 2)], "regions")


# --- fix round 1, finding 2: variant-source identity must be enforced by content,
# not just by the coarse backend string. ---


def test_accepts_matching_variant_fingerprints():
    fp = Fingerprint(n_bytes_hashed=4, digest="abc", size_bytes=4)
    validate_concat(
        [_mk(["a"], 2, fingerprint=fp), _mk(["b"], 2, fingerprint=fp)], "samples"
    )


def test_rejects_mismatched_variant_fingerprints():
    fp_a = Fingerprint(n_bytes_hashed=4, digest="aaa", size_bytes=4)
    fp_b = Fingerprint(n_bytes_hashed=4, digest="bbb", size_bytes=4)
    with pytest.raises(ValueError, match="variant source"):
        validate_concat(
            [_mk(["a"], 2, fingerprint=fp_a), _mk(["a"], 2, fingerprint=fp_b)],
            "regions",
        )


def test_tolerates_missing_variant_fingerprints():
    """`None` fingerprints (e.g. tracks-only datasets) skip the identity check."""
    validate_concat(
        [_mk(["a"], 2, fingerprint=None), _mk(["b"], 2, fingerprint=None)], "samples"
    )


# --- fix round 1, finding 1: `has_dosages` and the variant fingerprint must be
# derived by resolving the real on-disk layout, not a path that gvl.write never
# produces. These exercise `load_inputs` against a real temp directory. ---


def test_load_inputs_resolves_svar_dosages_from_linked_store(tmp_path):
    """Dosages live inside the externally-linked svar store, not under the
    dataset's own `genotypes/` directory."""
    svar_dir = tmp_path / "cohort.svar"
    svar_dir.mkdir()
    (svar_dir / "dosages.npy").write_bytes(b"\x00")

    link = SvarLink(
        relative_path="../cohort.svar",
        absolute_path=str(svar_dir),
        fingerprint=SvarFingerprint(n_variants=3, variant_idxs_bytes=24),
    )
    ds_dir = _write_dataset_dir(tmp_path, "ds.gvl", svar_link=link)

    [inp] = load_inputs([ds_dir])

    assert inp.backend == "svar"
    assert inp.has_dosages is True
    assert inp.fingerprint == link.fingerprint


def test_load_inputs_svar_without_dosages_file(tmp_path):
    svar_dir = tmp_path / "cohort.svar"
    svar_dir.mkdir()  # no dosages.npy

    link = SvarLink(
        relative_path="../cohort.svar",
        absolute_path=str(svar_dir),
        fingerprint=SvarFingerprint(n_variants=3, variant_idxs_bytes=24),
    )
    ds_dir = _write_dataset_dir(tmp_path, "ds.gvl", svar_link=link)

    [inp] = load_inputs([ds_dir])

    assert inp.has_dosages is False


def test_load_inputs_resolves_svar2_dosages_from_linked_store(tmp_path):
    svar2_dir = tmp_path / "cohort.svar2"
    svar2_dir.mkdir()
    (svar2_dir / "dosages.npy").write_bytes(b"\x00")

    link = Svar2Link(
        relative_path="../cohort.svar2",
        absolute_path=str(svar2_dir),
        fingerprint=Svar2Fingerprint(n_files=1, store_bytes=1),
    )
    ds_dir = _write_dataset_dir(tmp_path, "ds.gvl", svar2_link=link)

    [inp] = load_inputs([ds_dir])

    assert inp.backend == "svar2"
    assert inp.has_dosages is True
    assert inp.fingerprint == link.fingerprint


def test_load_inputs_pgen_vcf_ignores_stray_dosages_file_and_fingerprints_variants(
    tmp_path,
):
    """Regression for the bug where `has_dosages` read `<dataset>/genotypes/
    dosages.npy` -- a path gvl.write never creates. A stray file there (e.g. left
    over from manual tinkering) must not flip `has_dosages`, and the variant
    fingerprint must come from `genotypes/variants.arrow` content."""
    ds_dir = _write_dataset_dir(
        tmp_path,
        "ds.gvl",
        extra_genotypes_files=[
            ("variants.arrow", b"variant-bytes"),
            ("dosages.npy", b"\x00"),  # stray -- pgen/vcf never writes this
        ],
    )

    [inp] = load_inputs([ds_dir])

    assert inp.backend == "pgen_vcf"
    assert inp.has_dosages is False
    assert inp.fingerprint == variants_fingerprint(ds_dir)


def test_load_inputs_tracks_only_has_no_variant_source(tmp_path):
    ds_dir = _write_dataset_dir(tmp_path, "ds.gvl", with_genotypes_dir=False)

    [inp] = load_inputs([ds_dir])

    assert inp.backend == "tracks_only"
    assert inp.has_dosages is False
    assert inp.fingerprint is None
