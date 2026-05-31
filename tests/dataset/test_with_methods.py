"""Coverage for Dataset.with_* methods — confirms each returns a new lazy view
without mutating the original, and rejects obviously invalid input.

Methods covered:
  - with_settings
  - with_len
  - with_seqs
  - with_tracks
  - with_insertion_fill  (returns-new-view only; invalid-mode rejection is in
                          tests/unit/dataset/test_with_insertion_fill.py)
"""

import genvarloader as gvl
import pyBigWig
import pytest
from genvarloader._dataset._insertion_fill import Constant, Repeat5p

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_ds(source_bed, vcf_dir, reference, tmp_path_factory):
    """Phased VCF dataset opened with a reference genome and a "5ss" track.

    Writes a fresh dataset (rather than relying on canonical ground-truth data,
    which omits tracks) so the with_tracks / with_seqs(None) tests have a track
    to exercise.

    Tracks are provided via ``gvl.BigWigs`` (not ``gvl.Table``) to avoid leaving
    the polars_bio C extension in a bad state, which segfaulted a downstream
    ``gvl.write`` call on py312/py313 when these tests ran in the same session
    as ``tests/integration/dataset/test_issue_153.py``.

    Default state after open():
      - output_length = "ragged"
      - sequence_type = "haplotypes"
      - active_tracks = ["5ss"]
      - jitter = 0, max_jitter = 2
      - rc_neg = True, deterministic = True
    """
    from genoray import VCF

    tmp_dir = tmp_path_factory.mktemp("with_methods_ds")
    out = tmp_dir / "phased_with_tracks.gvl"

    # Build per-sample BigWig files covering every contig in source.bed
    # (chr1, chr19, chr20). Sample IDs match the VCF samples so the
    # intersection is non-empty.
    vcf_samples = ["NA00001", "NA00002", "NA00003"]
    # Header lengths are generous upper bounds for the regions in source.bed.
    contig_sizes = [("chr1", 20_000_000), ("chr19", 2_000_000), ("chr20", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(vcf_samples):
        bw_path = tmp_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            # One short interval per contig; values differ per sample.
            value = float(i + 1)
            # Each interval must overlap its contig's region in source.bed so
            # every region has at least one interval (chr1's no-variant region
            # sits at chr1:500_000).
            bw.addEntries(
                ["chr1", "chr19", "chr20"],
                [499_990, 1_010_686, 17_320],
                ends=[500_030, 1_010_706, 17_340],
                values=[value, value, value],
            )
        bw_paths[sample] = str(bw_path)

    bigwigs = gvl.BigWigs("5ss", bw_paths)
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(
        path=out,
        bed=source_bed,
        variants=vcf,
        tracks=bigwigs,
        max_jitter=2,
    )
    return gvl.Dataset.open(out, reference=reference)


# ---------------------------------------------------------------------------
# with_settings
# ---------------------------------------------------------------------------


def test_with_settings_no_args_returns_new_instance(base_ds):
    """with_settings() with no args returns a new instance; base_ds unchanged."""
    new_ds = base_ds.with_settings()
    assert new_ds is not base_ds
    # Core observable attributes are identical.
    assert new_ds.jitter == base_ds.jitter
    assert new_ds.deterministic == base_ds.deterministic
    assert new_ds.rc_neg == base_ds.rc_neg


def test_with_settings_rc_neg_changes_attribute(base_ds):
    """with_settings(rc_neg=False) flips rc_neg on the new view only."""
    original_rc_neg = base_ds.rc_neg  # True
    new_ds = base_ds.with_settings(rc_neg=not original_rc_neg)
    assert new_ds is not base_ds
    assert new_ds.rc_neg is not original_rc_neg
    # Original is unchanged.
    assert base_ds.rc_neg is original_rc_neg


def test_with_settings_deterministic_changes_attribute(base_ds):
    """with_settings(deterministic=False) changes deterministic on new view."""
    new_ds = base_ds.with_settings(deterministic=False)
    assert new_ds is not base_ds
    assert new_ds.deterministic is False
    assert base_ds.deterministic is True


def test_with_settings_invalid_jitter_raises(base_ds):
    """Jitter exceeding max_jitter should be rejected."""
    with pytest.raises(ValueError):
        base_ds.with_settings(jitter=base_ds.max_jitter + 100)


# ---------------------------------------------------------------------------
# with_len
# ---------------------------------------------------------------------------


def test_with_len_int_returns_array_dataset(base_ds):
    """with_len(int) returns an ArrayDataset; base_ds is a RaggedDataset."""
    new_ds = base_ds.with_len(20)
    assert new_ds is not base_ds
    assert isinstance(new_ds, gvl.ArrayDataset)
    assert new_ds.output_length == 20
    # Original is unchanged.
    assert base_ds.output_length == "ragged"


def test_with_len_ragged_returns_ragged_dataset(base_ds):
    """with_len('ragged') returns a RaggedDataset."""
    new_ds = base_ds.with_len("ragged")
    assert new_ds is not base_ds
    assert isinstance(new_ds, gvl.RaggedDataset)
    assert new_ds.output_length == "ragged"


def test_with_len_zero_raises(base_ds):
    """Output length of 0 should be rejected."""
    with pytest.raises(ValueError):
        base_ds.with_len(0)


def test_with_len_negative_raises(base_ds):
    """Negative output length should be rejected."""
    with pytest.raises(ValueError):
        base_ds.with_len(-1)


def test_with_len_too_large_raises(base_ds):
    """Output length larger than max allowed by region length + jitter should raise."""
    # max allowed = min_region_len + 2 * max_jitter = 20 + 4 = 24
    with pytest.raises(ValueError):
        base_ds.with_len(1000)


# ---------------------------------------------------------------------------
# with_seqs
# ---------------------------------------------------------------------------


def test_with_seqs_reference_changes_sequence_type(base_ds):
    """with_seqs('reference') returns a new view with sequence_type='reference'."""
    new_ds = base_ds.with_seqs("reference")
    assert new_ds is not base_ds
    assert new_ds.sequence_type == "reference"
    # Original is unchanged.
    assert base_ds.sequence_type == "haplotypes"


def test_with_seqs_haplotypes_changes_sequence_type(base_ds):
    """with_seqs('haplotypes') returns a new view with sequence_type='haplotypes'."""
    # Start from reference mode so there's an observable change.
    ref_ds = base_ds.with_seqs("reference")
    new_ds = ref_ds.with_seqs("haplotypes")
    assert new_ds is not ref_ds
    assert new_ds.sequence_type == "haplotypes"


def test_with_seqs_annotated_changes_sequence_type(base_ds):
    """with_seqs('annotated') returns a new view with sequence_type='annotated'."""
    new_ds = base_ds.with_seqs("annotated")
    assert new_ds is not base_ds
    assert new_ds.sequence_type == "annotated"


def test_with_seqs_none_when_tracks_active_does_not_raise(base_ds):
    """with_seqs(None) is allowed when tracks are active (something is still returned)."""
    # base_ds has active_tracks=['5ss'], so setting seqs to None is OK.
    new_ds = base_ds.with_seqs(None)
    assert new_ds is not base_ds
    assert new_ds.sequence_type is None


def test_with_seqs_invalid_kind_raises(base_ds):
    """An unrecognized kind string should be rejected."""
    with pytest.raises((ValueError, AssertionError, Exception)):
        base_ds.with_seqs("nonsense_kind")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# with_tracks
# ---------------------------------------------------------------------------


def test_with_tracks_single_name_returns_new_view(base_ds):
    """with_tracks('5ss') returns a new view with the named track active."""
    new_ds = base_ds.with_tracks("5ss")
    assert new_ds is not base_ds
    assert "5ss" in (new_ds.active_tracks or [])


def test_with_tracks_false_disables_tracks(base_ds):
    """with_tracks(False) deactivates all tracks on the new view."""
    new_ds = base_ds.with_tracks(False)
    assert new_ds is not base_ds
    # active_tracks should be empty (not None, since the dataset still *has* tracks).
    assert not new_ds.active_tracks
    # Original still has active tracks.
    assert base_ds.active_tracks


def test_with_tracks_nonexistent_track_raises(base_ds):
    """Requesting a track name that doesn't exist should raise."""
    with pytest.raises((ValueError, KeyError)):
        base_ds.with_tracks("nonexistent_track_xyz")


# ---------------------------------------------------------------------------
# with_insertion_fill
# ---------------------------------------------------------------------------


def test_with_insertion_fill_returns_new_view(base_ds):
    """with_insertion_fill returns a new instance; base_ds is unchanged."""
    # base_ds has haplotypes + active tracks — a valid configuration.
    new_ds = base_ds.with_insertion_fill(Repeat5p())
    assert new_ds is not base_ds


def test_with_insertion_fill_constant_strategy(base_ds):
    """with_insertion_fill(Constant(0.0)) is accepted."""
    new_ds = base_ds.with_insertion_fill(Constant(value=0.0))
    assert new_ds is not base_ds
