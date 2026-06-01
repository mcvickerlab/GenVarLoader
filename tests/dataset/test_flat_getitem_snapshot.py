"""Byte-identical characterization gate for the flat-buffer getitem refactor.

First run (no snapshot present) writes reference .npz files from the CURRENT
code; commit them. Every later run asserts getitem output is byte-identical.

Fixture notes
-------------
``snap_dataset`` is a session-scoped fixture that writes a fresh toy dataset
(phased VCF + "5ss" BigWig track + reference) into a tmp directory, then opens
it.  It uses the same ingredients as the ``base_ds`` fixture in
``tests/dataset/test_with_methods.py`` so there is an established precedent.

Track name: "5ss"   (only track written by this fixture)
SEQLEN: 20          (region length = 20 bp, max_jitter=2, jitter=0 → max=24)

Cases kept (all 10)
-------------------
haps_ragged         seqs="haplotypes",                   out_len="ragged"
haps_fixed          seqs="haplotypes",                   out_len=SEQLEN
haps_variable       seqs="haplotypes",                   out_len="variable"
annot_fixed         seqs="annotated",                    out_len=SEQLEN
tracks_ragged       seqs=None, tracks="5ss",             out_len="ragged"
tracks_fixed        seqs=None, tracks="5ss",             out_len=SEQLEN
ref_fixed           seqs="reference",                    out_len=SEQLEN
haps_tracks_fixed   seqs="haplotypes", tracks="5ss",     out_len=SEQLEN
haps_tracks_ragged  seqs="haplotypes", tracks="5ss",     out_len="ragged"
variants_ragged     seqs="variants",                     out_len="ragged"

No cases were dropped; all are satisfiable with the fixture above.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig
import pytest

import genvarloader as gvl
from genvarloader import RaggedVariants
from genvarloader._ragged import RaggedAnnotatedHaps
from genvarloader._types import AnnotatedHaps
from seqpro.rag import Ragged

SNAP = Path(__file__).parent / "_snapshots"
SEQLEN = 20

CASES = [
    ("haps_ragged", dict(seqs="haplotypes"), "ragged"),
    ("haps_fixed", dict(seqs="haplotypes"), SEQLEN),
    ("haps_variable", dict(seqs="haplotypes"), "variable"),
    ("annot_fixed", dict(seqs="annotated"), SEQLEN),
    ("tracks_ragged", dict(seqs=None, tracks="5ss"), "ragged"),
    ("tracks_fixed", dict(seqs=None, tracks="5ss"), SEQLEN),
    ("ref_fixed", dict(seqs="reference"), SEQLEN),
    ("haps_tracks_fixed", dict(seqs="haplotypes", tracks="5ss"), SEQLEN),
    ("haps_tracks_ragged", dict(seqs="haplotypes", tracks="5ss"), "ragged"),
    ("variants_ragged", dict(seqs="variants"), "ragged"),
]

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def snap_dataset(source_bed, vcf_dir, reference, tmp_path_factory):
    """Phased VCF dataset with a "5ss" BigWig track, opened with a reference.

    Mirrors the ``base_ds`` fixture in ``tests/dataset/test_with_methods.py``.
    Opened with default settings (output_length="ragged", sequence_type="haplotypes",
    jitter=0, max_jitter=2, deterministic=True, rc_neg=True).
    """
    from genoray import VCF

    tmp_dir = tmp_path_factory.mktemp("snap_ds")
    out = tmp_dir / "snap.gvl"

    vcf_samples = ["s0", "s1", "s2"]
    # Header lengths are generous upper bounds for the regions in source.bed.
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(vcf_samples):
        bw_path = tmp_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            # One short interval per contig region in source.bed; values differ
            # per sample. Mirrors base_ds in tests/dataset/test_with_methods.py.
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[value, value, value, value],
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
# Helpers
# ---------------------------------------------------------------------------


def _build(ds, seqs, tracks=None, out_len="ragged"):
    """Apply with_seqs / with_tracks / with_len and return configured dataset.

    Tracks are explicitly controlled:
    - If ``tracks`` is a string, activate that track (and set seqs accordingly).
    - If ``tracks`` is None and ``seqs`` is not None (seqs-only case), explicitly
      disable tracks with ``with_tracks(False)`` so the dataset returns ONLY the
      sequence output — not a (seqs, tracks) tuple.
    - If ``seqs`` is None (tracks-only case), tracks must be a string; seqs-off
      is handled by ``with_seqs(None)``.
    """
    ds = ds.with_seqs(seqs)
    if tracks is not None:
        # Activate the requested track (string name).
        ds = ds.with_tracks(tracks)
    elif seqs is not None:
        # Seqs-only case: explicitly turn tracks OFF so output is seqs alone,
        # not a (seqs, tracks) tuple from the default-on track state.
        ds = ds.with_tracks(False)
    # If seqs is None and tracks is None that would be invalid; our CASES never
    # reach that state (tracks-only cases always supply a track name).
    if out_len == "ragged":
        return ds
    if out_len == "variable":
        return ds.with_len("variable")
    return ds.with_len(out_len)


def _flatten_output(obj) -> dict[str, np.ndarray]:
    """Normalize any getitem return into a dict of plain ndarrays."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, tuple):
        for i, o in enumerate(obj):
            out.update({f"{i}_{k}": v for k, v in _flatten_output(o).items()})
    # RaggedAnnotatedHaps is a separate dataclass (not a Ragged subclass); branch before Ragged by convention.
    elif isinstance(obj, RaggedAnnotatedHaps):
        out["haps_data"] = np.asarray(obj.haps.data)
        out["haps_off"] = np.asarray(obj.haps.offsets)
        out["vidx_data"] = np.asarray(obj.var_idxs.data)
        out["vidx_off"] = np.asarray(obj.var_idxs.offsets)
        out["pos_data"] = np.asarray(obj.ref_coords.data)
        out["pos_off"] = np.asarray(obj.ref_coords.offsets)
    elif isinstance(obj, Ragged):
        out["data"] = np.asarray(obj.data)
        out["off"] = np.asarray(obj.offsets)
    elif isinstance(obj, AnnotatedHaps):
        out["haps"] = np.asarray(obj.haps)
        out["var_idxs"] = np.asarray(obj.var_idxs)
        out["ref_coords"] = np.asarray(obj.ref_coords)
    elif isinstance(obj, RaggedVariants):
        # Serialize each field to plain arrays. Numeric fields -> data+offsets.
        # alt/ref are doubly-nested bytes -> leaf bytes + both offset levels.
        for fld in sorted(obj.fields):
            v = obj[fld]
            if fld in ("alt", "ref"):
                lay = v.layout
                out[f"{fld}_group_off"] = np.asarray(lay.content.offsets)
                out[f"{fld}_allele_off"] = np.asarray(lay.content.content.offsets)
                out[f"{fld}_bytes"] = np.asarray(lay.content.content.content.data)
                out[f"{fld}_ploidy"] = np.asarray(lay.size)
            else:
                out[f"{fld}_data"] = np.asarray(v.data)
                out[f"{fld}_off"] = np.asarray(v.offsets)
    elif isinstance(obj, np.ndarray):
        out["arr"] = obj
    else:
        raise NotImplementedError(
            f"_flatten_output: unhandled getitem return type {type(obj).__name__!r}; "
            "add an explicit branch before snapshotting this case."
        )
    return out


# ---------------------------------------------------------------------------
# Snapshot gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,view,out_len", [(c[0], c[1], c[2]) for c in CASES])
def test_getitem_snapshot(snap_dataset, name, view, out_len):
    ds = _build(snap_dataset, view["seqs"], view.get("tracks"), out_len)
    n_regions = min(8, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]
    result = _flatten_output(ds[regions, samples])

    SNAP.mkdir(exist_ok=True)
    path = SNAP / f"{name}.npz"
    if not path.exists():
        np.savez(path, **result)
        pytest.skip(f"wrote snapshot {path.name}; commit it and re-run")

    ref = np.load(path)
    assert set(ref.files) == set(result), (
        f"{name}: key drift — expected {set(ref.files)}, got {set(result)}"
    )
    for k in ref.files:
        np.testing.assert_array_equal(result[k], ref[k], err_msg=f"{name}:{k}")
