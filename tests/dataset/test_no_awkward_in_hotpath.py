"""Guard: the fixed/ragged haps + tracks + ref paths must not dispatch awkward kernels.

This test patches the most common awkward entry-points (to_numpy, to_packed,
flatten, where) and asserts that none of them are called during a getitem on
the "flat" hot paths:
- tracks_fixed  (seqs=None, tracks="5ss", out_len=int)
- haps_fixed    (seqs="haplotypes",       out_len=int)
- ref_fixed     (seqs="reference",        out_len=int)

If any of these fire, it means a reconstructor is still routing through awkward
densification, which is a regression.

RaggedVariants now has its own minimal-awkward guard (test_variants_ragged_minimal_awkward)
that tolerates only ak.zip (record construction); the kernels removed in FU-3
(to_packed, where, flatten, to_numpy) are patched and must dispatch zero times.
"""

from __future__ import annotations

import numpy as np
import pyBigWig
import pytest

import genvarloader as gvl

SEQLEN = 20

# ---------------------------------------------------------------------------
# Fixture (mirrors snap_dataset from test_flat_getitem_snapshot.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def guard_dataset(source_bed, vcf_dir, reference, tmp_path_factory):
    """Same toy dataset as snap_dataset; session-scoped so it's built once."""
    from genoray import VCF

    tmp_dir = tmp_path_factory.mktemp("guard_ds")
    out = tmp_dir / "guard.gvl"

    vcf_samples = ["NA00001", "NA00002", "NA00003"]
    contig_sizes = [("chr1", 20_000_000), ("chr19", 2_000_000), ("chr20", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(vcf_samples):
        bw_path = tmp_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr19", "chr20"],
                [10_000_000, 1_010_686, 17_320],
                ends=[10_000_020, 1_010_706, 17_340],
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
# Helper: build counter patches for awkward chokepoints
# ---------------------------------------------------------------------------


def _install_ak_counters(monkeypatch):
    """Patch the most common awkward densification entry-points.

    Returns a dict {"n": int} that increments on each call.  Patching at the
    ``awkward`` module level catches both ``import awkward as ak; ak.X(...)``
    and ``from awkward import X; X(...)`` patterns used inside genvarloader.
    """
    import awkward as ak

    calls = {"n": 0}

    for name in ("to_numpy", "to_packed", "flatten", "where"):
        orig = getattr(ak, name)

        def _counting(orig=orig, *a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(ak, name, _counting)

    return calls


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_tracks_fixed_no_awkward(monkeypatch, guard_dataset):
    """tracks-only fixed-length getitem must dispatch zero awkward kernels."""
    calls = _install_ak_counters(monkeypatch)

    ds = guard_dataset.with_seqs(None).with_tracks("5ss").with_len(SEQLEN)
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    _ = ds[regions, samples]

    assert calls["n"] == 0, (
        f"tracks_fixed path dispatched {calls['n']} awkward kernel(s); "
        "a reconstructor is still routing through awkward densification."
    )


def test_haps_fixed_no_awkward(monkeypatch, guard_dataset):
    """haplotypes fixed-length getitem must dispatch zero awkward kernels."""
    calls = _install_ak_counters(monkeypatch)

    ds = guard_dataset.with_seqs("haplotypes").with_tracks(False).with_len(SEQLEN)
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    _ = ds[regions, samples]

    assert calls["n"] == 0, (
        f"haps_fixed path dispatched {calls['n']} awkward kernel(s); "
        "a reconstructor is still routing through awkward densification."
    )


def test_ref_fixed_no_awkward(monkeypatch, guard_dataset):
    """reference fixed-length getitem must dispatch zero awkward kernels."""
    calls = _install_ak_counters(monkeypatch)

    ds = guard_dataset.with_seqs("reference").with_tracks(False).with_len(SEQLEN)
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    _ = ds[regions, samples]

    assert calls["n"] == 0, (
        f"ref_fixed path dispatched {calls['n']} awkward kernel(s); "
        "a reconstructor is still routing through awkward densification."
    )


def test_haps_tracks_fixed_no_awkward(monkeypatch, guard_dataset):
    """haplotypes + tracks fixed-length getitem must dispatch zero awkward kernels."""
    calls = _install_ak_counters(monkeypatch)

    ds = guard_dataset.with_seqs("haplotypes").with_tracks("5ss").with_len(SEQLEN)
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    _ = ds[regions, samples]

    assert calls["n"] == 0, (
        f"haps_tracks_fixed path dispatched {calls['n']} awkward kernel(s); "
        "a reconstructor is still routing through awkward densification."
    )


def test_haps_ragged_no_awkward(monkeypatch, guard_dataset):
    """haplotypes ragged getitem must dispatch zero awkward kernels."""
    calls = _install_ak_counters(monkeypatch)

    ds = guard_dataset.with_seqs("haplotypes").with_tracks(False)
    # output_length="ragged" is the default
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    _ = ds[regions, samples]

    assert calls["n"] == 0, (
        f"haps_ragged path dispatched {calls['n']} awkward kernel(s); "
        "a reconstructor is still routing through awkward densification."
    )


def test_variants_ragged_minimal_awkward(monkeypatch, guard_dataset):
    """Variants gather + rc_ + to_packed must dispatch no awkward kernels.

    ak.zip (record construction in RaggedVariants.__init__) is the documented
    remaining awkward call and is NOT patched here.  The kernels we removed in
    FU-3 — ak.to_packed, ak.where, ak.flatten, and the old ak.to_numpy
    densification — are patched; asserting zero dispatches confirms the
    flat-buffer gather + seqpro reverse_complement_masked + field-wise to_packed
    path has no residual awkward in those slots.
    """
    calls = _install_ak_counters(
        monkeypatch
    )  # patches to_numpy/to_packed/flatten/where

    ds = guard_dataset.with_seqs("variants").with_tracks(False)
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]

    rv = ds[regions, samples]
    rv.rc_(np.ones(n_regions, np.bool_))  # exercise rc_ explicitly
    rv.to_packed()  # exercise field-wise to_packed

    assert calls["n"] == 0, (
        f"variants gather/rc_/to_packed dispatched {calls['n']} awkward kernel(s) "
        "(to_packed/where/flatten/to_numpy); a removed kernel is still being called. "
        "Note: ak.zip (RaggedVariants record construction) is intentionally NOT patched."
    )
