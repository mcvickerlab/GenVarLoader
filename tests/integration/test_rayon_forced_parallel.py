"""Forced-parallel dispatch must be byte-identical to the serial size-gated path.

GVL_FORCE_PARALLEL=1 makes should_parallelize() return True, so dataset[...]
runs the real rayon path end-to-end on the small test corpus — coverage the
tiny-golden parity suite cannot reach.

Note: parametrized only for "haplotypes" because RaggedVariants.data is an
object-dtype array of dicts that np.testing.assert_array_equal cannot compare
element-wise; the haplotypes path fully covers the rayon reconstruct+track path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig
import pytest
from genoray import VCF

import genvarloader as gvl


@pytest.fixture()
def variant_track_dataset(source_bed, vcf_dir, reference, tmp_path: Path):
    """A haplotypes+track dataset whose getitem hits reconstruct + intervals_to_tracks."""
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(["s0", "s1", "s2"]):
        bw_path = tmp_path / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[value, value, value, value],
            )
        bw_paths[sample] = str(bw_path)
    out = tmp_path / "ds.gvl"
    gvl.write(
        path=out,
        bed=source_bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("5ss", bw_paths),
        max_jitter=2,
    )
    return out


def _materialize(ds):
    """Reduce a getitem result to a list of numpy arrays for comparison.

    For ragged outputs we compare BOTH the flat data buffer and the offsets
    (row boundaries), so a same-bytes/different-layout difference cannot slip
    through the byte-identical equivalence check.
    """
    out = ds[:, :]
    items = out if isinstance(out, tuple) else (out,)
    arrays = []
    for it in items:
        # Ragged-like objects expose .data (flat buffer) and .offsets (row
        # boundaries); dense arrays are ndarrays already.
        arrays.append(np.asarray(getattr(it, "data", it)))
        offsets = getattr(it, "offsets", None)
        if offsets is not None:
            arrays.append(np.asarray(offsets))
    return arrays


@pytest.mark.parametrize("seq_kind", ["haplotypes"])
def test_forced_parallel_matches_serial(
    variant_track_dataset, reference, monkeypatch, seq_kind
):
    def open_ds():
        return gvl.Dataset.open(variant_track_dataset, reference=reference).with_seqs(
            seq_kind
        )

    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    serial = _materialize(open_ds())

    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")
    parallel = _materialize(open_ds())

    assert len(serial) == len(parallel)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_equal(s, p)
