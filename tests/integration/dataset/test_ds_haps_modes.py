"""Oracle parity for the ``annotated`` and ``reference`` output modes.

Parallel to ``test_ds_haps.py`` which covers ``with_seqs("haplotypes")`` against
the bcftools-generated consensus FASTAs in ``tests/data/consensus/``. These
tests exercise the other two reference-backed output modes.
"""

from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp


@pytest.fixture(
    scope="session",
    params=["vcf", "pgen", "svar"],
)
def base_dataset(request, phased_vcf_gvl, phased_pgen_gvl, phased_svar_gvl, ref_fasta):
    gvl_path = {
        "vcf": phased_vcf_gvl,
        "pgen": phased_pgen_gvl,
        "svar": phased_svar_gvl,
    }[request.param]
    return (
        gvl.Dataset
        .open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_tracks(False)
    )


def test_annotated_haps_match_consensus(base_dataset, consensus_dir: Path):
    """``with_seqs("annotated")`` ``.haps`` must equal the consensus oracle, and
    the parallel annotation arrays must be internally consistent.
    """
    ds = base_dataset.with_seqs("annotated")
    for region, sample in product(range(ds.n_regions), ds.samples):
        result = ds[region, sample]
        for h in range(2):
            actual = sp.cast_seqs(result.haps[h])
            fpath = f"source_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(consensus_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                err_msg=f"haps mismatch region={region} sample={sample} h={h}",
            )

            v_idxs = np.asarray(result.var_idxs[h])
            r_coords = np.asarray(result.ref_coords[h])
            # Shape parity: every nucleotide has a variant index and ref coord.
            assert v_idxs.shape == r_coords.shape == actual.shape, (
                f"shape mismatch region={region} sample={sample} h={h}: "
                f"haps={actual.shape} var_idxs={v_idxs.shape} ref_coords={r_coords.shape}"
            )
            # Invariant: positions padded beyond reference bounds carry no
            # variant. ``ref_coords == -1`` is the documented sentinel for
            # out-of-reference padding.
            ref_pad = r_coords == -1
            if ref_pad.any():
                assert np.all(v_idxs[ref_pad] == -1), (
                    "var_idxs must be -1 at out-of-reference padded positions "
                    f"(region={region} sample={sample} h={h})"
                )


def test_reference_mode_returns_unaltered_reference(base_dataset, ref_fasta):
    """``with_seqs("reference")`` returns the bare reference slice, identical
    across samples (variants are ignored)."""
    ds = base_dataset.with_seqs("reference")
    with pysam.FastaFile(str(ref_fasta)) as f:
        for region in range(ds.n_regions):
            chrom, start, end, _ = ds.regions.select(
                "chrom", "chromStart", "chromEnd", "strand"
            ).row(region)
            desired = sp.cast_seqs(f.fetch(chrom, start, end).upper())
            # Sample-independence: every sample yields the same reference slice.
            seen = None
            for sample in ds.samples:
                actual = sp.cast_seqs(ds[region, sample])
                np.testing.assert_equal(
                    actual,
                    desired,
                    err_msg=(
                        f"reference mismatch region={region} sample={sample} "
                        f"coords={chrom}:{start + 1}-{end}"
                    ),
                )
                if seen is None:
                    seen = actual
                else:
                    np.testing.assert_equal(
                        actual,
                        seen,
                        err_msg=(
                            f"reference mode varies across samples at region={region}"
                        ),
                    )
