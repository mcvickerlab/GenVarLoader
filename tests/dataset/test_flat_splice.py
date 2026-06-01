"""Byte-identity tests for the flat-buffer spliced reconstruction paths (Task 9).

Verifies that the spliced `_Flat` outputs are byte-identical to values derived
from unspliced queries for Ref and Haps reconstructors, going through
`_query._getitem_spliced` → `_regroup`.

We do NOT use the snapshot harness because the snapshot fixture lacks a
`splice_id` column.  Instead we derive expected bytes analytically:
the spliced output for a transcript equals the concatenation of the
corresponding unspliced exon sequences in splice order — using the same
gvl code, so any variants applied in the haplotype path are preserved.

Setup
-----
Uses the pre-built ``phased_dataset.vcf.gvl`` on-disk fixture (10 regions,
3 samples, ploidy=2, SEQLEN=20) with the ``reference`` session fixture.
We inject a ``transcript_id`` column by replacing ``_full_bed`` via
``dataclasses.replace``, grouping regions 0+1 → T1, regions 2+3 → T2,
then call ``with_settings(splice_info="transcript_id")`` to enable splicing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from seqpro.rag import Ragged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_s1(r: Ragged) -> np.ndarray:
    """Flatten Ragged data to a 1-D S1 array (handles uint8 or S1 dtype)."""
    data = np.asarray(r.data)
    if data.dtype.kind in ("u", "V"):
        return data.view("S1")
    return data.ravel().view("S1")


def _row(r: Ragged, i: int) -> np.ndarray:
    """Extract row i from a 1-D-ragged Ragged as a 1-D S1 array."""
    s = int(r.offsets[i])
    e = int(r.offsets[i + 1])
    return _flat_s1(r)[s:e]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ref_spliced_flat_byte_identity(phased_vcf_gvl, reference):
    """Spliced Ref flat output is byte-identical to concatenated unspliced bytes.

    Groups regions 0+1 → T1, regions 2+3 → T2.  Verifies both transcripts
    for sample 0 (reference is sample-agnostic; sample index just selects a
    column in the 2-D dataset grid).
    """
    ds = gvl.Dataset.open(phased_vcf_gvl, reference=reference)
    ref_ds = ds.with_seqs("reference").with_tracks(False)

    n = 4
    sub_bed = ref_ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"])
    )
    patched = replace(ref_ds, _full_bed=sub_bed)
    spliced = patched.with_settings(splice_info="transcript_id")

    assert spliced.is_spliced
    assert spliced.shape == (2, 3)  # 2 transcripts × 3 samples

    # Unspliced fixed-length reference for exons 0..3, sample 0.
    unsp = ref_ds.with_len(20)
    # unsp[exon, sample] → (20,) ndarray of S1
    u = [unsp[i, 0] for i in range(n)]

    # Expected spliced bytes = concat exon0+exon1 (T1) or exon2+exon3 (T2).
    exp_t1 = np.concatenate([u[0].ravel(), u[1].ravel()])
    exp_t2 = np.concatenate([u[2].ravel(), u[3].ravel()])

    # spliced[transcript, sample] → squozen ndarray (40,) of S1.
    t1_s0 = spliced[0, 0]
    t2_s0 = spliced[1, 0]

    np.testing.assert_array_equal(t1_s0.ravel(), exp_t1, err_msg="T1 ref spliced bytes")
    np.testing.assert_array_equal(t2_s0.ravel(), exp_t2, err_msg="T2 ref spliced bytes")


def test_haps_spliced_flat_byte_identity(phased_vcf_gvl, reference):
    """Spliced Haps flat output is byte-identical to concatenated unspliced haplotype bytes.

    Groups regions 0+1 → T1, regions 2+3 → T2.  Verifies haplotype 0 and
    haplotype 1 of transcript T1 for sample 0.
    """
    ds = gvl.Dataset.open(phased_vcf_gvl, reference=reference)
    hap_ds = ds.with_seqs("haplotypes").with_tracks(False)

    n = 4
    sub_bed = hap_ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"])
    )
    patched = replace(hap_ds, _full_bed=sub_bed)
    spliced = patched.with_settings(splice_info="transcript_id")

    assert spliced.is_spliced
    assert spliced.shape == (2, 3)  # 2 transcripts × 3 samples

    # Unspliced fixed-length haplotypes for exons 0+1, sample 0.
    unsp = hap_ds.with_len(20)
    u0_s0 = unsp[0, 0]  # exon 0, sample 0 → (2, 20) ndarray (ploidy, length)
    u1_s0 = unsp[1, 0]  # exon 1, sample 0 → (2, 20) ndarray (ploidy, length)

    # Expected: concat exon0+exon1 for each haplotype.
    exp_h0 = np.concatenate([u0_s0[0].ravel(), u1_s0[0].ravel()])
    exp_h1 = np.concatenate([u0_s0[1].ravel(), u1_s0[1].ravel()])

    # spliced[0, 0] = T1, sample 0 → Ragged (2 haplotypes, ~l).
    t1_s0 = spliced[0, 0]
    assert isinstance(t1_s0, Ragged), f"Expected Ragged, got {type(t1_s0)}"
    assert t1_s0.shape == (2, None)

    np.testing.assert_array_equal(
        _row(t1_s0, 0), exp_h0, err_msg="T1 s0 h0 spliced bytes"
    )
    np.testing.assert_array_equal(
        _row(t1_s0, 1), exp_h1, err_msg="T1 s0 h1 spliced bytes"
    )
