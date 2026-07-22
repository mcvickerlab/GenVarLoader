"""Wave B PR-B1: streaming with_seqs("variants") is byte-identical to the written path.

`StreamingDataset.with_seqs("variants")` drives `RecordStreamEngine.next_batch_variants`
(`RecordBackend::generate_variants`, Task 2's Rust core exposed via Task 3's FFI) and packs
the returned flat buffers into a `RaggedVariants`. This compares that streamed output,
cell by cell, against the SAME VCF/PGEN source written via `gvl.write` and read back with
`Dataset.with_seqs("variants")` -- two independent decoders (Rust `ChunkAssembler` for
streaming vs. Python cyvcf2/pgenlib + `dense2sparse` for the written path) that must agree
byte-for-byte on `alt`/`start`/`ilen`.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl

BACKENDS = ["svar1", "vcf", "pgen"]


def _assert_variants_cell_matches(streamed, expected, ploidy) -> int:
    """Assert the streamed cell matches the written cell hap-by-hap; return the total
    number of variants seen across all haps (so callers can guard against a vacuous
    all-empty pass -- see the module docstring's byte-identity claim)."""
    n_variants = 0
    for h in range(ploidy):
        streamed_alt = np.asarray(streamed.alt[h])
        np.testing.assert_array_equal(streamed_alt, np.asarray(expected.alt[h]))
        np.testing.assert_array_equal(
            np.asarray(streamed.start[h]), np.asarray(expected.start[h])
        )
        np.testing.assert_array_equal(
            np.asarray(streamed.ilen[h]), np.asarray(expected.ilen[h])
        )
        # A ragged hap with exactly one variant collapses `.alt[h]` to a 0-d scalar
        # (bytes) rather than a length-1 array -- `atleast_1d` normalizes both cases
        # before counting.
        n_variants += np.atleast_1d(streamed_alt).shape[0]
    return n_variants


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_variants_matches_written(streaming_case, backend):
    regions, reference, variants, written = streaming_case(backend)
    ds = written.with_seqs("variants")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variants")

    seen = set()
    total_variants = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            total_variants += _assert_variants_cell_matches(
                data[k], ds[r, s], sds.ploidy
            )
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}
    # Guard against a vacuous pass: the streaming_case fixtures carry real SNP/INS/DEL
    # variants, so a byte-identical parity check that saw zero variants everywhere would
    # be trivially (and wrongly) green.
    assert total_variants > 0
