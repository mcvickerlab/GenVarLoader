"""Task 2 (issue #277, Wave A): fixed-length (`with_len(L)`) haplotype output
must be byte-identical to the written oracle for all three streaming
backends (SVAR1, VCF, PGEN), threaded through the shared Rust
`generate_batch_core` and both stream engines.

`length` is chosen <= the smallest region length across all three
`streaming_case` backends (the SVAR1 multi-contig fixture uses 20bp sliding
windows; VCF/PGEN fixtures span a single 250bp contig) so `written.with_len
(length)` never trips the dataset's max-output-length-vs-region-length guard
(`Dataset.with_len`, `_impl.py`) for any backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl

BACKENDS = ["svar1", "vcf", "pgen"]


def _assert_cell_matches(streamed, expected, ploidy: int, length: int) -> None:
    """Per-haplotype comparison (mirrors `test_streaming_vcf_parity.py`'s
    `_assert_cell_matches` / `test_streaming_parity.py`'s inline per-hap
    loop), plus the fixed-length assertion this task adds: every hap must be
    exactly `length` long (no raggedness survives `with_len`)."""
    for h in range(ploidy):
        got = np.asarray(streamed[h])
        assert got.shape[-1] == length, (
            f"hap {h}: expected fixed length {length}, got shape {got.shape}"
        )
        np.testing.assert_array_equal(got, np.asarray(expected[h]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", [10, 16])
def test_fixed_length_matches_written(streaming_case, backend, length):
    regions, reference, variants, written = streaming_case(backend)

    ds = written.with_len(length).with_seqs("haplotypes")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_len(length)
        .with_seqs("haplotypes")
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            r, s = int(r_idx[row]), int(s_idx[row])
            _assert_cell_matches(data[row], ds[r, s], sds.ploidy, length)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_ragged_default_unaffected(streaming_case, backend):
    """Sanity companion: the default (`with_len` never called, i.e. "ragged")
    must remain untouched by this task -- `output_length=-1` internally, same
    as pre-Task-2 behavior. Regression coverage for this lives in
    `test_streaming_parity.py`/`test_streaming_vcf_parity.py`; this is a
    same-file quick check that `streaming_case` itself round-trips ragged
    output correctly for all three backends."""
    regions, reference, variants, written = streaming_case(backend)

    ds = written.with_seqs("haplotypes")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            r, s = int(r_idx[row]), int(s_idx[row])
            expected = ds[r, s]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[row][h]), np.asarray(expected[h])
                )
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }
