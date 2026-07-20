"""Task 5: `RecordStreamEngine`'s Python constructor + FFI seam (issue #276).

Byte-identical end-to-end parity against `gvl.write`+`gvl.Dataset.open` is
Task 8; this only proves the pyclass wires up correctly -- construct it with
`source_kind="vcf"` over the Task 4 VCF fixture and drain `next_batch()`
until it cleanly returns `None`.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_record_stream_engine_vcf_yields_then_none(streaming_vcf_fixture):
    from genvarloader.genvarloader import RecordStreamEngine

    f = streaming_vcf_fixture
    eng = RecordStreamEngine(
        "vcf",
        str(f.vcf),
        f.sample_names,
        f.ploidy,
        ["chr1"],
        [f.chr1_ref_bytes],
        [0],
        [[0]],
        [[len(f.chr1_ref_bytes)]],
        [0],
        [f.n_samples],
        None,  # fasta_path=None -- matches gvl.write's VCF parity (no read-time left-align)
        ord("N"),
        False,
        32,
        -1,  # output_length: ragged
    )

    batches = []
    while (b := eng.next_batch()) is not None:
        batches.append(b)

    assert len(batches) >= 1
    data, offsets = batches[0]
    assert data.dtype == np.uint8
    assert offsets.dtype == np.int64

    # Exhaustion must be clean/idempotent, not a hang or a second, different result.
    assert eng.next_batch() is None


def test_streaming_dataset_accepts_vcf(streaming_vcf_fixture):
    """Task 6: `StreamingDataset(variants=<vcf>)` constructs via `_VcfBackend`
    instead of raising `NotImplementedError`. End-to-end read parity is
    Task 8; this only proves construction + header metadata wiring."""
    import genvarloader as gvl

    f = streaming_vcf_fixture
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.vcf)
    ).with_seqs("haplotypes")
    assert sds.n_samples == f.n_samples
    assert sds.ploidy == 2


def test_record_stream_engine_pgen_source_kind_rejects_non_pgen_ploidy(
    streaming_vcf_fixture,
):
    """`source_kind="pgen"` is implemented (Task 11) over a real PGEN
    `WindowFiller`, not a stub -- but PGEN is diploid-only, so a non-2 ploidy
    must be rejected loudly rather than silently desyncing the engine's
    per-hap CSR indexing from the filler's hardwired 2-hap-per-sample layout
    (see `src/record_stream/engine.rs`'s `"pgen"` arm). This is exercised with
    a VCF fixture's ploidy/sample_names (not a real PGEN path) purely to drive
    the ploidy guard, which fires before the (unrelated) `.pgen`/`.pvar` file
    is ever touched -- PGEN-specific construction + read behavior is covered
    by `test_streaming_pgen.py`'s `streaming_pgen_fixture` tests."""
    from genvarloader.genvarloader import RecordStreamEngine

    f = streaming_vcf_fixture
    with pytest.raises(ValueError, match="ploidy=2"):
        RecordStreamEngine(
            "pgen",
            str(f.vcf),
            f.sample_names,
            f.ploidy + 1,
            ["chr1"],
            [f.chr1_ref_bytes],
            [0],
            [[0]],
            [[len(f.chr1_ref_bytes)]],
            [0],
            [f.n_samples],
            None,
            ord("N"),
            False,
            32,
            -1,  # output_length: ragged
        )
