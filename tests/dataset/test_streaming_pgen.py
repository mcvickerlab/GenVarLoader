"""Task 11: `_PgenBackend` (Python) + ladder wiring + engine constructor
(issue #276).

Mirrors `test_streaming_vcf.py`'s Task 5/6 tests exactly, over the PGEN
file-set fixture instead of a VCF. End-to-end streaming-vs-written parity is
Task 12; this only proves the pyclass/backend wire up correctly:
`RecordStreamEngine("pgen", ...)` constructs and drains cleanly, and
`StreamingDataset(variants=<pgen>)` constructs via `_PgenBackend` with the
right header metadata.
"""

from __future__ import annotations

import numpy as np


def test_record_stream_engine_pgen_yields_then_none(streaming_pgen_fixture):
    from genvarloader.genvarloader import RecordStreamEngine

    f = streaming_pgen_fixture
    eng = RecordStreamEngine(
        "pgen",
        str(f.pgen),
        f.sample_names,
        f.ploidy,
        ["chr1"],
        [f.chr1_ref_bytes],
        [0],
        [[0]],
        [[len(f.chr1_ref_bytes)]],
        [0],
        [f.n_samples],
        None,  # fasta_path=None -- matches gvl.write's PGEN parity (no read-time left-align)
        ord("N"),
        False,
        32,
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


def test_streaming_dataset_accepts_pgen(streaming_pgen_fixture):
    """`StreamingDataset(variants=<pgen>)` constructs via `_PgenBackend`
    instead of raising `NotImplementedError`. End-to-end read parity is
    Task 12; this only proves construction + header metadata wiring."""
    import genvarloader as gvl

    f = streaming_pgen_fixture
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.pgen)
    ).with_seqs("haplotypes")
    assert sds.n_samples == f.n_samples
    assert sds.ploidy == 2
