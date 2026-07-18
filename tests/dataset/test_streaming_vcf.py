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


def test_record_stream_engine_pgen_source_kind_not_implemented(streaming_vcf_fixture):
    """`source_kind="pgen"` is a deliberate stub until Task 10 fills in a PGEN
    `WindowFiller` -- must raise `NotImplementedError`, not silently build a
    broken (or VCF-backed) engine."""
    from genvarloader.genvarloader import RecordStreamEngine

    f = streaming_vcf_fixture
    with pytest.raises(NotImplementedError):
        RecordStreamEngine(
            "pgen",
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
            None,
            ord("N"),
            False,
            32,
        )
