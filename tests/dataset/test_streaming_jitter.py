"""Property tests for StreamingDataset read-time jitter (issue #277, Task 3).

Translation-only jitter: a per-region rng draw translates the region's read
window (clamped to the contig), keeping the window size unchanged. This is a
reproducible, rng-seeded augmentation -- NOT byte-parity with a written
`Dataset` (see `StreamingDataset.to_iter`'s docstring). These tests are
property tests, not parity tests: reproducibility for a fixed rng, divergence
across rngs, and fixed-length shape preservation.
"""

import numpy as np
import pytest

import genvarloader as gvl


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_reproducible_same_rng(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)

    def mk():
        return gvl.StreamingDataset(
            regions, reference=reference, variants=variants
        ).with_settings(jitter=8, rng=0, deterministic=False)

    a = [np.asarray(d.to_padded(b"N")) for d, *_ in mk().to_iter(batch_size=4)]
    b = [np.asarray(d.to_padded(b"N")) for d, *_ in mk().to_iter(batch_size=4)]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_different_rng_differs(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)

    def mk(seed):
        return gvl.StreamingDataset(
            regions, reference=reference, variants=variants
        ).with_settings(jitter=8, rng=seed, deterministic=False)

    a = np.concatenate(
        [np.asarray(d.to_padded(b"N")).ravel() for d, *_ in mk(0).to_iter(4)]
    )
    b = np.concatenate(
        [np.asarray(d.to_padded(b"N")).ravel() for d, *_ in mk(1).to_iter(4)]
    )
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_fixed_length_output_shape(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_len(16)
        .with_settings(jitter=8, rng=0, deterministic=False)
    )
    for data, *_ in sds.to_iter(batch_size=4):
        assert data.to_padded(b"N").shape[-1] == 16
