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


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_independent_of_sample_chunking(streaming_case, backend):
    """Regression test (issue #277 review finding): jitter offsets must be drawn
    ONCE per region, not once per `_plan()` job. `_plan()` re-yields the same
    region-window `r_idx` once per sample chunk whenever `n_samples >
    _window_samples` (cohort-scale memory chunking), so a per-job draw gives the
    same region a *different* offset for each sample chunk -- misaligning
    samples of the same region depending purely on `max_mem`/cohort size, not on
    anything the user controls via `rng`.

    All `streaming_case` fixtures have 3 samples with a default `_window_samples
    == n_samples` (one sample chunk covers the whole cohort). Forcing
    `_window_samples = 1` splits that single chunk into three, one per sample,
    without changing anything else -- so under the bug, the *same* region+sample
    cell would land in a different sample chunk and draw a different offset than
    it does at the default chunking. This test fails before the fix (offsets
    depend on `_window_samples`) and passes after (offsets are keyed by absolute
    region index, independent of chunking).
    """
    regions, reference, variants, _ = streaming_case(backend)

    def mk():
        return (
            gvl.StreamingDataset(regions, reference=reference, variants=variants)
            .with_len(16)
            .with_settings(jitter=8, rng=0, deterministic=False)
        )

    sds_a = mk()
    sds_b = mk()
    # Force per-sample-chunk windows: every sample chunk now holds exactly one
    # sample, so a region spanning the cohort's 3 samples is visited via 3
    # separate `_plan()` jobs instead of 1.
    object.__setattr__(sds_b, "_window_samples", 1)

    def cells(sds):
        out = {}
        for data, r_idx, s_idx in sds.to_iter(batch_size=4):
            padded = np.asarray(data.to_padded(b"N"))
            for i in range(len(r_idx)):
                out[(int(r_idx[i]), int(s_idx[i]))] = padded[i]
        return out

    cells_a = cells(sds_a)
    cells_b = cells(sds_b)
    assert cells_a.keys() == cells_b.keys()
    for key in cells_a:
        np.testing.assert_array_equal(cells_a[key], cells_b[key])
