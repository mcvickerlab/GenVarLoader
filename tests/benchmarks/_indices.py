"""Shared (region, sample) index generation for the benchmark suite.

A single source of truth used by the fixtures (conftest), the end-to-end
benchmarks, and the profiling driver, so they all drive the reconstructor with
the same batch shape.
"""

from __future__ import annotations


def batch_indices(
    n_regions: int, n_samples: int, n: int
) -> tuple[list[int], list[int]]:
    """A flat batch of ``n`` (region_idx, sample_idx) pairs within bounds.

    Regions cycle ``0..n_regions-1`` and samples rotate ``0..n_samples-1`` in
    lockstep, so a batch exercises a spread of both even when ``n`` is small.
    Clamped to ``n_regions * n_samples`` so indices never exceed the dataset.
    """
    n = min(n, n_regions * n_samples)
    regions = [i % n_regions for i in range(n)]
    samples = [i % n_samples for i in range(n)]
    return regions, samples
