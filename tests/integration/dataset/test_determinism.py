"""Determinism tests for ``gvl.Dataset``.

Covers:
- Two opens with the same ``rng`` seed and ``jitter`` produce identical reads.
- With ``jitter=0`` and the default deterministic settings, repeated reads of
  the same ``(region, sample)`` pair within one dataset are identical.
- A seeded ``torch.Generator`` makes ``to_dataloader(shuffle=True)`` produce
  the same first batch across two independent DataLoader instances.
"""

from pathlib import Path

import genvarloader as gvl
import numpy as np
import pytest


SEED = 1234


def _materialize(item):
    """Flatten a (possibly nested) Ragged / array / tuple item into a list of
    plain numpy arrays suitable for byte-equality comparison.

    Ragged arrays are flattened to (data, lengths) so two reads are equal iff
    they produce identical values *and* identical per-row lengths.
    """
    if isinstance(item, tuple):
        out = []
        for x in item:
            out.extend(_materialize(x))
        return out
    # seqpro Ragged or our RaggedAnnotatedHaps-style container
    if hasattr(item, "data") and hasattr(item, "lengths"):
        return [np.asarray(item.data), np.asarray(item.lengths)]
    # RaggedAnnotatedHaps: has haps / var_idxs / ref_coords (each Ragged)
    if hasattr(item, "haps") and hasattr(item, "var_idxs"):
        return (
            _materialize(item.haps)
            + _materialize(item.var_idxs)
            + _materialize(item.ref_coords)
        )
    return [np.asarray(item)]


def _read_pairs(ds: gvl.RaggedDataset, pairs):
    """Read a few (region, sample) pairs as padded numpy arrays/tuples."""
    return [_materialize(ds[r, s]) for r, s in pairs]


def _assert_equal(a, b):
    """Both args are lists of numpy arrays (output of _materialize)."""
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_same_seed_same_output(phased_vcf_gvl: Path, ref_fasta: Path):
    """Two independent opens with the same seed + jitter produce identical reads."""
    pairs = [(0, 0), (1, 1), (3, 0), (5, 2)]

    ds_a = gvl.Dataset.open(
        phased_vcf_gvl, ref_fasta, jitter=2, rng=SEED, deterministic=False
    )
    ds_b = gvl.Dataset.open(
        phased_vcf_gvl, ref_fasta, jitter=2, rng=SEED, deterministic=False
    )

    out_a = _read_pairs(ds_a, pairs)
    out_b = _read_pairs(ds_b, pairs)

    for a, b in zip(out_a, out_b):
        _assert_equal(a, b)


def test_jitter_zero_is_deterministic(phased_vcf_gvl: Path, ref_fasta: Path):
    """With jitter=0 (and default deterministic=True), repeated reads of the same
    (region, sample) within a single dataset are byte-identical.

    Note: the toy dataset is built with max_jitter=2, so jitter=0 is a legal but
    non-trivial setting (it explicitly disables the stochastic jitter shift).
    """
    ds = gvl.Dataset.open(phased_vcf_gvl, ref_fasta).with_settings(jitter=0)
    pairs = [(0, 0), (2, 1), (4, 2)]

    first = _read_pairs(ds, pairs)
    second = _read_pairs(ds, pairs)
    third = _read_pairs(ds, pairs)

    for a, b, c in zip(first, second, third):
        _assert_equal(a, b)
        _assert_equal(a, c)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Dataset.with_settings(rng=...) raises TypeError: "
        "python/genvarloader/_dataset/_impl.py:271 writes to_evolve['rng'] but "
        "the dataclass field is named '_rng', so attrs.evolve rejects the kwarg. "
        "Workaround in other tests: pass rng= to Dataset.open(). Flip this xfail "
        "to a regular assertion once line 271 uses to_evolve['_rng']."
    ),
)
def test_with_settings_rng_kwarg_broken(phased_vcf_gvl: Path, ref_fasta: Path):
    """Capture the with_settings(rng=...) bug so it surfaces when fixed."""
    ds = gvl.Dataset.open(phased_vcf_gvl, ref_fasta)
    ds2 = ds.with_settings(rng=SEED)
    assert isinstance(ds2, gvl.Dataset)


# -- torch-only test ----------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "get_sampler() creates RandomSampler without forwarding the DataLoader "
        "generator, so the sampler uses the global torch RNG and shuffled batch "
        "order is not reproducible from a seeded torch.Generator. See _torch.py "
        "get_sampler / get_dataloader. Flip this xfail to a regular assertion "
        "once the sampler accepts/forwards the generator."
    ),
    strict=True,
)
def test_dataloader_seeded_batch_order_reproducible(
    phased_vcf_gvl: Path, ref_fasta: Path
):
    """A seeded ``torch.Generator`` should make shuffled DataLoader batches
    reproducible across two independent DataLoader instances over the same
    dataset. Currently fails because of the generator-forwarding bug above."""
    torch = pytest.importorskip("torch")

    ds = gvl.Dataset.open(phased_vcf_gvl, ref_fasta).with_settings(jitter=0)

    gen_a = torch.Generator()
    gen_a.manual_seed(SEED)
    dl_a = ds.to_dataloader(
        batch_size=2, shuffle=True, num_workers=0, generator=gen_a, return_indices=True
    )

    gen_b = torch.Generator()
    gen_b.manual_seed(SEED)
    dl_b = ds.to_dataloader(
        batch_size=2, shuffle=True, num_workers=0, generator=gen_b, return_indices=True
    )

    batch_a = next(iter(dl_a))
    batch_b = next(iter(dl_b))

    # return_indices appends row + sample index arrays as the last two elements
    # of the batch tuple. Compare those to assert the sampler ordering matches.
    row_a, samp_a = batch_a[-2], batch_a[-1]
    row_b, samp_b = batch_b[-2], batch_b[-1]
    np.testing.assert_array_equal(np.asarray(row_a), np.asarray(row_b))
    np.testing.assert_array_equal(np.asarray(samp_a), np.asarray(samp_b))
