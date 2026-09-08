"""``Dataset.with_settings(parallel=...)`` reaches the kernels (issue #352).

The value has to arrive at ``should_parallelize`` call sites that live in free
functions with no ``Dataset`` in scope (``_genotypes``, ``_intervals``,
``_reference``), which is why it travels as a context-scoped policy rather than
a threaded-through argument. These tests pin that it actually gets there, that
it beats the environment, and that it cannot leak past the read.
"""

import genvarloader as gvl
import pytest

from genvarloader import _threads as th


@pytest.fixture(scope="module")
def ds(phased_svar_gvl, reference):
    return gvl.Dataset.open(phased_svar_gvl, reference=reference).with_seqs(
        "haplotypes"
    )


def _observed_policy(dataset, monkeypatch) -> list[bool]:
    """Record what the gate answers during one read of ``dataset``."""
    seen: list[bool] = []
    real = th.should_parallelize

    def spy(total_bytes: int) -> bool:
        answer = real(total_bytes)
        seen.append(answer)
        return answer

    # Patch at the definition site AND in every module that imported the name
    # by value -- `from .._threads import should_parallelize` binds a reference,
    # so patching only `_threads` would not be observed by any caller.
    monkeypatch.setattr(th, "should_parallelize", spy)
    for mod in th_importers():
        monkeypatch.setattr(mod, "should_parallelize", spy, raising=False)
    dataset[:2, :]
    return seen


def th_importers():
    from genvarloader._dataset import (
        _genotypes,
        _haps,
        _intervals,
        _reconstruct,
        _reference,
        _svar2_haps,
    )

    return (_genotypes, _haps, _intervals, _reconstruct, _reference, _svar2_haps)


def test_parallel_false_forces_every_kernel_serial(ds, monkeypatch):
    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")  # would force parallel on "auto"
    seen = _observed_policy(ds.with_settings(parallel=False), monkeypatch)
    assert seen, "no kernel consulted the gate; the test proves nothing"
    assert not any(seen), "an explicit parallel=False did not reach every kernel"


def test_parallel_true_forces_every_kernel_parallel(ds, monkeypatch):
    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    seen = _observed_policy(ds.with_settings(parallel=True), monkeypatch)
    assert seen
    assert all(seen), "an explicit parallel=True did not reach every kernel"


def test_auto_still_defers_to_the_environment(ds, monkeypatch):
    """The default is unchanged behaviour -- env still wins on "auto"."""
    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")
    seen = _observed_policy(ds, monkeypatch)
    assert seen
    assert all(seen), "auto stopped deferring to GVL_FORCE_PARALLEL"


def test_policy_does_not_leak_past_the_read(ds, monkeypatch):
    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    ds.with_settings(parallel=False)[:2, :]
    assert th._PARALLEL.get() == "auto"
    assert th.should_parallelize(1 << 30) is True


def test_setting_is_per_dataset_not_global(ds, monkeypatch):
    """Configuring one dataset must not reconfigure another."""
    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")
    serial = ds.with_settings(parallel=False)
    assert serial.parallel is False
    assert ds.parallel == "auto", "with_settings mutated the original dataset"
    # Reading the serial one must leave the untouched one still on "auto",
    # where GVL_FORCE_PARALLEL still applies.
    _observed_policy(serial, monkeypatch)
    assert all(_observed_policy(ds, monkeypatch)), "policy leaked between datasets"


def test_output_is_identical_regardless_of_policy(ds):
    import numpy as np

    a = ds.with_settings(parallel=False)[:2, :]
    b = ds.with_settings(parallel=True)[:2, :]
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(a.offsets, b.offsets)


def test_invalid_policy_rejected_at_with_settings(ds):
    with pytest.raises(ValueError, match="parallel"):
        ds.with_settings(parallel="sometimes")
