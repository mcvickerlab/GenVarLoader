"""``StreamingDataset.with_settings(parallel=...)`` reaches the kernels (issue #359).

The streaming twin of ``tests/unit/dataset/test_parallel_setting.py``. Same
mechanism, same precedence contract: the policy travels as a context-scoped
``ContextVar`` rather than a threaded-through argument, because the consumers
live in free functions and backend methods with no dataset in scope.

Two things are genuinely different here, and together they are why this is a
separate module rather than another parametrization of the written-path suite.

**The policy has two distinct consumers on the streaming side.** The SVAR2
super-batch and the track-realign kernels call ``should_parallelize`` per batch
with that batch's own byte count, exactly like the written path. The
SVAR1/VCF/PGEN record engines do not: they take a single ``parallel: bool`` at
CONSTRUCTION that governs every batch they will ever produce, and it was
hardcoded ``True``. So the written-path trick of spying on ``should_parallelize``
proves nothing about the record drive -- that path never calls it. These tests
assert on the argument the Rust constructor actually received instead, which is
the thing that decides whether rayon is used.

**``to_iter`` is a generator, and generators do not get their own context.** A
``with parallel_policy(...)`` spanning the ``yield`` would leave the policy set
in the CALLER's context while the consumer's loop body runs, silently
re-policying any unrelated ``"auto"`` read made between batches. So ``to_iter``
establishes the policy per ADVANCE of the inner generator and releases it before
yielding -- ``test_policy_is_not_set_inside_the_loop_body`` is the gate on that,
and it is the one assertion the written-path suite has no analog for.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
from genvarloader import _threads as th
from genvarloader._dataset import _streaming

BACKENDS = ["svar1", "vcf", "pgen"]

# The Rust engine classes the record backends construct. `build_engine` resolves
# each by attribute lookup on the extension module at call time
# (`from ..genvarloader import ...` inside the method), so replacing the
# attribute is enough to observe the constructor arguments.
_ENGINE_NAMES = ("Svar1StreamEngine", "RecordStreamEngine", "Svar2StreamEngine")


def _sds(streaming_case, backend: str):
    regions, reference, variants, _ = streaming_case(backend)
    return gvl.StreamingDataset(regions, reference=reference, variants=variants)


def _drive_capturing_engine_args(sds, monkeypatch, batch_size: int = 2) -> list[tuple]:
    """Iterate ``sds`` to exhaustion, returning each engine's positional args."""
    from genvarloader import genvarloader as _rust

    seen: list[tuple] = []
    with monkeypatch.context() as m:
        for name in _ENGINE_NAMES:
            real = getattr(_rust, name, None)
            if real is None:
                continue

            def spy(*args, _real=real, **kwargs):
                seen.append(args)
                return _real(*args, **kwargs)

            m.setattr(_rust, name, spy)
        for _ in sds.to_iter(batch_size=batch_size):
            pass
    return seen


def _sole_differing_bool(a: tuple, b: tuple) -> tuple[bool, bool]:
    """The one boolean argument that differs between two otherwise-equal calls.

    Index-free on purpose: asserting "``parallel`` is positional argument 22"
    would fail for an unrelated signature change and tell the reader nothing.
    Every other argument is identical across the two runs (same dataset, same
    batch size), so a single differing bool IS the policy flag -- and requiring
    it to be the ONLY one also catches the flag being wired to a second,
    unintended place.
    """
    assert len(a) == len(b), "engine arity changed between runs"
    diffs = [
        (i, x, y)
        for i, (x, y) in enumerate(zip(a, b))
        if isinstance(x, bool) and isinstance(y, bool) and x is not y
    ]
    assert len(diffs) == 1, f"expected exactly one differing bool arg, got {diffs}"
    return diffs[0][1], diffs[0][2]


# --- the resolver itself ----------------------------------------------------


def test_engine_flag_follows_an_explicit_policy():
    """``_engine_parallel`` honors an explicit policy and defaults to parallel."""
    assert _streaming._engine_parallel() is True  # "auto", the default
    with th.parallel_policy(False):
        assert _streaming._engine_parallel() is False
    with th.parallel_policy(True):
        assert _streaming._engine_parallel() is True


def test_engine_flag_ignores_the_size_gate_under_auto(monkeypatch):
    """``"auto"`` stays parallel regardless of the environment.

    The engine flag is chosen once, before any batch exists, so there is no byte
    count to gate on -- see ``_engine_parallel``'s docstring. Pinning this keeps
    a future refactor from quietly routing it through ``should_parallelize``,
    which would flip small-batch streams to serial as a side effect.
    """
    monkeypatch.setenv("GVL_FORCE_PARALLEL", "0")
    assert _streaming._engine_parallel() is True


# --- the flag reaches the Rust constructor ----------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_policy_reaches_the_engine_constructor(streaming_case, backend, monkeypatch):
    """An explicit ``parallel=`` changes the flag the engine is built with.

    Before #359 every record backend passed a literal ``True`` here, so the
    per-dataset setting could not affect the reconstruction kernels at all.
    """
    sds = _sds(streaming_case, backend)
    serial = _drive_capturing_engine_args(
        sds.with_settings(parallel=False), monkeypatch
    )
    par = _drive_capturing_engine_args(sds.with_settings(parallel=True), monkeypatch)
    assert serial and par, "no engine was constructed; the test proves nothing"
    assert len(serial) == len(par)
    for a, b in zip(serial, par):
        assert _sole_differing_bool(a, b) == (False, True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_auto_builds_a_parallel_engine(streaming_case, backend, monkeypatch):
    """The default is unchanged behavior: ``"auto"`` builds what the literal did."""
    sds = _sds(streaming_case, backend)
    auto = _drive_capturing_engine_args(sds, monkeypatch)
    par = _drive_capturing_engine_args(sds.with_settings(parallel=True), monkeypatch)
    assert auto and len(auto) == len(par)
    for a, b in zip(auto, par):
        assert all(
            not (isinstance(x, bool) and isinstance(y, bool) and x is not y)
            for x, y in zip(a, b)
        ), "auto built a different engine flag than an explicit parallel=True"


# --- precedence -------------------------------------------------------------


def test_per_dataset_beats_an_ambient_policy(streaming_case, monkeypatch):
    """An ambient block does not override the per-dataset setting.

    ``parallel_policy`` and ``with_settings(parallel=)`` write the SAME
    ``ContextVar``, so which wins is entirely a question of who sets it last --
    and ``to_iter`` sets it per advance, INSIDE any ambient block the caller
    opened. This pins that ordering; reversing it would make a dataset's own
    declared policy silently ignorable by its caller.
    """
    sds = _sds(streaming_case, "svar1").with_settings(parallel=False)
    with th.parallel_policy(True):
        serial = _drive_capturing_engine_args(sds, monkeypatch)
    par = _drive_capturing_engine_args(
        _sds(streaming_case, "svar1").with_settings(parallel=True), monkeypatch
    )
    assert serial and len(serial) == len(par)
    for a, b in zip(serial, par):
        assert _sole_differing_bool(a, b) == (False, True), (
            "an ambient parallel_policy(True) overrode parallel=False"
        )


def test_ambient_policy_still_applies_to_an_auto_dataset(streaming_case, monkeypatch):
    """The converse: an ``"auto"`` dataset takes the ambient policy.

    That is the pre-#359 behavior for every kernel that reads the ``ContextVar``
    and must not regress now that ``"auto"`` has a second consumer.
    """
    sds = _sds(streaming_case, "svar1")
    with th.parallel_policy(False):
        serial = _drive_capturing_engine_args(sds, monkeypatch)
    par = _drive_capturing_engine_args(sds.with_settings(parallel=True), monkeypatch)
    assert serial and len(serial) == len(par)
    for a, b in zip(serial, par):
        assert _sole_differing_bool(a, b) == (False, True), (
            "an auto StreamingDataset stopped honoring parallel_policy"
        )


def test_setting_is_per_dataset_not_global(streaming_case):
    """Configuring one dataset must not reconfigure another."""
    sds = _sds(streaming_case, "svar1")
    serial = sds.with_settings(parallel=False)
    assert serial.parallel is False
    assert sds.parallel == "auto", "with_settings mutated the original dataset"


# --- ContextVar hygiene around the generator --------------------------------


def test_policy_is_not_set_inside_the_loop_body(streaming_case):
    """The policy is released before each ``yield``.

    ``to_iter`` is a generator, and a generator shares its caller's context.
    Holding the policy across the ``yield`` would re-policy whatever the
    consumer does between batches -- including reads of a completely unrelated
    ``"auto"`` dataset. Asserting the ContextVar is back to its outer value in
    the loop body is the direct test of that; asserting only after the loop
    would pass even for the leaky implementation.
    """
    sds = _sds(streaming_case, "svar1").with_settings(parallel=False)
    batches = 0
    for _ in sds.to_iter(batch_size=2):
        batches += 1
        assert th._PARALLEL.get() == "auto", (
            "parallel policy leaked into the consumer's loop body"
        )
    assert batches > 0, "vacuous pass: the stream yielded nothing"


def test_policy_does_not_leak_past_iteration(streaming_case, monkeypatch):
    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    sds = _sds(streaming_case, "svar1").with_settings(parallel=False)
    for _ in sds.to_iter(batch_size=2):
        pass
    assert th._PARALLEL.get() == "auto"
    assert th.should_parallelize(1 << 30) is True


def test_policy_released_when_iteration_is_abandoned(streaming_case):
    """Abandoning the iterator mid-stream must not strand the policy.

    ``parallel_policy`` resets in a ``finally``, and the scope is exited before
    every yield, so the ContextVar is already restored at the moment the
    consumer walks away -- there is no window in which an abandoned generator
    holds it. (Related: issue #399, abandoning a record-stream iterator.)
    """
    sds = _sds(streaming_case, "svar1").with_settings(parallel=False)
    it = sds.to_iter(batch_size=2)
    next(iter(it))
    del it
    assert th._PARALLEL.get() == "auto"


# --- the policy is an optimization, never a result change -------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_is_identical_regardless_of_policy(streaming_case, backend):
    sds = _sds(streaming_case, backend)
    serial = [d for d, _, _ in sds.with_settings(parallel=False).to_iter(batch_size=2)]
    par = [d for d, _, _ in sds.with_settings(parallel=True).to_iter(batch_size=2)]
    assert len(serial) == len(par) > 0
    for a, b in zip(serial, par):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
        np.testing.assert_array_equal(np.asarray(a.offsets), np.asarray(b.offsets))


def test_invalid_policy_rejected_at_with_settings(streaming_case):
    """Rejected where it was written, not deep inside an iteration that may not
    start until much later."""
    sds = _sds(streaming_case, "svar1")
    with pytest.raises(ValueError, match="parallel"):
        sds.with_settings(parallel="sometimes")
