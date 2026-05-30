import types

from tests.benchmarks._capture import capture_first_call


def test_capture_records_first_call_args_across_namespaces():
    # Two modules that both bound the same function by name.
    defining = types.ModuleType("defining")
    consumer_a = types.ModuleType("consumer_a")
    consumer_b = types.ModuleType("consumer_b")

    def target(x, y, *, z=0):
        return x + y + z

    defining.target = target
    consumer_a.target = target  # `from defining import target`
    consumer_b.target = target

    calls = []

    def run():
        # Only the consumer namespaces are exercised at runtime.
        calls.append(consumer_a.target(1, 2, z=3))
        calls.append(consumer_b.target(10, 20, z=30))

    captured = capture_first_call(
        targets=[(consumer_a, "target"), (consumer_b, "target")],
        thunk=run,
    )

    # The real function still ran (capture is transparent).
    assert calls == [6, 60]
    # Only the first invocation was recorded.
    assert captured.args == (1, 2)
    assert captured.kwargs == {"z": 3}
    # Originals restored.
    assert consumer_a.target is target
    assert consumer_b.target is target


def test_capture_raises_if_never_called():
    mod = types.ModuleType("mod")
    mod.target = lambda: None

    import pytest

    with pytest.raises(RuntimeError, match="was never called"):
        capture_first_call(targets=[(mod, "target")], thunk=lambda: None)
