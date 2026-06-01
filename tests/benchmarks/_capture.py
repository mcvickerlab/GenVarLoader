"""Capture-and-replay: record the first call's arguments of a hot function so a
micro-benchmark can replay them with realistic inputs.

The hot numba functions are imported *by name* into consumer modules
(``from ._genotypes import reconstruct_haplotypes_from_sparse``), so we must
patch each consumer namespace, not the defining module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CapturedCall:
    """The first observed call's positional and keyword arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def capture_first_call(
    targets: list[tuple[Any, str]],
    thunk: Callable[[], Any],
) -> CapturedCall:
    """Run ``thunk``; record the first call to the patched function; restore.

    Parameters
    ----------
    targets
        ``(module_or_namespace, attribute_name)`` pairs that all hold a
        reference to the *same* function. Every pair is patched so the call is
        recorded no matter which namespace invokes it.
    thunk
        Zero-arg callable that triggers at least one call to the target. Only
        the first call's arguments are kept.

    Returns
    -------
    CapturedCall

    Raises
    ------
    RuntimeError
        If the target was never called while running ``thunk``.
    """
    captured: list[CapturedCall] = []
    original = getattr(targets[0][0], targets[0][1])

    def recorder(*args: Any, **kwargs: Any) -> Any:
        if not captured:
            captured.append(CapturedCall(args=args, kwargs=dict(kwargs)))
        return original(*args, **kwargs)

    for namespace, attr in targets:
        setattr(namespace, attr, recorder)
    try:
        thunk()
    finally:
        for namespace, attr in targets:
            setattr(namespace, attr, original)

    if not captured:
        name = getattr(original, "__name__", str(original))
        raise RuntimeError(f"{name} was never called while running the thunk")
    return captured[0]
