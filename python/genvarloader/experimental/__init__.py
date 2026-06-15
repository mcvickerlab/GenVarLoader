"""Experimental, unsupported features.

Symbols here are opt-in and intentionally **not** re-exported from the top-level
``genvarloader`` namespace. Import them explicitly, e.g.::

    from genvarloader.experimental import Table

They are not exercised in CI, may change or be removed without notice, and may
require optional dependencies installed via extras. ``Table`` needs the overlap
backend from the ``table`` extra: ``pip install genvarloader[table]``.
"""

from .._table import ExperimentalWarning, Table

__all__ = [
    "ExperimentalWarning",
    "Table",
]
