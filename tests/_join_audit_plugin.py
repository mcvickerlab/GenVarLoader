"""Pytest plugin that perturbs polars join row order to expose order-dependent bugs.

Activate with ``-p tests._join_audit_plugin`` (or set ``GVL_JOIN_AUDIT=1``).

Polars' ``DataFrame.join`` / ``LazyFrame.join`` do not guarantee row order unless
``maintain_order`` is passed explicitly (default is ``'none'``). Small inputs usually
*look* ordered, hiding positional-alignment bugs that only surface on large data or
across polars versions. This plugin reverses every join result whose ``maintain_order``
is unset/``'none'`` -- the strongest deterministic perturbation -- so any code that
relies on join output order breaks loudly under the test suite.
"""

from __future__ import annotations

import os

import polars as pl

_orig_df_join = pl.DataFrame.join
_orig_lf_join = pl.LazyFrame.join


def _unordered(kwargs: dict) -> bool:
    mo = kwargs.get("maintain_order", None)
    return mo is None or mo == "none"


def _patched_df_join(self, *args, **kwargs):
    res = _orig_df_join(self, *args, **kwargs)
    if _unordered(kwargs) and res.height > 1:
        res = res.reverse()
    return res


def _patched_lf_join(self, *args, **kwargs):
    res = _orig_lf_join(self, *args, **kwargs)
    if _unordered(kwargs):
        res = res.reverse()
    return res


def _install():
    pl.DataFrame.join = _patched_df_join
    pl.LazyFrame.join = _patched_lf_join


def pytest_configure(config):
    _install()


# Allow activation via env var as well (e.g. for cargo/other harnesses).
if os.environ.get("GVL_JOIN_AUDIT") == "1":
    _install()
