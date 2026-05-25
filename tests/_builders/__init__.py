"""In-memory builders for tests.

Builders take plain Python/numpy/pyarrow inputs and return real internal
GenVarLoader types. Mocks are reserved for the ``Reader`` protocol boundary.
See docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md.

This package is intentionally importable on its own (no test fixtures here)
so it can be reused from ``tests/conftest.py`` and from unit tests directly.
"""
