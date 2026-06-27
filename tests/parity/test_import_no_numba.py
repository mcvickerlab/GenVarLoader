"""genvarloader's OWN modules must not import numba (Phase 5 W5).

NOTE: `import genvarloader` may still pull numba transitively via seqpro
(seqpro 0.20.0 eagerly imports numba). That is outside genvarloader's control;
this guard asserts genvarloader's own source is numba-free. See the seqpro
follow-up issue for the transitive import and the W6 RSS impact.
"""
from __future__ import annotations

import pathlib

import genvarloader


def test_genvarloader_own_code_imports_no_numba():
    pkg_dir = pathlib.Path(genvarloader.__file__).parent
    offenders: list[str] = []
    for py in pkg_dir.rglob("*.py"):
        for ln, line in enumerate(py.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("import numba") or s.startswith("from numba"):
                offenders.append(f"{py.relative_to(pkg_dir)}:{ln}: {s}")
    assert not offenders, "genvarloader modules import numba:\n" + "\n".join(offenders)
