#!/usr/bin/env python
"""Classify Python docstrings as Google-, NumPy-, or section-less style.

This is a lightweight lint/inventory helper for the numpy -> google docstring
migration. It walks Python sources, extracts every module/class/function
docstring via :mod:`ast`, and classifies each one by the section syntax it
uses:

* ``google`` -- has section headers ending in a colon (``Args:``) and no
  NumPy-style dashed underlines.
* ``numpy`` -- has a section header immediately followed by a line of dashes
  (``Parameters`` / ``----------``).
* ``mixed`` -- shows both markers (usually a half-converted docstring).
* ``plain`` -- a docstring with no recognized sections (one-liners, prose).

Run without arguments to inventory ``python/genvarloader``. Pass ``--check`` to
exit non-zero when any ``numpy`` or ``mixed`` docstrings remain, so agents and
CI can gate on a fully-converted tree.

Examples:
    Inventory the package::

        python scripts/docstring_style.py

    Verify a set of files is fully Google-style::

        python scripts/docstring_style.py --check python/genvarloader/_fasta.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

# Section names recognized by NumPy and Google docstring styles (napoleon's
# superset). Used for both the NumPy underline probe and the Google header
# probe, so keep it broad.
_SECTION_NAMES = frozenset(
    {
        "Args",
        "Arguments",
        "Attributes",
        "Example",
        "Examples",
        "Keyword Args",
        "Keyword Arguments",
        "Methods",
        "Note",
        "Notes",
        "Other Parameters",
        "Parameters",
        "Raises",
        "References",
        "Return",
        "Returns",
        "See Also",
        "Todo",
        "Warning",
        "Warnings",
        "Warns",
        "Yield",
        "Yields",
    }
)

# A Google section header: one of the section names on its own line, ending in a
# colon (e.g. ``Args:``).
_GOOGLE_HEADER = re.compile(
    r"^(" + "|".join(re.escape(n) for n in _SECTION_NAMES) + r"):$"
)
# A NumPy underline: a run of dashes on its own line under a section header.
_NUMPY_UNDERLINE = re.compile(r"^-{3,}$")

Style = str  # one of: "google", "numpy", "mixed", "plain"


@dataclass
class Finding:
    """A single classified docstring.

    Attributes:
        path: Source file the docstring lives in.
        lineno: 1-based line of the owning module/class/function.
        qualname: Dotted name of the owner (``<module>`` for the module
            docstring).
        style: Classification, one of ``"google"``, ``"numpy"``, ``"mixed"``,
            or ``"plain"``.
    """

    path: Path
    lineno: int
    qualname: str
    style: Style


@dataclass
class FileReport:
    """Per-file tally of docstring styles.

    Attributes:
        path: The classified file.
        findings: Every docstring found in the file.
    """

    path: Path
    findings: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[Style, int]:
        """Return the number of docstrings of each style in this file.

        Returns:
            Mapping from style name to count, including zero-valued styles.
        """
        out: dict[Style, int] = {"google": 0, "numpy": 0, "mixed": 0, "plain": 0}
        for f in self.findings:
            out[f.style] += 1
        return out


def classify_docstring(doc: str) -> Style:
    """Classify a single docstring by its section syntax.

    Args:
        doc: The raw docstring text (as returned by :func:`ast.get_docstring`).

    Returns:
        ``"numpy"`` if NumPy dashed-underline sections are present, ``"google"``
        if Google colon-headers are present, ``"mixed"`` if both appear, and
        ``"plain"`` if no recognized section markers are found.
    """
    lines = dedent(doc).splitlines()
    stripped = [ln.strip() for ln in lines]

    has_numpy = False
    for i, line in enumerate(stripped[:-1]):
        if line in _SECTION_NAMES and _NUMPY_UNDERLINE.match(stripped[i + 1]):
            has_numpy = True
            break

    has_google = any(_GOOGLE_HEADER.match(line) for line in stripped)

    if has_numpy and has_google:
        return "mixed"
    if has_numpy:
        return "numpy"
    if has_google:
        return "google"
    return "plain"


def _iter_docstring_nodes(
    tree: ast.Module,
) -> "list[tuple[int, str, str]]":
    """Yield ``(lineno, qualname, docstring)`` for every docstring in a module.

    Args:
        tree: A parsed module AST.

    Returns:
        A list of ``(lineno, qualname, docstring)`` tuples for the module and
        every nested class and function that carries a docstring.
    """
    out: list[tuple[int, str, str]] = []

    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc is not None:
        out.append((1, "<module>", module_doc))

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                qualname = f"{prefix}{child.name}"
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    out.append((child.lineno, qualname, doc))
                walk(child, f"{qualname}.")

    walk(tree, "")
    return out


def classify_file(path: Path) -> FileReport:
    """Classify every docstring in a Python file.

    Args:
        path: Path to a ``.py`` source file.

    Returns:
        A :class:`FileReport` with one :class:`Finding` per docstring.

    Raises:
        SyntaxError: If the file cannot be parsed as Python.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    report = FileReport(path=path)
    for lineno, qualname, doc in _iter_docstring_nodes(tree):
        report.findings.append(Finding(path, lineno, qualname, classify_docstring(doc)))
    return report


def iter_python_files(paths: "list[Path]") -> "list[Path]":
    """Expand paths into a sorted list of ``.py``/``.pyi`` files.

    Directories are searched recursively; ``__pycache__`` and hidden
    directories are skipped.

    Args:
        paths: Files or directories to expand.

    Returns:
        Sorted list of ``.py`` and ``.pyi`` files.
    """
    out: set[Path] = set()
    for p in paths:
        if p.is_dir():
            for pattern in ("*.py", "*.pyi"):
                for f in p.rglob(pattern):
                    if "__pycache__" in f.parts:
                        continue
                    out.add(f)
        elif p.suffix in (".py", ".pyi"):
            out.add(p)
    return sorted(out)


def main(argv: "list[str] | None" = None) -> int:
    """Run the docstring-style inventory as a CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. ``0`` on success, or ``1`` under ``--check`` when any
        NumPy or mixed docstrings remain.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("python/genvarloader")],
        help="Files or directories to scan (default: python/genvarloader).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any numpy or mixed docstrings are found.",
    )
    parser.add_argument(
        "--show",
        choices=["numpy", "mixed", "google", "plain", "all", "non-google"],
        default="non-google",
        help="Which findings to list (default: non-google = numpy + mixed).",
    )
    args = parser.parse_args(argv)

    files = iter_python_files(args.paths)
    reports = [classify_file(f) for f in files]

    totals: dict[Style, int] = {"google": 0, "numpy": 0, "mixed": 0, "plain": 0}
    for r in reports:
        for style, n in r.counts().items():
            totals[style] += n

    show = {
        "numpy": {"numpy"},
        "mixed": {"mixed"},
        "google": {"google"},
        "plain": {"plain"},
        "all": {"numpy", "mixed", "google", "plain"},
        "non-google": {"numpy", "mixed"},
    }[args.show]

    for r in reports:
        listed = [f for f in r.findings if f.style in show]
        if not listed:
            continue
        print(f"\n{r.path}")
        for f in listed:
            print(f"  {f.lineno:>5}  {f.style:<7}  {f.qualname}")

    total = sum(totals.values())
    print(
        f"\nTotal docstrings: {total}  |  "
        f"google: {totals['google']}  "
        f"numpy: {totals['numpy']}  "
        f"mixed: {totals['mixed']}  "
        f"plain: {totals['plain']}"
    )

    if args.check and (totals["numpy"] or totals["mixed"]):
        print(
            f"\nFAIL: {totals['numpy'] + totals['mixed']} non-Google docstring(s) remain.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
