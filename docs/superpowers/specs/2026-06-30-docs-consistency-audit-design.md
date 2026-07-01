# Docs consistency pass + CLAUDE.md docs-audit gate

**Date:** 2026-06-30
**Branch:** `docs/consistency-audit`

## Problem

Recent gvl work (Phase 5: numba read-path backend deleted → Rust-only; awkward →
`_core.Ragged` migration; new rayon threading knobs) left user-facing docs stale,
and there is no process gate ensuring docs stay consistent with future
feature/breaking-change PRs.

Key facts established during the audit:
- gvl's **own** code is numba-free (`pixi.toml` comment; `tests/parity/test_import_no_numba.py`).
  Numba survives only as a conda pin because **seqpro** transitively imports it. The
  residual `_numba`-suffixed names in gvl route only to Rust or numpy.
- The read path is parallelized in Rust with rayon, tuned via env vars in
  `python/genvarloader/_threads.py`: `GVL_NUM_THREADS`, `GVL_FORCE_PARALLEL`, and a
  `RAYON_NUM_THREADS` override (issue #263). None were documented user-side.

## Scope (focused fix — not a full line-by-line sweep)

### Part A — docs fixes
1. `docs/source/faq.md` — rewrite the "Ragged objects" answer's stale
   "subclass of Awkward Arrays / numba JIT'd functions" paragraph to reflect the
   `seqpro.rag.Ragged` (`_core.Ragged`, Rust) backend; note awkward is no longer a dep.
2. `docs/source/faq.md` — new entry "How do I control how many threads GVL uses?"
   documenting the three env vars, sourced from `_threads.py`.
3. `README.md` — replace the `tbb`/`pyomp`-for-numba install note with a note that
   parallelism is built-in (Rust/rayon), tunable via `GVL_NUM_THREADS`.
4. `skills/genvarloader/SKILL.md` — `_core.Ragged` "Rust+numba backend" → "Rust backend"
   (seqpro-core's rag layer is numba-free).
5. Targeted leftover sweep of README + `docs/source/*.md` + SKILL.md for other
   `numba`/`awkward`/`GVL_BACKEND`/`tbb`/`pyomp` references — none remaining (the
   surviving `awkward` mentions in SKILL.md describe "zero-awkward" as a feature).

The auto-generated `docs/source/changelog.md` is left untouched (built from commit
messages via `changelog.md.j2`).

### Part B — CLAUDE.md gate
Add a "Docs audit before feature/breaking-change PRs" section that requires auditing
README + `docs/source/*.md` + SKILL.md before such PRs, lists what to check
(now-false claims, new config/env vars, changed preprocessing), and states the
auto-generated changelog does not count as documentation. Complements the existing
skill-maintenance rule.

## Verification
- Markdown edits are prose-only in existing files with no new MyST directives.
- Full `pixi run -e docs doc` build not run in-worktree (docs env not provisioned there);
  low build-break risk given no directive changes.
