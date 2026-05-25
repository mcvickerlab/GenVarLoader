# Test Suite Overhaul — Status & Resume Notes

**As of:** 2026-05-24 (committed through `b44a8b0`)
**Branch:** `worktree-test-suite-overhaul`
**Worktree:** `/Users/david/projects/GenVarLoader/.claude/worktrees/test-suite-overhaul`
**PR:** https://github.com/mcvickerlab/GenVarLoader/pull/194 (open)

Living status snapshot. Read this first when resuming with fresh context — it points to the authoritative spec/audit/plan files but captures the deltas that aren't visible in those frozen documents.

---

## Authoritative reference files

- **Design spec** (frozen intent): `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- **Audit** (frozen classification): `docs/superpowers/specs/2026-05-24-test-audit.md`
- **Coverage baseline** (frozen, Phase 3): `docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt`
- **Plans** (one per component executed):
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phases-1-3.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase4.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-prelude-and-ragged.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-reconstruct.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-variants.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-splice.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-svar-link.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-utility.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-tracks-broader.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-ref-fasta.md`
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-dataset-polymorphism.md`

---

## Current state

### Test counts

- **Non-slow tier:** 351 passed, 3 skipped, 3 deselected, 2 xfailed
- **Slow tier (1kg, where data exists):** 3 passed
- **Unit tier alone:** 167 passed, 2 xfailed (~12s combined, ~3.2s unit alone)
- **Coverage:** 63% line+branch (parity with Phase 3 baseline; no production code modified)

### File layout

```
tests/
├── conftest.py                     # session-scoped path fixtures (Phase 2)
├── _builders/
│   ├── __init__.py
│   ├── ragged.py                   # make_ragged_seqs, make_ragged_intervals
│   └── reconstruct.py              # make_tracks
├── unit/                           # ← 167 tests
│   ├── test_fasta.py
│   ├── test_table.py
│   ├── test_utils.py
│   ├── dataset/
│   │   ├── genotypes/
│   │   │   ├── test_choose_exonic_variants.py
│   │   │   └── test_reconstruct.py
│   │   ├── test_build_reconstructor.py
│   │   ├── test_indexing.py
│   │   ├── test_ref_ds.py
│   │   ├── test_svar_link_models.py
│   │   ├── test_with_insertion_fill.py
│   │   └── test_write_tracks.py
│   ├── ragged/
│   │   ├── test_rag_variants.py
│   │   └── test_ragged_rc_packing.py
│   ├── splice/
│   │   ├── test_get_splice_bed.py
│   │   ├── test_ref_ds_splice_settings.py
│   │   └── test_splice_plan.py
│   ├── tracks/
│   │   ├── test_i2t_t2i.py
│   │   ├── test_insertion_fill.py
│   │   ├── test_random_nonoverlapping.py
│   │   ├── test_realign.py
│   │   ├── test_tracks_splice.py
│   │   └── utils.py                # sibling helper for test_random_nonoverlapping
│   └── variants/
│       ├── test_variant_utils.py
│       └── test_variants_info_fields.py
├── integration/                    # ← 184 tests
│   ├── test_ref_ds_splicing.py
│   ├── dataset/
│   │   ├── test_dataset.py
│   │   ├── test_ds_haps.py
│   │   ├── test_ds_haps_1kg.py
│   │   ├── test_dummy_dataset_insertion_fill.py   # renamed in Phase 5 reconstruct
│   │   ├── test_issue_153.py
│   │   ├── test_issue_191_var_fields.py
│   │   ├── test_jitter.py
│   │   ├── test_open_vs_settings_parity.py
│   │   ├── test_rc_packing.py
│   │   ├── test_subset.py
│   │   ├── test_svar_link.py
│   │   ├── test_with_settings_var_filter.py
│   │   ├── test_write.py
│   │   └── test_write_tracks_e2e.py               # renamed in Phase 5 tracks (basename collision)
│   ├── tracks/
│   │   └── test_annot_tracks.py
│   └── variants/
│       └── test_sites.py
├── data/
└── test_bigwig.rs                  # Rust test, untouched throughout overhaul
```

---

## What shipped (50 commits)

| Phase | Component | Result |
|---|---|---|
| 1–3 | Bootstrap, conftest, audit | `pytest-cov` config, `test-cov` pixi task, unit/integration/_builders scaffolding, relocated 33 files under integration/, centralized path fixtures, generated coverage baseline, classified 137 test functions in audit doc |
| 4 | Delete pass | 5 deletions: `test_filter_af.py` (whole-file), `test_rs_indexing` (147 cases), `test_interval_track.py` (whole-file), `test_refdataset_unspliced_defaults`, `test_write` (skipped) |
| 5 prelude + ragged | First builders + 11 moves | `make_ragged_seqs`, `make_ragged_intervals`; 9 file moves/extractions including 4 whole-file (test_build_reconstructor, test_indexing, test_realign, test_reconstruct, test_variant_utils, test_i2t_t2i, test_rag_variants) and 2 splice splits (test_splice_plan + test_tracks_splice extraction) and 1 ragged extraction (test_rc_returns_packed_buffer) |
| 5 reconstruct | `make_tracks` builder + insertion-fill split | 19 unit tests extracted to `unit/tracks/test_insertion_fill.py`; 3 dataset-dependent kept in `integration/dataset/test_dummy_dataset_insertion_fill.py` (renamed to avoid basename collision) |
| 5 variants | _Variants info-field unit tests | 5 extracted to `unit/variants/test_variants_info_fields.py`; 10 dataset-dependent kept. test_choose_exonic_variants moved whole-file |
| 5 splice | get_splice_bed move + ref_ds_splicing split | 11 moved to `unit/splice/test_get_splice_bed.py`; 5 RefDataset settings/validation tests extracted to `unit/splice/test_ref_ds_splice_settings.py`; 4 byte-comparison tests stay in integration |
| 5 svar_link | 7 pydantic-model tests extracted | `unit/dataset/test_svar_link_models.py`; 14 dataset-dependent kept |
| 5 utility | test_utils.py whole-file move | 9 collected tests (5 audit-Port entries; pytest_cases expands `test_normalize_contig_name` to 5 cases) moved to `unit/test_utils.py`; no source changes; no builder needed |
| 5 tracks (broader) | test_random_nonoverlapping + utils.py whole-move; test_write_tracks atomic split (integration renamed to `_e2e`); test_table.py whole-file move | 3 files relocated, 1 atomic split, 1 integration rename for basename-collision avoidance |
| 5 ref/fasta | reference fixture → conftest; test_fasta.py + test_ref_ds.py whole-file moves | Session-scoped `reference` fixture promoted with docstring carve-out (metadata-only Reference is cheap); 2 duplicate local fixtures dropped; 2 whole-file moves; no basename collisions |
| 5 dataset polymorphism (minimal) | atomic split of test_dummy_dataset_insertion_fill.py | 1 test (test_with_insertion_fill_rejects_when_no_tracks_active) extracted to tests/unit/dataset/test_with_insertion_fill.py; test_ds_indexing deferred (would require a `make_dataset` builder wrapping gvl.write — speculative scaffolding per YAGNI) |

---

## What's left (by remaining component)

Numbers are best-effort estimates from the audit; verify against the current integration tree before planning.

### Components with NO port-bucket tests remaining

- **Haps** — Audit identified no Haps-specific Port tests outside the 5 already moved in the variants plan. A "haps component" plan would be builder-only scaffolding (`make_variants_table`, `make_variants`, `make_haps`). Per YAGNI, deferred until a real consumer surfaces. Reconsider if/when later plans need to construct synthetic `_Variants`/`Haps` without a real SVAR.
- **Dataset polymorphism (closed, minimal scope)** — `test_with_insertion_fill_rejects_when_no_tracks_active` moved to unit. `test_dataset.py:test_ds_indexing` deferred: porting requires a `make_dataset` builder that wraps `gvl.write()` with synthetic BED + variants + tracks + reference inputs. That builder has no other consumer; per YAGNI, build it when a real need surfaces.

### Deferred individual tests

- **`test_resolve_svar_prefers_override`, `test_resolve_svar_falls_back_to_sibling`** — Audit-classified Port but their current bodies use the `svar_dataset_paths` fixture. To move, they'd need to be rewritten to synthesize a fake SVAR directory with matching `variant_idxs.npy` fingerprint. Skip until there's a reason to invest.

---

## Notable decisions / gotchas

1. **Basename collisions** — pytest's default `--import-mode=prepend` treats same-basename test files in different dirs as the same module. Discovered when `tests/unit/tracks/test_insertion_fill.py` collided with `tests/integration/dataset/test_insertion_fill.py`. Briefly tried `--import-mode=importlib`, but it broke `tests/integration/tracks/test_random_nonoverlapping.py`'s `from utils import ...` (which relies on prepend-mode sys.path injection). **Settled fix:** rename the integration file in each collision. The reconstruct plan's commit `d33a521` did this for insertion_fill. Watch for this when porting `test_table.py`, `test_random_nonoverlapping.py`, or any other file that may collide with a future unit-tier counterpart.

2. **Builder imports use sys.path injection** — Unit tests that consume `_builders/` use:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
   from _builders.<module> import ...   # noqa: E402
   ```
   Required because `tests/` has no `__init__.py`. See `tests/unit/tracks/test_insertion_fill.py` for the canonical example. **Don't promote `tests/` to a package** — that breaks pytest collection in the default import mode.

3. **Pre-push hook runs ruff** — Format and lint are enforced at push time. Several pushes required follow-up commits to drop unused imports from trimmed integration files (e.g., commit `fdd02ec` cleaned up `SemanticVersion`/`ValidationError` after the svar_link extraction). When trimming an integration file, **proactively run** `pixi run -e dev ruff check tests/integration/<path>.py --fix` before committing to fold the fix into the same commit.

4. **Plan-doc bodies vs source bodies** — Several plans had cases where my plan reproduced test bodies from incomplete reads (the insertion_fill plan was the worst — I hand-reconstructed kernel-test bodies that differed significantly from source). **Rule going forward:** before writing a plan that extracts tests, read every test body in full and quote verbatim. When a plan says "copy from source", the implementer must actually verify before pasting.

5. **`_resolve_svar` reset state in the integration file** — After Phase 5 svar_link extraction, `tests/integration/dataset/test_svar_link.py` no longer imports `SemanticVersion` or `ValidationError` (Phase 5 ruff cleanup). Don't re-add those if revisiting.

6. **1kg slow tests in the worktree** — The worktree at `.claude/worktrees/test-suite-overhaul/tests/data/1kg/` is NOT populated by default. To run slow tier here you'd need `pixi run -e dev gen-1kg`. The main checkout at `/Users/david/projects/GenVarLoader/tests/data/1kg/` IS populated; for slow-tier verification, `git checkout --detach worktree-test-suite-overhaul` in the main checkout, run tests, then `git checkout main` to return.

7. **Second basename-collision case (tracks-broader plan)** — `test_write_tracks.py` collided between `tests/integration/dataset/` (3 Keeps) and the new `tests/unit/dataset/` (1 Port). Resolved by renaming the integration file to `test_write_tracks_e2e.py`, consistent with the reconstruct-plan precedent (gotcha 1). Watch the same pattern for any future split that wants the natural integration filename.

8. **Stale `bench_cpu_gpu.py` in tracks dir** — `tests/integration/tracks/bench_cpu_gpu.py` is a typer CLI benchmark script (not a pytest-collected test). It imports `from utils import nonoverlapping_intervals` lazily inside `main()`. The tracks-broader plan moved `utils.py` to `tests/unit/tracks/`, so this script's lazy import now fails — but the script was already broken (imports `genvarloader.dataset.intervals`, `genvarloader.types`, `genvarloader.utils` — old module paths that no longer exist post-rename). No test regression; flag for eventual deletion or restoration.

9. **Conftest `reference` fixture carve-out** — The original conftest convention (docstring) said fixtures yield *paths*, not opened Datasets, because Datasets are expensive. Phase 5 ref/fasta added a `reference` fixture to conftest as a deliberate exception: `Reference.from_path(path, in_memory=False)` reads only FAI/GZI metadata, so session-scoped centralization is cheap. The fixture's own docstring records the rationale. Future work that wants to add another "opened object" fixture should explicitly justify the same way or stick to paths.

---

## Recommended next plan

All component-level plans are complete. The next two phases (per design spec):

1. **Phase 6 (integration trim)** — Review each remaining integration-tier file. Where unit coverage now strictly subsumes an integration test, delete the redundancy. Candidates worth examining first: integration files where the unit-tier extraction left a thin shell (e.g. `test_dummy_dataset_insertion_fill.py`, `test_ref_ds_splicing.py`, `test_write_tracks_e2e.py`, `test_dataset.py`).
2. **Phase 7 (CI report)** — Wire `htmlcov/` upload into CI per the design spec.

Phase 6 should land as its own plan; Phase 7 is a small CI-config change that can probably ride along with whatever PR completes the overhaul.

---

## How to resume

1. `cd /Users/david/projects/GenVarLoader/.claude/worktrees/test-suite-overhaul && git status` — should be on `worktree-test-suite-overhaul`, clean.
2. Read this file.
3. Read the design spec (intent) and audit (test classification) if you want full context.
4. Pick the next plan from "Recommended next plan" above.
5. Follow the same pattern as the executed plans: ask user for scope (one component at a time per their preference), write the plan via `superpowers:writing-plans`, execute via inline (small) or subagent-driven (larger).
