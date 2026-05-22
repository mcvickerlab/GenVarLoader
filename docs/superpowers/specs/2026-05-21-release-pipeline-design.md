# Release Pipeline Redesign

**Date:** 2026-05-21
**Status:** Design

## Problem

The current release flow has four loosely-connected workflows triggered via `workflow_run`:

- `bump.yaml` (workflow_dispatch) — commitizen bumps version, pushes tag, creates GH release from `body.md`.
- `publish.yaml` (`workflow_run` on bump success) — maturin builds wheels, publishes to PyPI.
- `merge.yaml` (`workflow_run` on bump success) — rebases `stable` onto `main`.
- `test.yaml` — PRs only.

Concrete failure modes this causes:

1. **No mid-pipeline recovery.** If `publish.yaml` fails on one wheel platform after `bump.yaml` succeeded, the tag and GH release already exist; there is no way to re-run only `publish.yaml` pinned to that tag. Recovery is manual (delete tag, force-push, repeat).
2. **`workflow_run` builds off `main` HEAD, not the tag.** Any commit landing between bump and publish would be built into the wheels.
3. **No dry-run.** No way to preview a release without committing to it.
4. **No partial-bump recovery.** If commitizen pushes a tag but the bump commit fails to push, the repo enters a broken state requiring manual fix.
5. **No lint CI.** `ruff` and `basedpyright` are listed in `CLAUDE.md` but not enforced anywhere.
6. **Tests do not run on push.** A bad merge to `main` or `stable` ships untested.
7. **Conventional-commit format is unenforced.** A non-conforming commit on `main` breaks auto-bump silently until the next release attempt.

SeqPro (sibling repo) recently solved (1)–(4) with a `release-pipeline.yaml` orchestrator + reusable workflows. This spec adopts that pattern, adapted to GVL specifics, and adds (5)–(7).

## Design

### Workflow layout (after this change)

```
.github/workflows/
├── release-pipeline.yaml   # NEW: orchestrator (workflow_dispatch only)
├── bump.yaml               # REWRITE: workflow_call, dry-run, partial recovery
├── release.yaml            # NEW: split from bump; creates GH release from changelog
├── publish.yaml            # REWRITE: workflow_call, tag-pinned checkout, dry-run, uv publish
├── merge.yaml              # REWRITE: workflow_call, dry-run summary
└── test.yaml               # UPDATE: run on push to main/stable, add cargo test job
```

### release-pipeline.yaml (orchestrator)

Single entry point, `workflow_dispatch` only. Inputs:

- `increment` (choice: `auto` | `PATCH` | `MINOR` | `MAJOR`, default `auto`) — forces a bump level, or lets commitizen decide.
- `skip_bump` (boolean, default false) — skip the bump job and re-run release/publish/merge against an existing tag. Recovery mode.
- `tag` (string, optional) — required when `skip_bump=true`; must exist on the remote and match `vX.Y.Z`.
- `dry_run` (boolean, default false) — propagated to every job; no external side effects.

Jobs (sequential, each calling a reusable workflow):

1. `resolve` — validates inputs. When `skip_bump=true`, asserts `tag` matches `^v[0-9]+\.[0-9]+\.[0-9]+$` and exists on the remote.
2. `bump` — `uses: ./.github/workflows/bump.yaml`, skipped when `skip_bump=true`. Emits the resulting tag.
3. `release` — `uses: ./.github/workflows/release.yaml`, with `tag = needs.bump.outputs.tag || needs.resolve.outputs.tag`.
4. `publish` — `uses: ./.github/workflows/publish.yaml`, with the same resolved tag.
5. `merge` — `uses: ./.github/workflows/merge.yaml`.

Each downstream job uses `if: always() && needs.<previous>.result == 'success'` so a `skipped` bump still allows the rest to proceed.

### bump.yaml (rewrite)

`workflow_call` + `workflow_dispatch`. Inputs: `increment`, `dry_run`. Output: `tag` (resolved version string with `v` prefix).

Steps:

1. Checkout with full history and `secrets.COMMITIZEN` token.
2. `pip install commitizen --quiet`.
3. **Partial-bump detection** — compare `cz version --project` against the latest `vX.Y.Z` tag on the remote. If they differ, a previous bump pushed the tag but not the files; set `partial=true`.
4. `Bump version (real)` — runs `commitizen-action` when `partial=false` and not dry-run. Honors the `increment` input when not `auto`.
5. `Bump version (dry-run)` — runs `cz bump --dry-run --yes` instead when `dry_run=true`.
6. `Fix partial bump` — when `partial=true` (orphan tag on remote without matching pyproject version): run `cz bump --files-only --yes` to update `pyproject.toml` and `docs/source/changelog.md` to a new version that supersedes the orphan, commit with `bump: version <new>`, push to `main`, then push the new tag. The orphan tag is left in place; the next release moves past it. (This matches SeqPro's recovery — it deliberately advances past the bad state rather than trying to rewrite history.)
7. `Emit resulting tag` — sets output to `v$(cz version --project)`.

GVL-specific deltas from SeqPro:

- Tag format is `v$version` (per `pyproject.toml`), not bare `X.Y.Z`. All regexes use `^v[0-9]+\.[0-9]+\.[0-9]+$`. The output `tag` carries the `v` prefix.
- Changelog path is `docs/source/changelog.md`, not `CHANGELOG.md`.
- The current changelog has duplicated `# Changelog` headers at the top — leave alone; the awk extractor in `release.yaml` keys on `## v` not `# Changelog`.
- Secret name is `COMMITIZEN` (existing), not `GH_TOKEN`.

### release.yaml (new, split out)

Triggers: `push` on tag `v[0-9]+.[0-9]+.[0-9]+`, `workflow_dispatch`, `workflow_call`.

Inputs (call form): `tag` (required), `dry_run` (default false).

Steps:

1. Checkout full history.
2. **Resolve tag** — from `GITHUB_REF` on push, or `inputs.tag` otherwise.
3. **Extract changelog section** — `awk` that prints lines between `## $tag ` and the next `## ` heading from `docs/source/changelog.md`, writes to `body.md`. Heading format in current changelog is `## v0.24.1 (2026-05-13)`.
4. **Dry-run summary** — when called with `dry_run=true`, write `body.md` into `$GITHUB_STEP_SUMMARY` inside a fenced block and stop.
5. **Create GitHub release** — `softprops/action-gh-release@v2` (current GVL version; upgrade to v3 in same PR) with `tag_name`, `body_path: body.md`, token from `secrets.COMMITIZEN`.

### publish.yaml (rewrite)

`workflow_call` + `workflow_dispatch`. Inputs: `dry_run` (default false), `tag` (default `""`; checked out via `actions/checkout` `ref:` when set).

Job structure mirrors SeqPro's adapted from maturin's autogen template:

- `linux` matrix: x86_64, x86, aarch64, armv7, s390x, ppc64le on `ubuntu-22.04`. Both GIL and free-threaded (`python3.14t`) builds.
- `musllinux` matrix: x86_64, x86, aarch64, armv7 with `manylinux: musllinux_1_2`. Both GIL and 3.14t.
- `windows` matrix: x64, x86 on `windows-latest`, aarch64 on `windows-11-arm`. Both GIL and 3.14t.
- `macos` matrix: x86_64 on `macos-15-intel`, aarch64 on `macos-latest`. Both GIL and 3.14t.
- `sdist` — single `ubuntu-latest` job.
- `release` — `needs: [linux, musllinux, windows, macos, sdist]`. Downloads all artifacts, runs `actions/attest-build-provenance@v3` over `wheels-*/*`, then either `uv publish 'wheels-*/*'` (real) or writes a dry-run summary listing the wheels.

GVL-specific deltas from SeqPro:

- Add the free-threaded build steps (SeqPro has them; GVL currently doesn't — adopt now while rewriting).
- Header comment notes the file is hand-edited from `maturin generate-ci`; lists the edits to re-apply on regen (workflow_call block, dry-run gate, tag checkout, free-threaded steps).

### merge.yaml (rewrite)

`workflow_call` + `workflow_dispatch`. Input: `dry_run` (default false).

Steps:

1. Checkout `stable` with `secrets.COMMITIZEN` token.
2. Configure git identity.
3. `git rebase origin/main`.
4. If `dry_run=false`: `git push origin stable`. If `dry_run=true`: write `git log --oneline origin/stable..HEAD` to `$GITHUB_STEP_SUMMARY`.

### test.yaml (update)

Add `push` triggers for `main` and `stable` (in addition to existing `pull_request` to `main`).

Bump `prefix-dev/setup-pixi` to latest stable (verify at implementation time).

Add a second job `cargo-test` running `cargo test --release` on `ubuntu-latest`. Reuse the same pixi env to ensure the Rust toolchain matches. Keep the existing Python matrix job as `pytest`.

## Out of scope

- **`lint.yaml`** — deferred to a future batch of work that will migrate type checking from basedpyright to pyrefly. Lint CI (ruff + pyrefly + `cz check`) lands then.
- Docs deploy workflow (GVL presumably uses ReadTheDocs; not in scope per user selection).
- Removing `legacy_tag_formats = ['$version']` from `pyproject.toml` — old tags exist; leave it.
- Backporting the partial-bump recovery to existing broken tags (none known).
- Changing the `stable` branch model.

## Testing & rollout

1. Land the new workflows on a feature branch.
2. Trigger `release-pipeline.yaml` with `dry_run=true` from the branch via `workflow_dispatch` — verify each job's step summary.
3. Trigger with `skip_bump=true, tag=v0.24.1, dry_run=true` — verifies recovery path against an existing tag.
4. Merge to `main`.
5. Next real release uses the new pipeline; if it fails mid-flight, re-run with `skip_bump=true, tag=<the tag>`.

## Open questions

None — all decisions made during brainstorming. Implementation plan can proceed.

---

## Post-implementation notes (2026-05-21)

The pipeline shipped and released v0.25.0 on the day of design. Six real bugs were caught — three by code review pre-merge, three by post-merge validation runs. Capturing them here so the next person to adapt this pattern (or copy it to another repo) doesn't repeat them.

### Bugs caught in code review (before merge)

These were flagged on PR #168 by the code-quality reviewer subagent. All would have surfaced on the first or second use of the pipeline.

1. **`push:` trigger on `release.yaml` races with `workflow_call`.** Both fire on the tag push from commitizen; second one duplicates or errors. Removed the `push:` trigger; orchestrator is the sole entry point. Standalone `workflow_dispatch` remains as an escape hatch.
2. **Partial-bump auto-recovery is unsafe.** The original spec proposed `cz bump --files-only` + new tag push when an orphan tag is detected. Commitizen computes the next version from `pyproject.toml`, which can collide with the orphan. Replaced with a hard-fail that prints actionable remediation (delete orphan, or hand-bump pyproject to match). Auto-recovery in this scenario is dangerous; humans should resolve.
3. **`cz bump --dry-run` does not modify files**, so reading `cz version --project` after it returns the *current* version, not the projected one. Fixed by parsing the projected version from `cz bump --dry-run` stdout (`-> X.Y.Z` or `tag to create: vX.Y.Z`).
4. **`sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}`** is broken under `workflow_call`. `github.ref` in a reusable workflow is the **caller's** ref, never a tag ref — so sccache was always on, defeating the disable-on-release intent. Changed to `sccache: ${{ inputs.tag == '' }}` (tag pinned = release build = no sccache).
5. **`merge.yaml` had `environment: pypi` and `id-token: write`** copy-pasted from the old workflow. Neither is needed — the rebase + push authenticates via the `COMMITIZEN` PAT. Removed both.
6. **`commitizen-action@master`** was unpinned. Pinned to `0.27.1`. Worth re-pinning periodically.

### Bugs caught in post-merge validation

These are all variants of one underlying issue: **GitHub Actions evaluates `github.*` context against the *originating* event, not against `workflow_call`.** In a reusable workflow invoked by `release-pipeline.yaml`, `github.event_name` is `workflow_dispatch` (from the user's button click on the orchestrator), and `github.ref` is the orchestrator's dispatch ref. This trips up anything that tries to detect "am I being called by another workflow?"

7. **Dry-run UX bug: missing changelog section.** In the fresh-bump dry-run path, the projected tag (e.g. `v0.25.0`) has no section in `docs/source/changelog.md` because commitizen only writes it during the real bump. The awk extractor errored, failing the run. Fix (PR #171): when `body.md` is empty AND `inputs.dry_run == true`, emit a placeholder body and continue.
8. **Dry-run gate in `release.yaml` was structurally broken** (PR #172). The condition `github.event_name == 'workflow_call' && inputs.dry_run == true` always evaluates false in the reusable workflow (see above). So `Dry-run summary` skipped and `Create GitHub release` ran. `softprops/action-gh-release@v3` then auto-created the tag `v0.25.0` from `main` HEAD with the placeholder body. Cleanup required deleting the orphan release and tag.
    - **Lesson:** never gate dry-run on `github.event_name == 'workflow_call'`. Always check `inputs.dry_run` directly. The same input should be declared on both `workflow_call` and `workflow_dispatch`.
    - PyPI publish, push-to-stable, and the bump commit were unaffected — those gates correctly used `inputs.dry_run`.
9. **`publish.yaml` can't pin to a projected tag that doesn't exist** (PR #173). After fixing #8, the next dry-run failed because `publish.yaml`'s `actions/checkout` was passed `ref: v0.25.0` — a tag that never exists in dry-run (no real bump). Fix: in `release-pipeline.yaml`, only pass the tag to `publish` when `inputs.dry_run == false` OR `inputs.skip_bump == true`. Otherwise pass `''` so checkout falls back to the caller's ref.
10. **`release.yaml`'s checkout had no `ref:`** (PR #175). The first real release failed because the release job pulled the orchestrator's dispatch commit — the **parent** of the bump commit pushed seconds earlier — so the changelog at that ref didn't have the new version's section yet. Fix: `ref: ${{ inputs.dry_run == false && inputs.tag || '' }}`. Real mode pins to the just-pushed tag; dry-run falls back to caller ref.

### Lessons for the next adapter

- **`workflow_run` triggers are also a footgun** (the old GVL pipeline used them) — they look like they will work and then silently miss events or pull the wrong ref. Prefer reusable workflows (`workflow_call`).
- **Reusable workflows always pull the *dispatcher's* ref by default.** If a sibling job has pushed a commit since dispatch (like `bump`), and a downstream job needs that commit, pin its checkout to the new tag explicitly. This bit us twice (issues 9 and 10) for the same reason.
- **`github.event_name` and `github.ref` lie inside reusable workflows.** Don't use them for dry-run gating, recovery detection, or tag-vs-branch dispatch. Use `inputs.*` instead.
- **Dry-run is genuinely valuable but takes deliberate design** at every step that touches external state: changelog parsing, tag checkout, release creation, publish, push. Each needs a dry-run-aware fallback that doesn't depend on artifacts only created in real mode.
- **The recovery path (`skip_bump=true, tag=<existing>`) earned its keep on the first real release** — the initial run failed mid-flight (bug 10) and recovery completed the release/publish/merge against the orphan tag without any manual git surgery. Keep this affordance in any adaptation.

### Things that worked well

- Code-quality reviewer subagent caught the high-impact bugs before merge (six issues, all real).
- The orchestrator's `if: always() && needs.X.result == 'success'` gating let us isolate failures to their job rather than aborting the run — useful for diagnosis.
- Tag-pinned checkouts in publish.yaml (Task 4) meant the recovery-path dry-run validated the full wheel matrix against a stable, known tag. That validation gave confidence the build itself was sound; the bugs we found later were all in the surrounding plumbing.
- All cleanup of the dry-run misfire was scriptable (`gh release delete --cleanup-tag`). The blast radius never reached PyPI, `stable`, or the source tree.
