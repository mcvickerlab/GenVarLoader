# CI hg38 Reference Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop CI from re-downloading the ~1 GB hg38 reference on every job by caching the prepared `tests/data/fasta/` directory across runs.

**Architecture:** Add an `actions/cache@v4` step keyed on a static string derived from the reference's pooch `known_hash` to both jobs in `.github/workflows/test.yaml`, placed before the step that runs the pixi test task. On a cache hit, `gen` finds `hg38.fa.gz` (hash-valid), `hg38.fa.bgz`, and the index already present and skips the download, bgzip, and faidx. Add a comment at the `known_hash` line so the cache key gets bumped if the reference is ever swapped.

**Tech Stack:** GitHub Actions (`actions/cache@v4`), pixi, pooch, samtools/bgzip (used by the generation script — not modified).

**Spec:** `docs/superpowers/specs/2026-05-26-ci-reference-cache-design.md`

---

## Conventions used throughout

- Use `rtk git ...` for git commands (per CLAUDE.md).
- This change only touches CI config and a comment; there is no local pytest to run. Verification is by YAML validity and by inspecting a CI run after merge.
- Working directory: `/Users/david/projects/GenVarLoader/.claude/worktrees/test-coverage-initiative` on branch `worktree-test-coverage-initiative`. If the branch differs, adapt.

---

## File structure

- `.github/workflows/test.yaml` — add one cache step to the `pytest` matrix job and one to the `coverage` job. Both restore/save `tests/data/fasta` under the same key, so they share the cache.
- `tests/data/generate_ground_truth.py` — add a one-line comment above the `known_hash=` argument pointing maintainers to the CI cache key. No behavior change.

---

## Task 1: Add the reference cache step to both CI jobs

**Files:**
- Modify: `.github/workflows/test.yaml`

The current file (for reference — do not assume line numbers, match on content):

```yaml
name: Test

on:
  pull_request:
    branches:
      - main
  push:
    branches:
      - main
      - stable
  workflow_dispatch:

jobs:
  pytest:
    runs-on: ubuntu-latest
    name: "pytest on Python ${{ matrix.environment }}"
    strategy:
      fail-fast: false
      matrix:
        environment: [py310, py311, py312, py313]
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: ${{ matrix.environment }}
          locked: false
      - name: Test
        run: pixi run -e ${{ matrix.environment }} test

  coverage:
    runs-on: ubuntu-latest
    name: "coverage (py312)"
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: py312
          locked: false
      - name: Run pytest with coverage
        run: pixi run -e py312 test-cov
      - name: Upload HTML coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: htmlcov
          path: htmlcov/
          if-no-files-found: error
      - name: Upload coverage.xml
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
          if-no-files-found: error
```

- [ ] **Step 1: Insert the cache step in the `pytest` job, between `Setup pixi` and `Test`**

In the `pytest` job, replace this block:

```yaml
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: ${{ matrix.environment }}
          locked: false
      - name: Test
        run: pixi run -e ${{ matrix.environment }} test
```

with:

```yaml
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: ${{ matrix.environment }}
          locked: false
      - name: Cache hg38 reference
        uses: actions/cache@v4
        with:
          path: tests/data/fasta
          # Static key derived from the pooch known_hash in
          # tests/data/generate_ground_truth.py. Bump this if the reference changes.
          key: hg38-ref-c1dd8706
      - name: Test
        run: pixi run -e ${{ matrix.environment }} test
```

- [ ] **Step 2: Insert the same cache step in the `coverage` job, between `Setup pixi` and `Run pytest with coverage`**

In the `coverage` job, replace this block:

```yaml
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: py312
          locked: false
      - name: Run pytest with coverage
        run: pixi run -e py312 test-cov
```

with:

```yaml
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: py312
          locked: false
      - name: Cache hg38 reference
        uses: actions/cache@v4
        with:
          path: tests/data/fasta
          # Static key derived from the pooch known_hash in
          # tests/data/generate_ground_truth.py. Bump this if the reference changes.
          key: hg38-ref-c1dd8706
      - name: Run pytest with coverage
        run: pixi run -e py312 test-cov
```

- [ ] **Step 3: Verify the YAML still parses and contains exactly two cache steps**

Run:

```bash
pixi run -e dev python -c "import yaml; d=yaml.safe_load(open('.github/workflows/test.yaml')); import json; print(json.dumps([s.get('name') for j in d['jobs'].values() for s in j['steps']], indent=0))"
```

Expected: the printed list of step names includes `"Cache hg38 reference"` twice (once in each job), and no error is raised.

Also confirm the key appears exactly twice:

```bash
grep -c "hg38-ref-c1dd8706" .github/workflows/test.yaml
```

Expected: `2`

- [ ] **Step 4: Commit**

```bash
rtk git add .github/workflows/test.yaml
rtk git commit -m "ci: cache hg38 reference to skip repeated 1GB download"
```

---

## Task 2: Add the bump-the-key comment at the reference known_hash

**Files:**
- Modify: `tests/data/generate_ground_truth.py`

The current code (match on content, not line numbers):

```python
    reference = Path(
        pooch.retrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            known_hash="c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4",
            fname="hg38.fa.gz",
            path=fasta_dir,
        )
    ).resolve()
```

- [ ] **Step 1: Add a comment above the `known_hash` argument**

Replace:

```python
    reference = Path(
        pooch.retrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            known_hash="c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4",
            fname="hg38.fa.gz",
            path=fasta_dir,
        )
    ).resolve()
```

with:

```python
    reference = Path(
        pooch.retrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            # If this reference (and thus known_hash) changes, bump the CI cache key
            # `hg38-ref-<...>` in .github/workflows/test.yaml so CI re-fetches it.
            known_hash="c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4",
            fname="hg38.fa.gz",
            path=fasta_dir,
        )
    ).resolve()
```

- [ ] **Step 2: Verify the script still imports/parses**

Run:

```bash
pixi run -e dev python -c "import ast; ast.parse(open('tests/data/generate_ground_truth.py').read()); print('ok')"
```

Expected: prints `ok` with no SyntaxError.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/data/generate_ground_truth.py
rtk git commit -m "docs: note CI cache key must be bumped if hg38 reference changes"
```

---

## Task 3: Push and verify on a real CI run

**Files:** none (verification only)

- [ ] **Step 1: Push the branch**

```bash
rtk git push
```

- [ ] **Step 2: Confirm the cache step appears and behaves in CI**

After the push triggers CI (PR #195 or the active PR for this branch), inspect the latest run:

```bash
gh pr view 195 --json statusCheckRollup -q '.statusCheckRollup[] | "\(.name): \(.conclusion // .state)"'
```

Then open one `pytest` job and confirm:

```bash
# Replace <job-id> with a pytest job id from the run.
gh run view --job <job-id> --log | grep -iE "Cache hg38 reference|cache (hit|miss|restored|saved)|hgdownload.soe.ucsc.edu" | head
```

Expected on the **first** run after this change: a "Cache hg38 reference" step that reports a cache miss, the UCSC download line present (cold cache), and the cache saved at job end.

Expected on a **subsequent** run (same key, after one warm run): the step reports a cache hit / "Cache restored", and the UCSC download line does **not** appear.

- [ ] **Step 3: Confirm test results are unchanged**

The run's pass/skip/xfail counts should match the pre-change baseline (no test behavior changed). If any test now fails, treat it as a regression to investigate — do not adjust the cache to mask it.

---

## Self-review notes

**Spec coverage:**
- "Add cache step to `pytest` job and `coverage` job" → Task 1, Steps 1–2.
- "Cache `tests/data/fasta` under a static key `hg38-ref-c1dd8706`" → Task 1, Steps 1–2 (same key in both, so they share cache).
- "No `restore-keys` fallback" → satisfied; no `restore-keys` is added.
- "Comment at the `known_hash` line pointing to the cache key" → Task 2.
- "Success criteria: subsequent runs skip download; editing the gen script does not invalidate the cache" → Task 3, Step 2 verifies the skip; the static key (not `hashFiles`) inherently satisfies the no-invalidation-on-script-edit requirement.
- Non-goals (no test-data migration, no committed subset, no caching of generated dirs, no pixi/script behavior change) → respected; only a cache step and a comment are added.

**Placeholder scan:** `<job-id>` in Task 3 Step 2 is an intentional runtime value the operator fills from the live run, not an unfilled plan placeholder; every code/YAML block is complete.

**Consistency:** the cache key `hg38-ref-c1dd8706`, path `tests/data/fasta`, and action `actions/cache@v4` are identical across both jobs and match the spec.
