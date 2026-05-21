# Release Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile `workflow_run` release chain with a single orchestrated, dry-runnable, recoverable release pipeline (SeqPro pattern), plus tighten the test workflow.

**Architecture:** One `release-pipeline.yaml` orchestrator (workflow_dispatch only) that calls four reusable workflows (`bump`, `release`, `publish`, `merge`) in sequence. Each reusable workflow has a `dry_run` input and is independently triggerable. Wheels are built from the resolved tag (not branch HEAD). Test workflow gains push triggers and a cargo job. `lint.yaml` is explicitly out of scope (deferred to pyrefly migration).

**Tech Stack:** GitHub Actions, commitizen (`v$version` tags), maturin, `uv publish`, pixi, `softprops/action-gh-release@v3`, `PyO3/maturin-action@v1`, `astral-sh/setup-uv@v7`.

**Reference spec:** `docs/superpowers/specs/2026-05-21-release-pipeline-design.md`

**Validation note:** GitHub Actions has no offline test harness. Each task's verification step is either (a) `actionlint` if available, otherwise a YAML syntax check via `python -c "import yaml; yaml.safe_load(open('...'))"`, and (b) a `workflow_dispatch`-triggered dry run after the orchestrator lands (Task 7). Do not delete the old workflow files until Task 7 passes.

**Working branch:** Create `feat/release-pipeline` off `main` before Task 1. All commits go there.

---

### Task 0: Branch setup

**Files:** none

- [ ] **Step 1: Create branch**

```bash
git checkout main
git pull --ff-only
git checkout -b feat/release-pipeline
```

- [ ] **Step 2: Verify clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean` on `feat/release-pipeline`.

---

### Task 1: Refactor `bump.yaml` to `workflow_call` with dry-run + partial recovery

**Files:**
- Modify: `.github/workflows/bump.yaml` (full rewrite)

**Context:** The current `bump.yaml` does bump + GH release in one job and is triggered only by `workflow_dispatch`. We split off the GH release into `release.yaml` (Task 2) and convert this to a reusable workflow that emits the resolved tag.

GVL specifics:
- `pyproject.toml` has `tag_format = "v$version"`, so tags are `vX.Y.Z`. The output tag MUST carry the `v` prefix.
- Secret name is `COMMITIZEN` (not `GH_TOKEN`).
- Changelog file is `docs/source/changelog.md`.

- [ ] **Step 1: Replace `.github/workflows/bump.yaml` with the new content**

```yaml
name: Bump version

on:
  workflow_call:
    inputs:
      increment:
        description: "Force a specific bump (auto|PATCH|MINOR|MAJOR)"
        type: string
        required: false
        default: "auto"
      dry_run:
        description: "Compute the bump but do not commit/push"
        type: boolean
        required: false
        default: false
    outputs:
      tag:
        description: "Resolved tag (with v prefix) after the bump, or projected tag in dry-run"
        value: ${{ jobs.bump.outputs.tag }}
  workflow_dispatch:
    inputs:
      increment:
        description: "Force a specific bump"
        type: choice
        required: false
        default: "auto"
        options: ["auto", "PATCH", "MINOR", "MAJOR"]
      dry_run:
        description: "Compute the bump but do not commit/push"
        type: boolean
        required: false
        default: false

jobs:
  bump:
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.emit.outputs.tag }}
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.COMMITIZEN }}

      - name: Install commitizen
        run: pip install commitizen --quiet

      - name: Detect partial bump
        id: detect
        run: |
          PROJECT_VERSION=$(cz version --project)
          LATEST_TAG=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
          # Strip v prefix from tag for comparison with project version
          LATEST_TAG_NOPREFIX="${LATEST_TAG#v}"
          if [ -n "$LATEST_TAG" ] && [ "$LATEST_TAG_NOPREFIX" != "$PROJECT_VERSION" ]; then
            echo "partial=true" >> "$GITHUB_OUTPUT"
            echo "orphan_tag=$LATEST_TAG" >> "$GITHUB_OUTPUT"
            echo "::warning::Detected orphan tag $LATEST_TAG (project version is $PROJECT_VERSION). Will advance past it."
          else
            echo "partial=false" >> "$GITHUB_OUTPUT"
          fi

      - name: Bump version (real)
        if: steps.detect.outputs.partial != 'true' && inputs.dry_run == false
        uses: commitizen-tools/commitizen-action@master
        with:
          github_token: ${{ secrets.COMMITIZEN }}
          increment: ${{ inputs.increment != 'auto' && inputs.increment || '' }}

      - name: Bump version (dry-run)
        if: steps.detect.outputs.partial != 'true' && inputs.dry_run == true
        run: |
          ARGS="--dry-run --yes"
          if [ "${{ inputs.increment }}" != "auto" ]; then
            ARGS="$ARGS --increment ${{ inputs.increment }}"
          fi
          cz bump $ARGS

      - name: Fix partial bump (advance past orphan tag)
        if: steps.detect.outputs.partial == 'true' && inputs.dry_run == false
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          cz bump --files-only --yes
          NEW_VERSION=$(cz version --project)
          git add pyproject.toml docs/source/changelog.md
          git commit -m "bump: version $NEW_VERSION"
          git tag "v$NEW_VERSION"
          git push origin main
          git push origin "v$NEW_VERSION"

      - name: Emit resulting tag
        id: emit
        run: |
          VERSION=$(cz version --project)
          TAG="v$VERSION"
          echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          echo "Resolved tag: $TAG"
```

- [ ] **Step 2: YAML syntax check**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/bump.yaml'))" && echo OK
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bump.yaml
git commit -m "ci: refactor bump.yaml to workflow_call with dry-run and partial recovery"
```

---

### Task 2: Create `release.yaml` (GH release creation from changelog section)

**Files:**
- Create: `.github/workflows/release.yaml`

**Context:** The old `bump.yaml` created the GH release from commitizen's `body.md`. We split that out so the release body comes from `docs/source/changelog.md` directly — this makes the GH release reproducible from the tag alone.

Heading format in `docs/source/changelog.md` is `## v0.24.1 (2026-05-13)` for recent versions. The awk script keys on `^## $tag ` (note the trailing space, which ensures we don't match prefixes).

- [ ] **Step 1: Create `.github/workflows/release.yaml`**

```yaml
name: Create release

on:
  push:
    tags:
      - 'v[0-9]+.[0-9]+.[0-9]+'
  workflow_dispatch:
    inputs:
      tag:
        description: 'Tag to release (e.g. v0.25.0)'
        required: true
  workflow_call:
    inputs:
      tag:
        description: 'Tag to release (with v prefix)'
        type: string
        required: true
      dry_run:
        description: 'Build the release body but do not create a GH release'
        type: boolean
        required: false
        default: false

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Resolve tag
        id: tag
        run: |
          case "${{ github.event_name }}" in
            push)              echo "tag=${GITHUB_REF#refs/tags/}" >> "$GITHUB_OUTPUT" ;;
            workflow_dispatch) echo "tag=${{ inputs.tag }}" >> "$GITHUB_OUTPUT" ;;
            workflow_call)     echo "tag=${{ inputs.tag }}" >> "$GITHUB_OUTPUT" ;;
          esac

      - name: Extract changelog section
        run: |
          TAG="${{ steps.tag.outputs.tag }}"
          awk -v tag="$TAG" '
            $0 ~ "^## " tag " " { flag=1; next }
            flag && /^## / { exit }
            flag { print }
          ' docs/source/changelog.md > body.md
          if [ ! -s body.md ]; then
            echo "::error::No changelog section found for $TAG in docs/source/changelog.md"
            exit 1
          fi

      - name: Dry-run summary
        if: github.event_name == 'workflow_call' && inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: release ${{ steps.tag.outputs.tag }}"
            echo
            echo "Would create a GitHub Release with the following body:"
            echo
            echo '```markdown'
            cat body.md
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"

      - name: Create GitHub release
        if: ${{ !(github.event_name == 'workflow_call' && inputs.dry_run == true) }}
        uses: softprops/action-gh-release@v3
        with:
          tag_name: ${{ steps.tag.outputs.tag }}
          body_path: body.md
          token: ${{ secrets.COMMITIZEN }}
```

- [ ] **Step 2: YAML syntax check**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Smoke-test the awk extractor locally against an existing tag**

Run:
```bash
awk -v tag="v0.24.1" '
  $0 ~ "^## " tag " " { flag=1; next }
  flag && /^## / { exit }
  flag { print }
' docs/source/changelog.md | head -20
```
Expected: non-empty output that is the body of the v0.24.1 changelog section (the lines immediately under `## v0.24.1 (2026-05-13)`).

If empty, the heading format differs from what we expect — stop and debug before continuing.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yaml
git commit -m "ci: add release.yaml (GH release from changelog section)"
```

---

### Task 3: Refactor `merge.yaml` to `workflow_call` with dry-run

**Files:**
- Modify: `.github/workflows/merge.yaml` (full rewrite)

**Context:** Drop the `workflow_run` trigger (the orchestrator will call this). Add `dry_run`. Keep the existing rebase logic.

- [ ] **Step 1: Replace `.github/workflows/merge.yaml`**

```yaml
name: Merge main -> stable

on:
  workflow_call:
    inputs:
      dry_run:
        description: 'Compute the rebase but do not push'
        type: boolean
        required: false
        default: false
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Compute the rebase but do not push'
        type: boolean
        required: false
        default: false

jobs:
  merge:
    environment: pypi
    permissions:
      id-token: write
    runs-on: ubuntu-latest
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          ref: stable
          fetch-depth: 0
          token: ${{ secrets.COMMITIZEN }}
      - name: Config git
        run: |
          git config --global user.name "${GITHUB_ACTOR}"
          git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
      - name: Rebase stable onto main
        run: git rebase origin/main
      - name: Push to stable
        if: inputs.dry_run == false
        run: git push origin stable
      - name: Dry-run summary (skipped push)
        if: inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: would push the following to stable"
            echo
            echo '```'
            git log --oneline origin/stable..HEAD
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 2: YAML syntax check**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/merge.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/merge.yaml
git commit -m "ci: refactor merge.yaml to workflow_call with dry-run"
```

---

### Task 4: Rewrite `publish.yaml` (tag-pinned, dry-run, uv publish, 3.14t wheels)

**Files:**
- Modify: `.github/workflows/publish.yaml` (full rewrite)

**Context:** The current `publish.yaml` is autogenerated by `maturin generate-ci` and triggered via `workflow_run`. Replacing it with a `workflow_call` version that:
- Checks out the resolved tag (`inputs.tag`) so wheels are built from the actual release point.
- Adds free-threaded (`python3.14t`) wheel builds across all platforms.
- Uses `uv publish` instead of `maturin upload`.
- Gates publish behind `dry_run`.

The file is hand-edited from maturin's template; document this clearly at the top so a future regen doesn't blow it away.

- [ ] **Step 1: Replace `.github/workflows/publish.yaml`**

```yaml
# This file was autogenerated by `maturin generate-ci github` and has been
# hand-modified to support workflow_call from release-pipeline.yaml.
#
# To regenerate, run:
#
#   maturin generate-ci -o .github/workflows/publish.yaml github \
#     --platform linux musllinux manylinux windows macos
#
# Regenerating overwrites the trigger block and removes our edits. After
# regeneration, re-apply ALL of the following:
#
#   1. Replace the `on:` block with the workflow_call + workflow_dispatch block
#      below (inputs: dry_run, tag).
#   2. Remove the `on.workflow_run` block entirely.
#   3. Remove any per-job `if: ${{ github.event_name ... }}` guards added by
#      maturin (the orchestrator handles gating).
#   4. Add `with: ref: ${{ inputs.tag }}` to every `actions/checkout` step.
#   5. Re-add the free-threaded build steps (each platform's second
#      `PyO3/maturin-action` invocation with `-i python3.14t`).
#   6. Replace the final maturin `upload` step with the `uv publish` +
#      dry-run summary block.

name: Publish

on:
  workflow_call:
    inputs:
      dry_run:
        description: 'Build wheels but skip uv publish'
        type: boolean
        required: false
        default: false
      tag:
        description: 'Tag to check out and build wheels from (defaults to caller ref)'
        type: string
        required: false
        default: ''
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Build wheels but skip uv publish'
        type: boolean
        required: false
        default: false
      tag:
        description: 'Tag to check out and build wheels from (defaults to current ref)'
        type: string
        required: false
        default: ''

permissions:
  contents: read

jobs:
  linux:
    runs-on: ${{ matrix.platform.runner }}
    strategy:
      matrix:
        platform:
          - runner: ubuntu-22.04
            target: x86_64
            sccache: true
          - runner: ubuntu-22.04
            target: x86
            sccache: true
          - runner: ubuntu-22.04
            target: aarch64
            sccache: false
          - runner: ubuntu-22.04
            target: armv7
            sccache: false
          - runner: ubuntu-22.04
            target: s390x
            sccache: false
          - runner: ubuntu-22.04
            target: ppc64le
            sccache: false
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ inputs.tag }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.10'
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') && matrix.platform.sccache }}
          manylinux: auto
        env:
          CFLAGS_aarch64_unknown_linux_gnu: -march=armv8-a -D__ARM_ARCH=8
          CFLAGS_armv7_unknown_linux_gnueabihf: -march=armv7-a -D__ARM_ARCH=7
      - name: Build free-threaded wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist -i python3.14t
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') && matrix.platform.sccache }}
          manylinux: auto
        env:
          CFLAGS_aarch64_unknown_linux_gnu: -march=armv8-a -D__ARM_ARCH=8
          CFLAGS_armv7_unknown_linux_gnueabihf: -march=armv7-a -D__ARM_ARCH=7
      - name: Upload wheels
        uses: actions/upload-artifact@v6
        with:
          name: wheels-linux-${{ matrix.platform.target }}
          path: dist

  musllinux:
    runs-on: ${{ matrix.platform.runner }}
    strategy:
      matrix:
        platform:
          - runner: ubuntu-22.04
            target: x86_64
          - runner: ubuntu-22.04
            target: x86
          - runner: ubuntu-22.04
            target: aarch64
          - runner: ubuntu-22.04
            target: armv7
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ inputs.tag }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.10'
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
          manylinux: musllinux_1_2
      - name: Build free-threaded wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist -i python3.14t
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
          manylinux: musllinux_1_2
      - name: Upload wheels
        uses: actions/upload-artifact@v6
        with:
          name: wheels-musllinux-${{ matrix.platform.target }}
          path: dist

  windows:
    runs-on: ${{ matrix.platform.runner }}
    strategy:
      matrix:
        platform:
          - runner: windows-latest
            target: x64
            python_arch: x64
          - runner: windows-latest
            target: x86
            python_arch: x86
          - runner: windows-11-arm
            target: aarch64
            python_arch: arm64
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ inputs.tag }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.13'
          architecture: ${{ matrix.platform.python_arch }}
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.14t'
          architecture: ${{ matrix.platform.python_arch }}
      - name: Build free-threaded wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist -i python3.14t
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
      - name: Upload wheels
        uses: actions/upload-artifact@v6
        with:
          name: wheels-windows-${{ matrix.platform.target }}
          path: dist

  macos:
    runs-on: ${{ matrix.platform.runner }}
    strategy:
      matrix:
        platform:
          - runner: macos-15-intel
            target: x86_64
          - runner: macos-latest
            target: aarch64
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ inputs.tag }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.10'
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.14t'
      - name: Build free-threaded wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist -i python3.14t
          sccache: ${{ !startsWith(github.ref, 'refs/tags/') }}
      - name: Upload wheels
        uses: actions/upload-artifact@v6
        with:
          name: wheels-macos-${{ matrix.platform.target }}
          path: dist

  sdist:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ inputs.tag }}
      - name: Build sdist
        uses: PyO3/maturin-action@v1
        with:
          command: sdist
          args: --out dist
      - name: Upload sdist
        uses: actions/upload-artifact@v6
        with:
          name: wheels-sdist
          path: dist

  release:
    environment: pypi
    name: Release
    runs-on: ubuntu-latest
    needs: [linux, musllinux, windows, macos, sdist]
    permissions:
      id-token: write
      contents: write
      attestations: write
    steps:
      - uses: actions/download-artifact@v7
      - name: Generate artifact attestation
        uses: actions/attest-build-provenance@v3
        with:
          subject-path: 'wheels-*/*'
      - name: Install uv
        uses: astral-sh/setup-uv@v7
      - name: Publish to PyPI
        if: inputs.dry_run == false
        run: uv publish 'wheels-*/*'
      - name: Dry-run summary (skipped uv publish)
        if: inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: would have published the following wheels"
            echo
            ls -1 wheels-*/* | sed 's/^/- /'
          } >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 2: YAML syntax check**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish.yaml
git commit -m "ci: rewrite publish.yaml — workflow_call, tag-pinned, uv publish, 3.14t wheels"
```

---

### Task 5: Create `release-pipeline.yaml` orchestrator

**Files:**
- Create: `.github/workflows/release-pipeline.yaml`

- [ ] **Step 1: Create `.github/workflows/release-pipeline.yaml`**

```yaml
name: Release pipeline

on:
  workflow_dispatch:
    inputs:
      increment:
        description: "Force a specific bump (auto = let commitizen decide)"
        type: choice
        required: false
        default: "auto"
        options: ["auto", "PATCH", "MINOR", "MAJOR"]
      skip_bump:
        description: "Skip bump; release/publish/merge an existing tag instead"
        type: boolean
        required: false
        default: false
      tag:
        description: "Existing tag to release (required when skip_bump=true; with v prefix)"
        type: string
        required: false
        default: ""
      dry_run:
        description: "Run end-to-end without external side effects"
        type: boolean
        required: false
        default: false

permissions:
  contents: write
  id-token: write
  attestations: write

jobs:
  resolve:
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.out.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Validate inputs
        id: out
        run: |
          if [ "${{ inputs.skip_bump }}" = "true" ]; then
            TAG="${{ inputs.tag }}"
            if [ -z "$TAG" ]; then
              echo "::error::skip_bump=true requires a tag input"
              exit 1
            fi
            if ! echo "$TAG" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
              echo "::error::tag '$TAG' does not match vX.Y.Z"
              exit 1
            fi
            git fetch --tags --quiet
            if ! git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
              echo "::error::tag '$TAG' does not exist on the remote"
              exit 1
            fi
            echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          else
            echo "tag=" >> "$GITHUB_OUTPUT"
          fi

  bump:
    needs: resolve
    if: inputs.skip_bump == false
    uses: ./.github/workflows/bump.yaml
    secrets: inherit
    with:
      increment: ${{ inputs.increment }}
      dry_run: ${{ inputs.dry_run }}

  release:
    needs: [resolve, bump]
    if: |
      always() &&
      needs.resolve.result == 'success' &&
      (needs.bump.result == 'success' || needs.bump.result == 'skipped')
    uses: ./.github/workflows/release.yaml
    secrets: inherit
    with:
      tag: ${{ needs.bump.outputs.tag || needs.resolve.outputs.tag }}
      dry_run: ${{ inputs.dry_run }}

  publish:
    needs: [resolve, bump, release]
    if: always() && needs.release.result == 'success'
    uses: ./.github/workflows/publish.yaml
    secrets: inherit
    with:
      dry_run: ${{ inputs.dry_run }}
      tag: ${{ needs.bump.outputs.tag || needs.resolve.outputs.tag }}

  merge:
    needs: publish
    if: always() && needs.publish.result == 'success'
    uses: ./.github/workflows/merge.yaml
    secrets: inherit
    with:
      dry_run: ${{ inputs.dry_run }}
```

- [ ] **Step 2: YAML syntax check**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release-pipeline.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release-pipeline.yaml
git commit -m "ci: add release-pipeline.yaml orchestrator"
```

---

### Task 6: Update `test.yaml` (push triggers + cargo test job)

**Files:**
- Modify: `.github/workflows/test.yaml`

**Context:** Add `push` to `main`/`stable`. Add a `cargo-test` job. Bump pixi version. Keep the existing pytest matrix.

- [ ] **Step 1: Look up the latest pixi version**

Run:
```bash
gh api repos/prefix-dev/setup-pixi/releases/latest --jq .tag_name
```
Note the result — it will be used as the new `pixi-version`. (At time of writing the spec, current value is `v0.68.1`; substitute whatever the API returns.)

- [ ] **Step 2: Replace `.github/workflows/test.yaml`**

Replace `<LATEST_PIXI>` below with the value from Step 1.

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
          pixi-version: <LATEST_PIXI>
          cache: true
          environments: ${{ matrix.environment }}
          locked: false
      - name: Test
        run: pixi run -e ${{ matrix.environment }} test

  cargo-test:
    runs-on: ubuntu-latest
    name: "cargo test"
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable
      - name: Cache cargo
        uses: Swatinem/rust-cache@v2
      - name: cargo test
        run: cargo test --release
```

- [ ] **Step 3: YAML syntax check**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 4: Verify `cargo test` works locally**

Run: `cargo test --release`
Expected: tests pass (or, if no Rust tests exist yet, output is `running 0 tests ... test result: ok`). If compilation fails, fix or remove the cargo-test job before continuing — do not ship a CI job that's broken on day 1.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/test.yaml
git commit -m "ci: run tests on push to main/stable and add cargo test job"
```

---

### Task 7: End-to-end dry-run validation

**Files:** none modified

**Context:** This task validates the new pipeline against GitHub Actions before merge. Both `workflow_call` and reusable workflow `uses: ./...` paths require the workflow files to exist on the branch being dispatched — so we push the branch and trigger from it.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/release-pipeline
```

- [ ] **Step 2: Dispatch release-pipeline with dry_run=true (fresh-bump path)**

Run:
```bash
gh workflow run release-pipeline.yaml \
  --ref feat/release-pipeline \
  -f increment=auto \
  -f skip_bump=false \
  -f dry_run=true
```

Then watch the run:
```bash
sleep 5
gh run watch "$(gh run list --workflow=release-pipeline.yaml --branch=feat/release-pipeline --limit=1 --json databaseId --jq '.[0].databaseId')"
```

Expected: all five jobs (`resolve`, `bump`, `release`, `publish`, `merge`) succeed. No real tag pushed, no GH release created, no PyPI upload. Each job's "Dry-run summary" step appears in `$GITHUB_STEP_SUMMARY`.

If `bump` fails because the dry-run can't determine the increment (e.g., no conventional commits since last tag), this is acceptable as long as the error message is clear — the pipeline correctly stopped at the bump step.

- [ ] **Step 3: Dispatch release-pipeline with skip_bump=true (recovery path)**

Pick an existing tag, e.g. `v0.24.1`, then:

```bash
gh workflow run release-pipeline.yaml \
  --ref feat/release-pipeline \
  -f skip_bump=true \
  -f tag=v0.24.1 \
  -f dry_run=true
```

Watch:
```bash
sleep 5
gh run watch "$(gh run list --workflow=release-pipeline.yaml --branch=feat/release-pipeline --limit=1 --json databaseId --jq '.[0].databaseId')"
```

Expected:
- `resolve` succeeds (tag exists, matches regex).
- `bump` is skipped.
- `release`, `publish`, `merge` succeed with dry-run summaries.
- `release`'s summary shows the v0.24.1 changelog body.
- `publish`'s summary lists the wheels that would be uploaded.

- [ ] **Step 4: Negative test — invalid tag**

```bash
gh workflow run release-pipeline.yaml \
  --ref feat/release-pipeline \
  -f skip_bump=true \
  -f tag=v999.0.0 \
  -f dry_run=true
```

Expected: `resolve` job fails with `tag 'v999.0.0' does not exist on the remote`. Downstream jobs do not run.

- [ ] **Step 5: Open PR**

```bash
gh pr create --base main --head feat/release-pipeline \
  --title "ci: release-pipeline orchestrator + tighter test workflow" \
  --body "$(cat <<'EOF'
## Summary
- Replaces the fragile workflow_run release chain with a single dispatch-only `release-pipeline.yaml` that orchestrates bump → release → publish → merge as reusable workflows.
- Each reusable workflow has a `dry_run` input. Wheels are now built from the resolved tag, not branch HEAD.
- Adds partial-bump recovery to `bump.yaml`.
- Splits GH-release creation out of bump into `release.yaml`, sourced from `docs/source/changelog.md`.
- Adds free-threaded (3.14t) wheel builds across all platforms; switches PyPI upload to `uv publish`.
- Tests now run on push to `main`/`stable` and include a `cargo test` job.

Spec: `docs/superpowers/specs/2026-05-21-release-pipeline-design.md`
Plan: `docs/superpowers/plans/2026-05-21-release-pipeline.md`

## Test plan
- [x] Dry-run of full pipeline (fresh-bump path) on the feature branch
- [x] Dry-run of recovery path (`skip_bump=true, tag=v0.24.1`)
- [x] Negative test (invalid tag rejected by `resolve`)
- [ ] First real release after merge uses the new pipeline
EOF
)"
```

- [ ] **Step 6: Merge after review**

After PR approval, merge to `main`. The old workflows are now fully replaced.

- [ ] **Step 7: Post-merge sanity**

After merge, confirm:
```bash
gh workflow list | grep -E "Release pipeline|Bump version|Create release|Publish|Merge main"
```
Expected: all five workflows listed and `active`.

---

## Self-review notes

Coverage against spec:
- ✅ Orchestrator with `increment`/`skip_bump`/`tag`/`dry_run` → Task 5
- ✅ `bump.yaml` workflow_call + dry-run + partial recovery → Task 1
- ✅ `release.yaml` split + changelog awk → Task 2
- ✅ `publish.yaml` rewrite with tag-pinned checkout + uv publish + 3.14t → Task 4
- ✅ `merge.yaml` workflow_call + dry-run → Task 3
- ✅ `test.yaml` push triggers + cargo test → Task 6
- ✅ `lint.yaml` explicitly out of scope per spec
- ✅ Rollout/testing plan → Task 7

Type/name consistency: tag is always `vX.Y.Z` (with prefix) across orchestrator, bump output, release input, publish input. Secret is `COMMITIZEN` everywhere. Changelog path is `docs/source/changelog.md` in Task 2 and Task 1's partial-bump commit.
