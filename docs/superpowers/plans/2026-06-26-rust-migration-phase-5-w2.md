# Rust Migration Phase 5 — PR2 (W2): close out #242 with max_jitter>0 dataset-parity coverage

> **For agentic workers:** executed via superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** The #242 `intervals_to_tracks` store-vs-query divergence was already root-caused and FIXED end-to-end (kernel left-clip `s = max(itv.start - query_start, 0); e = min(end, length)` in both backends, merged via PR #244, ancestor of `rust-migration`; issue #242 CLOSED). The investigation (`.superpowers/sdd/w2-investigation.md`) showed the clip is functionally CORRECT, not merely masking. The ONLY residue is that the dataset-level parity suite still pins `max_jitter=0` with **stale** "PanicException landmine" comments, so numba-vs-rust byte-identity is not gated end-to-end over the jittered-track domain. This PR adds that coverage with a hand-computed oracle and de-stales the comments. **No kernel/write-path changes** (user decision: skip the unnecessary upstream coordinate rewrite).

**Branch:** `phase-5-w2`, stacked on `phase-5-w1` (so roadmap edits don't conflict with the open W1 PR #256).

## Global Constraints

- Byte-identical numba/rust parity is the gate. Test work only — do NOT touch `_intervals.py`, `src/intervals.rs`, the write path, or any kernel.
- The new dataset-parity case MUST be deterministic across backends: write with `max_jitter > 0` but READ at the default `jitter = 0` (a freshly opened dataset has `jitter=0`, `Deterministic: True`, even when `max_jitter>0`). Random read-jitter would desync the two backend reads — do not enable it.
- The case MUST genuinely exercise the #242 condition: assert that a stored interval start is strictly LESS than its query start (i.e. `regions.npy` expanded start `< input_regions.arrow` original chromStart) for the fixture, so the test is non-vacuous.
- Backend switching follows the established pattern in `tests/parity/test_dataset_parity.py`: `monkeypatch.setenv("GVL_BACKEND", "rust"|"numba")` then re-read.
- pytest commands MUST include `--basetemp=$(pwd)/.pytest_tmp` (os.link Errno 18 otherwise). Rust changes need `maturin develop --release` first — but this PR has NO rust changes.
- Conventional commits; co-author trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Empirically verified facts (from the W2 investigation probe)
- For region chromStart=100, max_jitter=4: `regions.npy[:, :3] = [[0, 96, 114]]`; `input_regions.arrow` chromStart = 100; default `ds.jitter = 0`.
- Track-only dataset, constant-5.0 BigWig over chr1:[0,1000), region chr1:100-110, max_jitter=4, jitter=0 read → both backends return `[5.]*10` byte-identically; deterministic across re-reads. Stored start 96 < query 100 (condition hit).

---

## Task 1: Add track-only max_jitter>0 dataset-parity + oracle test

**Files:**
- Modify: `tests/parity/_fixtures.py` — add a `build_track_dataset_jittered(work_dir, max_jitter)` builder: a track-only dataset with a CONTROLLED BigWig (deterministic, hand-computable signal) and `max_jitter > 0`. Reuse the existing `build_track_dataset` pattern but (a) take `max_jitter` and (b) use a BigWig whose signal over each region is exactly known (e.g. a constant value per contig, or a known piecewise-constant pattern) so the expected painted track is hand-computable.
- Modify: `tests/parity/test_dataset_parity.py` — add `test_tracks_max_jitter_intervals_parity_and_oracle`.

**Test requirements (the new test):**
- [ ] Build the jittered track-only dataset with `max_jitter = 4` (or similar > 0).
- [ ] **Non-vacuity / condition guard:** load `regions.npy` and `input_regions.arrow`; assert at least one stored region start (`regions.npy[:,1]`) is strictly `<` the corresponding original `chromStart` (proves the #242 sub-query condition is exercised). Assert `ds.jitter == 0` after open (deterministic read).
- [ ] Open `Dataset.open(ds_dir).with_tracks("signal")`. Read `ds[:, :]` under `GVL_BACKEND=rust`, then under `GVL_BACKEND=numba`.
- [ ] **Byte-identity:** `assert_array_equal` on both track `.data` (float32) and `.offsets` (int64) across backends.
- [ ] **Hand-computed oracle:** for each (region, sample), the expected track is the known BigWig signal over the ORIGINAL region window `[chromStart, chromEnd)` (jitter=0). Assert the rust output equals this oracle exactly. Keep the BigWig signal simple enough to compute in the test (e.g. constant per contig, or a single known interval covering each region).
- [ ] **Non-triviality:** assert some output value is non-zero (not a vacuous all-zero match).

- [ ] **Step 1 (TDD-ish):** Write the test. It PASSES on the current (fixed) tree — this is regression coverage for a previously-untested domain, not red→green. The non-vacuity guard (stored start < query start + correct nonzero oracle) is the evidence it would have caught the pre-fix bug (which over-padded/wrapped on exactly this condition).
- [ ] **Step 2:** Run: `pixi run -e dev pytest tests/parity/test_dataset_parity.py::test_tracks_max_jitter_intervals_parity_and_oracle -v --basetemp=$(pwd)/.pytest_tmp`. Expected PASS, both backends compared, oracle matched.
- [ ] **Step 3:** Commit.
  ```
  test(parity): cover max_jitter>0 intervals_to_tracks end-to-end (numba==rust + oracle, #242)
  ```

## Task 2: De-stale the landmine comments + roadmap + full verification

**Files:**
- Modify: `tests/parity/_fixtures.py` — fix the stale "PanicException landmine" docstrings on `build_haps_tracks_dataset` and `build_strand_mixed_dataset`. The `max_jitter=0` there is now retained ONLY because those fixtures compare `ds[:,:]` across backends and want the SIMPLEST deterministic geometry — NOT because of any panic (the kernel left-clip fixed #242, PR #244). Rewrite the comment to state the accurate reason and point to the new `test_tracks_max_jitter_intervals_parity_and_oracle` for the max_jitter>0 coverage. Do NOT change `max_jitter=0` in those builders (lifting them would desync nothing since jitter defaults to 0, but it would change output-length geometry and is out of scope — leave the values, fix only the comments).
- Modify: `tests/parity/test_dataset_parity.py` — fix the identical stale landmine comment block in `test_tracks_realign_getitem_identical_across_backends` (lines ~150-156).
- Modify: `docs/roadmaps/rust-migration.md` — add a dated Phase 5 W2 entry: #242 was already fixed (clip, PR #244) and is now end-to-end parity-covered at max_jitter>0 (new test); the stale landmine comments were corrected; #242 stays CLOSED; the upstream coordinate rewrite was intentionally skipped (clip is functionally correct per the W2 investigation). Phase 5 stays 🚧 (W3–W9 remain). Reference `.superpowers/sdd/w2-investigation.md`.

- [ ] **Step 1:** Rewrite the three stale comment blocks accurately (no "PanicException"/"landmine"/"violates the contract" language implying a live bug).
- [ ] **Step 2:** Add the roadmap W2 entry.
- [ ] **Step 3:** Full parity suite, both backends:
  - `pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
  - `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
  Expected: green, matching profiles.
- [ ] **Step 4:** Lint + typecheck: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`. (No rust → cargo not required, but harmless.)
- [ ] **Step 5:** Commit.
  ```
  docs(parity,roadmap): correct stale #242 landmine comments; record W2 closure
  ```

---

## Finish (controller, after final review + user confirm)
- Open PR `phase-5-w2` → base `phase-5-w1` (stacked) OR `rust-migration` if W1 has merged by then. No squash. Reference #242 (keep closed) + the W2 investigation.
