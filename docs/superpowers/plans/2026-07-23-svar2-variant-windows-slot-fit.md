# SVAR2 variant-windows slot-fit (#315 Layer 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Dataset.to_dataloader(mode="double_buffered")` over flat `variant-windows` work on the real SVAR2-backed Hartwig corpus by restoring `_output_bytes_per_instance` as a provable upper bound on the `Svar2Haps` reconstruction path.

**Architecture:** The double-buffered producer packs instances into a fixed shared-memory slot sized from `_output_bytes_per_instance`. That estimate's `variant-windows` branch (`_impl.py:1564`) derives its per-instance variant count and window spans from the `Haps` quantities (`self.n_variants` = `genotypes.lengths`; `genotypes[r,s].to_packed()`). For an **SVAR2-format-2.0.0** dataset the reconstruction runs through `Svar2Haps._call_svar2` at `p_eff = 1 if unphased_union else P`, and the estimate no longer matches what the writer emits — it under-counts so severely that the whole corpus packs into one undersized slot and overflows. We (1) pin the exact diverging term against the real corpus, (2) lock it with a failing in-tree test on an SVAR2 dataset, (3) correct the estimate to mirror the `Svar2Haps` count, (4) verify the reported config end-to-end.

**Tech Stack:** Python 3.12, NumPy, Rust extension (maturin/PyO3), genoray SparseVar2, pixi (`dev` env), pytest, cargo.

## Global Constraints

- **This is a debug-gated plan.** Task 1 (Phase 0 spike) pins the diverging term; Task 3's fix code is confirmed against that pin, not guessed — the predecessor rule "Layer 1 does not proceed on a guess" holds. Tasks are **sequential**: 1 → 2 (red) → 3 (green) → 4. There is no safe parallelism (the fix depends on the pin; the test's record class depends on the pin).
- **Rebuild the Rust extension before any pytest that imports it:** `pixi run -e dev maturin develop --release` (run from the worktree root). The slot-fit path calls into the compiled window/svar2 kernels.
- **Run tests via** `pixi run -e dev pytest ...`. Do **not** use `--frozen` here — this repo's `dev` env is not shared with a co-tenant (that constraint was aster-specific).
- **Real Hartwig corpus (read-only), symlink into the worktree before Task 1:**
  `data/corpus/hartwig` → `/carter/users/dlaub/projects/aster/data/corpus/hartwig`, and the reference assets under `refs/` if the corpus's reference is needed (the corpus is trackless; variant-windows `ref="window"` reads the reference genome, so `refs/GRCh38.ensembl.fa.bgz{,.fai,.gzi,.gvlfa}` must resolve — symlink from `/carter/users/dlaub/projects/aster/refs/`).
- **Commit hooks:** the `pyrefly-check` pre-commit hook type-checks the whole tree (`pixi run -e dev pyrefly check python/genvarloader tests`, ~7+ min) with no file filter. For **docs-only** commits use `git commit --no-verify`. For **code** commits, let the hook run (it is the type-check gate); if iterating, run `pixi run -e dev pyrefly check python/genvarloader tests` manually once and keep commits batched.
- **Issue/scope:** [#315](https://github.com/mcvickerlab/GenVarLoader/issues/315). Out of scope: `buffered`/`manual` modes, Bug A (`realign_tracks`, separate issue), producer grow-or-split (escalation only). Spec: `docs/superpowers/specs/2026-07-23-svar2-variant-windows-slot-fit-design.md`.

---

## File Structure

- `python/genvarloader/_dataset/_impl.py` — `Dataset._output_bytes_per_instance`, `variant-windows` branch at **1564-1669**. The only production file Task 3 edits.
- `python/genvarloader/_dataset/_svar2_haps.py` — `Svar2Haps(Haps)`; `self.n_variants = self.genotypes.lengths` (208); window/allele reconstruction at `p_eff = 1 if unphased_union else P` (821, 964, 1049…). Task 3 reuses a counting entry point from here; edits only if none can be reused (small extract).
- `tests/unit/test_slot_fit_property.py` — the slot-fit invariant harness (`_views`, `_assert_upper_bound`). Task 2 adds SVAR2 coverage.
- `tests/conftest.py` — fixtures; `phased_svar_gvl` at 127. Task 2 adds a `phased_svar2_gvl` fixture (or a real-slice fixture per the Phase-0 decision).
- `tests/test_svar2_reconstruct.py:34` (`svar2_store`) — reference recipe for building a `.svar2` store from genoray for the synthetic fixture.
- `docs/superpowers/specs/2026-07-23-phase0-realcorpus-findings.md` — **created by Task 1** (the pin).
- `docs/superpowers/specs/2026-07-21-phase0-findings.md`, `…-double-buffered-vw-slot-fit-design.md` — predecessor docs Task 4 corrects/updates.
- Scratch (not committed to `python/`): `scratch/diag_315_realcorpus.py` under the worktree — Task 1's instrumentation.

---

### Task 1: Phase 0 — pin the divergence on the real corpus (spike)

This is an **investigation spike**, not TDD. Its deliverable is a committed findings doc that pins which quantity the estimate mis-counts, captured from the real corpus. Task 3's fix is written against this pin.

**Files:**
- Create (scratch, not committed under `python/`): `scratch/diag_315_realcorpus.py`
- Create (committed): `docs/superpowers/specs/2026-07-23-phase0-realcorpus-findings.md`

**Interfaces:**
- Consumes: `gvl.Dataset.open`, `Dataset._output_bytes_per_instance`, `Dataset.__getitem__` (`view[r,s]`), `genvarloader._shm_layout.write_chunk`, `HEADER_RESERVED`, `genvarloader._slot_overhead.slot_overhead_bytes`, `Svar2Haps`.
- Produces (consumed by Tasks 2/3): the pinned diverging term (one of: `n_vars_total`/`self.n_variants` count, `ref_span`, `alt_alleles`, or a `p_eff`/ploidy grouping error), a captured overflowing `(r_idx, s_idx)`, the variant record class driving it, and whether the fixture reproducing it can be synthetic or must be a real slice.

- [ ] **Step 1: Symlink the real corpus + reference into the worktree**

```bash
cd ~/projects/GenVarLoader/.claude/worktrees/315-svar2-slot
mkdir -p data/corpus refs
ln -sfn /carter/users/dlaub/projects/aster/data/corpus/hartwig data/corpus/hartwig
for f in GRCh38.ensembl.fa.bgz GRCh38.ensembl.fa.bgz.fai GRCh38.ensembl.fa.bgz.gzi GRCh38.ensembl.fa.bgz.gvlfa; do
  ln -sfn /carter/users/dlaub/projects/aster/refs/$f refs/$f
done
ls -l data/corpus/hartwig refs/
```
Expected: symlinks resolve (corpus `hartwig.gvl/` + `tumor.xolars/` visible; reference files listed).

- [ ] **Step 2: Build the Rust extension**

Run: `pixi run -e dev maturin develop --release`
Expected: builds and installs the extension into the `dev` env without error.

- [ ] **Step 3: Write the instrumentation script**

Create `scratch/diag_315_realcorpus.py`. It reproduces the reported config, confirms the reconstructor is `Svar2Haps`, and — for a sample of instances and for at least one instance from an overflowing chunk — compares the estimate's internal quantities against the real serialized payload. `main()` MUST sit under `if __name__ == "__main__":` (the producer spawns and re-imports).

```python
"""Pin the #315 estimate divergence against the real SVAR2 Hartwig corpus.
Not committed under python/. Run: pixi run -e dev python scratch/diag_315_realcorpus.py
"""
import sys
import numpy as np
import seqpro as sp
import genvarloader as gvl
from genvarloader._dataset._svar2_haps import Svar2Haps
from genvarloader._shm_layout import HEADER_RESERVED, write_chunk
from genvarloader._slot_overhead import slot_overhead_bytes

CORPUS = "data/corpus/hartwig/hartwig.gvl"
REF = "refs/GRCh38.ensembl.fa.bgz"
N_REGIONS = 40


def make_view():
    DNA = sp.alphabets.DNA
    ds = gvl.Dataset.open(CORPUS, reference=REF)
    # subset to the reported region count (all samples)
    ds = ds.subset_to(regions=slice(N_REGIONS))
    opt = gvl.VarWindowOpt(
        flank_length=128, token_alphabet=DNA, unknown_token=len(DNA),
        ref="window", alt="allele",
    )
    return (
        ds.with_tracks(False).with_output_format("flat")
          .with_seqs("variant-windows", opt)
          .with_settings(unphased_union=True, jitter=0)
    )


def main() -> int:
    view = make_view()
    print("reconstructor:", type(view._seqs).__name__,
          "is Svar2Haps:", isinstance(view._seqs, Svar2Haps))
    R, S = view.shape[:2]
    print(f"shape: {R} regions x {S} samples")

    # Per-instance estimate vs real, over a growing instance count, to test whether
    # the under-count is per-instance (grows with N) or a per-chunk constant.
    rng = np.random.default_rng(0)
    for N in (S, 4 * S, 16 * S):  # 1, 4, 16 regions worth of instances
        n_reg = N // S
        rr, ss = np.meshgrid(np.arange(n_reg), np.arange(S), indexing="ij")
        r, s = rr.reshape(-1), ss.reshape(-1)
        chunk = view[r, s]
        arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
        buf = memoryview(bytearray(512 * 1024 * 1024))
        real = write_chunk(buf, arrays, n_instances=len(r)) - HEADER_RESERVED
        est = int(np.asarray(
            view._output_bytes_per_instance(r, s, include_offsets=True)).sum())
        ovh = slot_overhead_bytes(view)
        print(f"N={len(r):>7}  est={est:>12}  overhead={ovh:>8}  real={real:>12}  "
              f"est+ovh-real={est + ovh - real:>12}  per_inst_gap={(real - est) / len(r):.1f}")

    # Decompose one region's estimate: n_vars_total vs emitted window count W.
    r = np.zeros(S, np.int64)
    s = np.arange(S, np.int64)
    haps = view._seqs
    n_vars = view.n_variants(r, s)
    n_vars_total = n_vars.reshape(-1, n_vars.shape[-1]).astype(np.int64).sum(-1)
    chunk = view[r, s]
    ref_slot = (chunk[0] if isinstance(chunk, tuple) else chunk)
    # emitted window count W per instance = len(ref window seq_offsets) - 1, per instance
    print("sum n_vars_total (estimate M):", int(n_vars_total.sum()))
    print("real_ploidy:", haps.genotypes.shape[-2],
          "unphased_union:", view.unphased_union)
    # Dump the worst-under-counted instance for its record class.
    est_pi = np.asarray(view._output_bytes_per_instance(r, s, include_offsets=True))
    worst = int(np.argmin(est_pi - 0))  # smallest estimate; refine vs per-instance real if needed
    print("example instance (r=0, s=%d): est_bytes=%d n_vars_total=%d"
          % (worst, int(est_pi[worst]), int(n_vars_total[worst])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the instrumentation and capture the pin**

Run: `pixi run -e dev python scratch/diag_315_realcorpus.py 2>&1 | tee scratch/diag_315_out.txt`
Expected observations to record: `is Svar2Haps: True`; `est+ovh-real` goes **negative** (the overflow), and `per_inst_gap` stays roughly constant as N grows (⇒ per-instance under-count, not per-chunk). Compare `sum n_vars_total (estimate M)` against the emitted window count — a mismatch (M ≠ W) localizes the divergence to the variant-count path; if M == W but bytes still under-count, the divergence is in `ref_span`/`alt_alleles`. If the per-instance decomposition needs the emitted `W`, extend the script to read `ref_slot.seq_offsets`/`var_offsets` per instance (the `_FlatVariantWindows` layout) and diff against `n_vars_total`.

- [ ] **Step 5: Write the findings doc (the pin)**

Create `docs/superpowers/specs/2026-07-23-phase0-realcorpus-findings.md` recording: confirmation the corpus opens as `Svar2Haps`; the `est/overhead/real` table across N with the sign of the gap and the per-instance slope; **the pinned diverging term** (M-vs-W count, or `ref_span`, or `alt_alleles`, or `p_eff`/ploidy grouping) with the numbers that localize it; the captured `(r_idx, s_idx)` and its variant record class (dense-vs-vk range, `ilen`, ALT, genotype/`unphased_union` pattern); and the **fixture decision** — whether a synthetic `.svar2` store can reproduce the pinned class (preferred) or a real-corpus slice is required (with size/policy note). End with an explicit statement of what Task 3 must change and what Task 2's fixture must contain.

- [ ] **Step 6: Commit the findings doc (docs-only)**

```bash
git add docs/superpowers/specs/2026-07-23-phase0-realcorpus-findings.md
git commit --no-verify -m "docs(spec): #315 Phase 0 on real corpus — pin the SVAR2 estimate divergence"
```
(Scratch under `scratch/` is left uncommitted; confirm `scratch/` is gitignored or add it to `.git/info/exclude`.)

---

### Task 2: Failing SVAR2 slot-fit test (TDD red — lock the bug in-tree)

Add SVAR2 coverage to the slot-fit property harness so #315 is reproduced by the test suite itself, and confirm it **fails** before the fix. Fixture source follows Task 1's decision.

**Files:**
- Modify: `tests/conftest.py` (add `phased_svar2_gvl` fixture)
- Modify: `tests/unit/test_slot_fit_property.py:58-63` (add the SVAR2 case)

**Interfaces:**
- Consumes: Task 1's fixture decision + pinned record class; existing `_views(ds)` and `_assert_upper_bound(view)` in `test_slot_fit_property.py`; `gvl.write` with a `.svar2` source; the `svar2_store` recipe at `tests/test_svar2_reconstruct.py:34`; `reference` fixture.
- Produces (consumed by Task 3): a red test `test_slot_fit_svar2_backend` that goes green exactly when the estimate becomes an upper bound on the `Svar2Haps` path.

- [ ] **Step 1: Add the SVAR2 dataset fixture**

In `tests/conftest.py`, add a fixture that produces an **SVAR2-format** gvl dataset (opens as `Svar2Haps`) reproducing the Phase-0 record class. Preferred synthetic form (adapt the `svar2_store` recipe at `tests/test_svar2_reconstruct.py:34` to build a `.svar2` store, then `gvl.write` from it so `metadata["svar2_link"]` is set):

```python
@pytest.fixture(scope="session")
def phased_svar2_gvl(tmp_path_factory, reference, synthetic_case) -> Path:
    """A gvl dataset written from a .svar2 source -> opens as Svar2Haps.
    Reproduces the Phase-0-pinned record class (see
    docs/superpowers/specs/2026-07-23-phase0-realcorpus-findings.md).
    """
    # Build a .svar2 store from the shared synthetic case (see
    # tests/test_svar2_reconstruct.py::svar2_store for the genoray recipe),
    # then write a gvl dataset from that .svar2 path.
    svar2_path = ...  # per svar2_store recipe, including the pinned record class
    out = tmp_path_factory.mktemp("svar2_gvl") / "ds.gvl"
    gvl.write(out, variants=svar2_path, bed=..., reference=reference)
    return out
```

> If Task 1 decided a synthetic store cannot reproduce the pin, instead add a fixture returning a **tiny committed real-corpus slice** (bounded regions×samples) as a test asset under `tests/data/`, and note the data-policy check in the findings doc. If neither is possible, add the SVAR2 case as `@pytest.mark.xfail(strict=True, reason="#315 SVAR2 coverage gap — see findings")` and document the gap; the plan then relies on Task 4's real-corpus e2e as the acceptance gate.

- [ ] **Step 2: Add the SVAR2 slot-fit test**

In `tests/unit/test_slot_fit_property.py`, after `test_slot_fit_file_backends`:

```python
def test_slot_fit_svar2_backend(phased_svar2_gvl, reference):
    """SVAR2 datasets open as Svar2Haps via the released reconstruct path — the
    coverage gap that let #315 through. The estimate must upper-bound the real
    serialized payload here too."""
    from genvarloader._dataset._svar2_haps import Svar2Haps
    ds = gvl.Dataset.open(phased_svar2_gvl, reference=reference)
    assert isinstance(ds._seqs, Svar2Haps), "fixture must open as Svar2Haps"
    for view in _views(ds):
        _assert_upper_bound(view)
```

- [ ] **Step 3: Build the extension and run the new test — expect FAIL**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/unit/test_slot_fit_property.py::test_slot_fit_svar2_backend -v
```
Expected: **FAIL** with `AssertionError: slot under-sized: est=... overhead=... real=...` (est+overhead < real) — the in-tree reproduction of #315. If it PASSES, the fixture does not reproduce the pinned class; return to Task 1 (the synthetic fixture is insufficient → use the real-slice path).

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/conftest.py tests/unit/test_slot_fit_property.py
pixi run -e dev pyrefly check python/genvarloader tests   # hook parity; must pass
git commit -m "test(dataloader): #315 SVAR2 slot-fit reproduction (red)"
```
Expected: commit succeeds (the test file is red but committed intentionally as the reproduction; the pre-commit type-check still passes).

---

### Task 3: Layer 1 — make the estimate an upper bound on the Svar2Haps path (TDD green)

Correct the `variant-windows` branch of `_output_bytes_per_instance` so its per-instance count/spans mirror what `Svar2Haps._call_svar2` emits, per Task 1's pin. Acceptance = Task 2's test flips to green with no regression elsewhere.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:1564-1669` (`variant-windows` branch)
- Modify (only if no reusable count exists): `python/genvarloader/_dataset/_svar2_haps.py` (extract a per-instance emitted-window-count entry point)

**Interfaces:**
- Consumes: Task 1's pinned term; `Svar2Haps` counting (`self.n_variants`, `p_eff`, the `_call_svar2` window kernel's emitted count); Task 2's red test.
- Produces: `test_slot_fit_svar2_backend` green; `est + slot_overhead ≥ real` holds on the `Svar2Haps` path.

- [ ] **Step 1: Apply the pinned correction**

Edit the `variant-windows` branch (`_impl.py:1564`). The correction shape depends on Task 1's pin; write the one the findings doc specifies. The most-likely pin (confirm against findings before applying) is that the estimate's variant count / window span must be taken from the **same quantity `Svar2Haps` emits at `p_eff`**, not the `Haps` `genotypes[r,s].to_packed()` path. Prefer **reusing** a `Svar2Haps` counting method over duplicating its `p_eff` logic (duplication is what drifted). Concretely, when `isinstance(haps_obj, Svar2Haps)`, derive `n_vars_total`/`ref_span`/`alt_alleles` from the Svar2Haps-emitted per-instance window count `W` (add a small method on `Svar2Haps` returning emitted windows per instance if none exists), so `estimate ≥ real`.

> **If Task 1 pins a term this shape does not cover** (e.g. an allele-byte backend difference, or a genuinely post-reconstruction-only quantity), implement the findings doc's specified correction instead. **If the findings doc invoked the escalation branch** (estimate cannot be made a cheap upper bound), STOP and return to the user for the grow-or-split scope decision — do not force an estimate hack.

- [ ] **Step 2: Build the extension and run the SVAR2 test — expect PASS**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/unit/test_slot_fit_property.py::test_slot_fit_svar2_backend -v
```
Expected: **PASS**.

- [ ] **Step 3: Run the whole slot-fit + variants/svar2 suites — expect no regression**

Run:
```bash
pixi run -e dev pytest tests/unit/test_slot_fit_property.py tests/test_svar2_reconstruct.py tests/unit/test_producer.py tests/unit/test_producer_schema.py tests/unit/test_shm_layout.py -v
```
Expected: all PASS (the dummy/vcf/pgen/svar Haps cases still hold — the fix must be gated on `Svar2Haps` and not change the `Haps` estimate).

- [ ] **Step 4: Commit the fix**

```bash
git add python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_svar2_haps.py
pixi run -e dev pyrefly check python/genvarloader tests
git commit -m "fix(dataloader): #315 estimate upper-bounds Svar2Haps variant-windows payload"
```

---

### Task 4: End-to-end verification on the real corpus + report

Confirm the reported config now *works* on the real corpus, run the full tree + cargo, update predecessor docs, and post the resolution to #315.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-21-double-buffered-vw-slot-fit-design.md` (status table: Layer 1 → done)
- Modify: `docs/superpowers/specs/2026-07-21-phase0-findings.md` (correct the "SVAR2 unreachable / every backend opens as Haps" claim; point to the real-corpus findings)

**Interfaces:**
- Consumes: the shipped fix; the real corpus symlink from Task 1.
- Produces: a green e2e run of the exact #315 repro; updated docs; an issue update.

- [ ] **Step 1: Re-run the exact reported repro on the real corpus — expect it to complete**

Create `scratch/repro_315_e2e.py` (guarded `__main__`) that opens the real corpus, builds the reported view (40 regions × all samples, flat variant-windows, `ref="window"`, `alt="allele"`, `unphased_union=True`, `jitter=0`, `flank_length=128`), and iterates a few batches of `to_dataloader(batch_size=4096, mode="double_buffered")` with the default `buffer_bytes`.

Run: `pixi run -e dev python scratch/repro_315_e2e.py`
Expected: yields batches; **no** `SlotOverflowError` / `ProducerError`. (If it raises `SlotOverflowError` naming a *specific* oversized instance, the estimate is still short for that class — return to Task 1 with that instance.)

- [ ] **Step 2: Full test tree + Rust**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q
cargo test
```
Expected: full suite green; `cargo test` green.

- [ ] **Step 3: Update predecessor docs**

Edit the two predecessor specs: in `…-double-buffered-vw-slot-fit-design.md` flip the status table's Layer 1 row to ✅ done with a pointer to `2026-07-23-svar2-variant-windows-slot-fit-design.md` and the real-corpus findings; in `2026-07-21-phase0-findings.md` add a correction note that SVAR2-format-2.0.0 datasets open as `Svar2Haps` on the released `to_dataloader` path (contradicting the earlier "SVAR2 unreachable" conclusion), with the pin.

- [ ] **Step 4: Commit docs + push branch**

```bash
git add docs/superpowers/specs/2026-07-21-double-buffered-vw-slot-fit-design.md docs/superpowers/specs/2026-07-21-phase0-findings.md
git commit --no-verify -m "docs(spec): #315 Layer 1 done — SVAR2 estimate fix; correct Phase 0 reachability claim"
git push -u origin fix/315-svar2-slot
```

- [ ] **Step 5: Post the resolution to #315**

Comment on #315 with: corrected repro params (SVAR2-backed corpus, `flank_length=128`, default 2 GiB `buffer_bytes`, 40×7089), the confirmed root cause (estimate used the `Haps` count; SVAR2 reconstructs via `Svar2Haps` at `p_eff`), the fix, and that the slot-fit property test now covers the SVAR2 path. (Draft in `scratch/`; post via `gh issue comment 315` after user review.)

---

## Self-Review

**Spec coverage:**
- Phase 0-real (pin) → Task 1. ✅
- Layer 1 (estimate upper bound on Svar2Haps path) → Task 3. ✅
- Layer 2b-svar2 (property test covers SVAR2 via released path) → Task 2. ✅
- Verify reported config works + full pytest/cargo + docs + #315 → Task 4. ✅
- Escalation branch (grow-or-split) → Task 3 Step 1 stop-and-return-to-user guard. ✅
- Fixture-source decision deferred to Phase 0 → Task 1 Step 5 + Task 2 Step 1 ladder. ✅
- Non-goals (buffered/manual, Bug A, slot_overhead) → untouched; not in any task. ✅

**Placeholder scan:** Task 2/3 intentionally carry a *pin-dependent* branch — this is a debug-gated plan, so the exact fix line and the exact fixture record class are outputs of Task 1, not omissions. Each such point names the precise file/function/location, the reuse target, and an executable pass/fail condition (red→green test, e2e no-overflow), plus an explicit escalation stop. The `...` in the fixture code marks the Phase-0-pinned record class and the `svar2_store` recipe to copy, both cross-referenced — not vague "implement later" work.

**Type consistency:** `_output_bytes_per_instance(r, s, include_offsets=True)`, `write_chunk(buf, arrays, n_instances=...) - HEADER_RESERVED`, `slot_overhead_bytes(view)`, `_views(ds)`/`_assert_upper_bound(view)`, and `Svar2Haps`/`gvl.Dataset.open(path, reference=...)` are used identically across tasks and match the source read at plan time (`test_slot_fit_property.py`, `_impl.py:1339/1564`, `_svar2_haps.py:182/208`).
