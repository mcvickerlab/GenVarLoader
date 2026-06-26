# Design: Rust Migration Phase 5 — Consolidation, numba deletion, rayon, final benchmark → main

**Date:** 2026-06-26
**Branch:** `rust-migration` (the persistent integration branch; pre-consolidation bug fixes land as their own PRs into it first)
**Roadmap:** `docs/roadmaps/rust-migration.md` — Phase 5 (⬜ → target ✅)
**Status:** design approved; spec for writing-plans

---

## 1. Context & goal

Phases 0–4 of the Rust migration are ✅: the read path (`Dataset.__getitem__`) and
write/update path are Rust-backed and rust-by-default, with byte-identical parity proven
against retained numba reference kernels. Those numba kernels were **deliberately kept
alive** as differential-test oracles, to be "deleted wholesale in Phase 5."

Phase 5 is the consolidation phase. Its roadmap checklist:

- Collapse the PyO3 surface so Python is a true shim.
- Delete all remaining core numba kernels (target count = 0).
- Confirm the crate is fully cargo-testable standalone.

**Goal of this work:** finish Phase 5, run a final numba-vs-rust benchmark on
`__getitem__` (wall-clock + peak RSS), and — if rust reaches parity or better — open the
`rust-migration → main` PR (the single big merge the branch strategy was built around).

### What is already satisfied

- **cargo-testable standalone:** `seqpro-core = "0.1.0"` is a published crates.io registry
  dependency (checksum-locked in `Cargo.lock`), not an editable path-dep. `cargo test`
  already runs without the Python/maturin layer (prior phases cite "cargo 109 passed").
  This checklist item needs only a final verification, not new work.

### Why this is not a no-op (the RSS gate)

All three hot read-path modules (`_genotypes.py`, `_flat_variants.py`, `_tracks.py`) still
`import numba as nb` at module load. The roadmap repeatedly records that peak RSS
(~3.53 GB) is "dominated by the numba/llvmlite JIT baseline (~3.2 GB)." Therefore the
rust-only peak-RSS win **cannot be measured until numba is deleted** — a benchmark today
would show near-parity RSS by construction (both backends import numba). The RSS metric
the user wants is gated on the numba deletion that is Phase 5's core.

---

## 2. Current state (measured 2026-06-26)

- `rust-migration` is **162 commits ahead of `main`, 0 behind, 123 files changed** — a
  clean fast-forward merge whenever chosen. `main` stays shippable.
- **~21 `register(...)` dual-backend kernels** across `_genotypes.py`, `_flat_variants.py`,
  `_intervals.py`, `_tracks.py`, `_reference.py`, all routed through the
  `python/genvarloader/_dispatch.py` registry (`GVL_BACKEND` override, per-kernel default
  `rust`).
- **~17 numba-oracle parity suites** in `tests/parity/` (e.g.
  `test_reconstruct_haplotypes_parity.py`, `test_fused_haps_parity.py`,
  `test_dataset_parity.py`) compare rust against the live numba impl.
- **Two known numba-vs-rust divergences are currently excluded from parity** (rust is
  correct in both; numba is the buggy oracle):
  1. **Haplotype trailing-fill** (`_genotypes.py:508`): when a deletion drives `ref_idx`
     past the contig end, `writable_ref = min(unfilled_length, len(ref) - ref_idx)` goes
     negative, so `out_end_idx = out_idx + writable_ref < out_idx`, and
     `out[out_end_idx:] = pad_char` uses Python-style negative indexing — it wraps and
     leaves trailing positions unwritten. Rust clamps `out_end_idx` to 0 and pads
     correctly. The same latent pattern exists at `_tracks.py:396`.
  2. **#242-family** (`intervals_to_tracks`): gvl stores intervals at
     `chromStart - max_jitter` but queries at `chromStart + jitter`, so for `max_jitter>0`
     datasets a stored interval can start before the query window. The numba/rust kernels
     diverge (debug_assert panic / clip behavior). Filed as
     [mcvickerlab/GenVarLoader#242](https://github.com/mcvickerlab/GenVarLoader/issues/242).
- **Deferred fusion:** the annotated+spliced *intersection* read path still runs on the
  unfused dispatched rust core (Phase 3 explicitly deferred its fusion to Phase 5).

---

## 3. Decisions (locked with the user)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Rayon batch parallelism | **In scope** for Phase 5 (the roadmap's "next lever"). |
| D2 | Fate of numba-oracle parity suites after deletion | **Golden-snapshot** them to frozen fixtures (preserve independent differential coverage in perpetuity), *after* fixing the numba bugs so the frozen oracle is correct. |
| D3 | PyO3 shim collapse aggressiveness | **Also fuse the deferred annotated+spliced path**, not just remove dispatch indirection. |
| D4 | Haplotype trailing-fill numba bug | **Fix it** (clamp), so the golden oracle is correct. |
| D5 | #242-family exclusion | **Fix it too**, so the golden oracle is fully exclusion-free (touches the write/store path; needs a correct-behavior investigation). |
| D6 | Final benchmark threading convention | **Single-thread verdict** (rayon=1 vs `NUMBA_NUM_THREADS=1`), comparable to all prior baselines; rayon multi-thread speedup reported separately as an additive bonus. |
| D7 | Bug fixes (D4, D5) PR strategy | **Separate PR(s), land first**, per the established numba-oracle-bug-policy (file issue + isolated fix + un-exclude from parity). |

---

## 4. Workstreams

### Stage A — Pre-consolidation correctness (separate PRs, land first)

These make numba a trustworthy, exclusion-free oracle **before** it is frozen as golden
fixtures and then deleted. Each uses systematic-debugging to establish the correct
behavior, and lands as its own PR into `rust-migration` (per D7).

**W1 — Fix the haplotype trailing-fill numba bug (D4).**
- File a GVL issue referencing the `_genotypes.py:508` trailing-fill divergence.
- Fix: `writable_ref = max(0, min(unfilled_length, len(ref) - ref_idx))` at
  `_genotypes.py:508`; mirror the clamp at `_tracks.py:396`.
- Verify rust already produces the correct (clamped/padded) output; confirm
  rust == numba after the fix across the previously-excluded overshoot sub-domain.
- Un-exclude that sub-domain: drop Guard 1 (the overshoot pre-check) in
  `tests/parity/test_reconstruct_haplotypes_parity.py`; remove the double-init sentinel
  guard where it only existed to mask this divergence.
- **Acceptance:** the overshoot sub-domain is parity-covered (not excluded), full tree
  green on both backends.

**W2 — Fix the #242-family divergence (D5).**
- Investigation (systematic-debugging): determine the correct `intervals_to_tracks`
  behavior when a stored interval starts before the query window (`max_jitter>0`),
  reconciling the `chromStart - max_jitter` store vs `chromStart + jitter` query offset.
  This may touch the write/store path and/or the query coordinate math, not only the
  kernel.
- Apply the fix to **both** backends so they agree and both are correct; reference/close
  #242.
- Un-exclude the #242-family sub-domain: remove the `assume(False)` / xfail guards in the
  affected parity + dataset suites (`test_reconstruct_haplotypes_parity.py`,
  `test_dataset_parity.py`, `test_shift_and_realign_tracks_parity.py`,
  `strategies.py`/`_fixtures.py` generators), lifting fixtures off the forced
  `max_jitter=0` where they were pinned only to dodge #242.
- **Acceptance:** `max_jitter>0` parity restored; #242 closed; full tree green on both
  backends.

### Stage B — Fusion (parity-gated against numba, before deletion)

**W3 — Fuse the deferred annotated+spliced intersection path (D3).**
- Add a fused rust kernel that collapses the remaining FFI crossings on the
  annotated+spliced read path (the intersection still on the unfused dispatched core),
  matching the fusion pattern of `reconstruct_annotated_haplotypes_fused` /
  `reconstruct_haplotypes_spliced_fused`.
- Gate on byte-identical parity against the composed numba oracle **while numba still
  exists**.
- **Acceptance:** annotated+spliced path is fused and byte-identical; parity suite extended
  to cover it.

### Stage C — Final numba-vs-rust benchmark (the gate; numba still present)

**W4 — Capture the single-thread parity verdict (D6).**
- Harness: existing `tests/benchmarks/test_e2e.py` (pytest-benchmark pedantic min) +
  `tests/benchmarks/profiling/profile.py` wall-clock, `NUMBA_NUM_THREADS=1`, rayon
  threads=1, release build, corpus `chr22_geuv.gvl` (format 2.0), Carter HPC.
- Run the numba-vs-rust A/B in **one back-to-back session** across all modes:
  tracks-only, tracks-seqs, haplotypes, annotated, variants, variant-windows.
- This is the canonical "final numba vs rust" wall-clock comparison; it must run while both
  backends exist (after deletion there is no numba to A/B).
- **Gate:** rust at **parity or better** (single-thread) on `__getitem__`. Per-path
  node-noise caveat applies (use within-session ratios; the durable signal is the
  established instruction-count reductions + parity).

### Stage D — Consolidation (the single big Phase 5 PR)

**W5 — Golden-snapshot the parity suites (D2).**
- Before deleting numba, generate frozen golden fixtures from the now-correct numba oracle
  for each of the ~17 parity suites (including the W3 fused path and the W1/W2
  un-excluded sub-domains).
- Convert the suites from "run-both-assert-byte-identical" to golden-file regression tests
  that need no live numba. Store fixtures compactly (compressed `.npz`/`.npy` keyed by the
  hypothesis-generated input, or a deterministic seeded sample set — chosen in the plan to
  keep the repo size bounded).
- **Acceptance:** golden suites pass against rust with numba uninstalled/uncalled.

**W6 — Delete numba + collapse to thin shim.**
- Delete the ~21 `register()` numba refs, all njit bodies, the `python/genvarloader/_dispatch.py`
  registry + `GVL_BACKEND`, and every `import numba` in the core modules.
- Replace `get(name)(...)` dispatch call sites (`_intervals.py`, `_reference.py`,
  `_reconstruct.py`, `_tracks.py`, `_flat_variants.py`, `_rag_variants.py`,
  `_genotypes.py`) with direct rust calls — Python becomes indexing sugar + torch +
  validation/error messages only.
- Remove `numba` from the project's runtime dependency set (verify nothing else in the
  package imports it).
- **Acceptance:** core numba kernel count = 0; `python -c "import genvarloader"` does not
  import numba or llvmlite (asserted by a test); full tree green.

**W7 — Add rayon batch parallelism (D1).**
- Parallelize the read-path batch drivers with rayon over the per-(query, hap) work items
  (disjoint output slices — proven safe / serial-equivalent in Phase 3). Rust-only;
  thread count controlled by an env/config knob, default chosen in the plan.
- **Acceptance:** byte-identical to the serial result (golden suites still pass);
  multi-thread speedup measured.

### Stage E — Measure & merge

**W8 — Rust-only RSS + rayon speedup.**
- After deletion, measure rust-only peak RSS on `__getitem__` (memray) vs the recorded
  numba baseline (3.53 GB) — expect the ~3.2 GB JIT removal.
- Measure rayon multi-thread speedup (rayon N vs rayon 1) as the additive bonus (D6).

**W9 — PR `rust-migration → main`.**
- If the Stage C verdict is parity-or-better and RSS is parity-or-better, open the merge
  PR (no squash — preserve commit history). Update `docs/roadmaps/rust-migration.md`:
  mark Phase 5 ✅, record the final single-thread A/B table, the rust-only RSS, the rayon
  speedup, and the PR link. Update `skills/genvarloader/SKILL.md` if any public symbol
  changed (e.g. removal of `GVL_BACKEND`).

---

## 5. Sequencing & PR strategy

```
W1 (haps trailing-fill fix)   ──┐  separate PRs into rust-migration
W2 (#242 fix)                 ──┘  (land first; un-exclude parity)
        │
W3 (annotated+spliced fusion) ───  PR into rust-migration (parity-gated vs numba)
        │
W4 (final numba-vs-rust A/B)  ───  benchmark only (both backends present) → GATE
        │
W5..W8 (golden snapshot, delete numba, rayon, RSS) ── single Phase 5 consolidation PR
        │
W9 (rust-migration → main)    ───  the big merge, if gate passes
```

Rationale for ordering: the numba bugs must be fixed (W1, W2) and the deferred path fused
(W3) **while numba still exists** as the oracle; the parity verdict (W4) must be captured
**before** deletion; only then is it safe to freeze golden fixtures (W5) and delete numba
(W6). Rayon (W7) is rust-only and lands after deletion. RSS (W8) is only meaningful after
deletion.

---

## 6. Out of scope

- **Phase 6 (absorb genoray):** variant IO stays on Python genoray.
- **Multi-thread numba (prange) A/B:** the verdict is single-thread per D6.
- Any further single-thread kernel micro-optimization (rounds 1–3 are complete; headroom
  is maximized per the roadmap).

---

## 7. Risks & mitigations

- **#242 is broader than a kernel clamp (W2).** It touches store-vs-query coordinate math;
  the correct behavior must be established by investigation before coding. Mitigation:
  systematic-debugging, fix both backends together, land as its own PR with the
  un-exclusion as the acceptance gate. If it proves larger than expected, it can be split
  out without blocking W1/W3.
- **Golden-fixture repo bloat (W5).** Frozen oracle outputs could be large. Mitigation:
  compress and/or use a bounded deterministic seeded sample rather than the full
  hypothesis space; decide the exact scheme in the plan.
- **Node-noise on the benchmark verdict (W4).** Carter is a shared node (absolute ms/batch
  drifts ≥2× across sessions). Mitigation: single back-to-back session, within-session
  ratios, pedantic min; lean on the durable instruction-count + parity evidence already in
  the roadmap.
- **Rayon non-determinism (W7).** Mitigation: disjoint output slices (already established);
  gate on byte-identical equality to the serial golden result.

---

## 8. Acceptance criteria (Phase 5 ✅)

1. Haplotype trailing-fill and #242 divergences fixed; both previously-excluded sub-domains
   parity-covered (W1, W2).
2. Annotated+spliced path fused, byte-identical (W3).
3. Final single-thread numba-vs-rust `__getitem__` A/B captured; rust at parity-or-better
   (W4).
4. Parity suites converted to golden fixtures; pass with numba absent (W5).
5. Core numba kernel count = 0; `import genvarloader` pulls neither numba nor llvmlite;
   `_dispatch`/`GVL_BACKEND` gone; PyO3 surface is a thin shim (W6).
6. Rayon batch parallelism byte-identical to serial; speedup measured (W7).
7. Rust-only peak RSS at parity-or-better vs the 3.53 GB numba baseline (W8).
8. `cargo test` green standalone; full Python tree green; lint/format/typecheck clean;
   abi3 wheel builds.
9. `rust-migration → main` PR opened (no squash); roadmap Phase 5 ✅ + final numbers + PR
   link recorded; skill updated if public API changed (W9).
