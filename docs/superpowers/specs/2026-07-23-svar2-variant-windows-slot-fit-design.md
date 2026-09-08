# SVAR2 variant-windows slot-fit — #315 Layer 1 (the real-corpus unblock)

**Date:** 2026-07-23
**Status:** Approved design — implementation pending.
**Issue:** [#315](https://github.com/mcvickerlab/GenVarLoader/issues/315)
**Branch/target:** `fix/315-svar2-slot` → `main`
**Predecessors (already on `main`):**
- Design: [`2026-07-21-double-buffered-vw-slot-fit-design.md`](./2026-07-21-double-buffered-vw-slot-fit-design.md) (Layers 2a/2b/2c shipped, **Layer 1 deferred**)
- Phase 0: [`2026-07-21-phase0-findings.md`](./2026-07-21-phase0-findings.md) (divergence declared unreproducible synthetically)

This spec resolves the **deferred Layer 1** of the predecessor design, using the real
Hartwig corpus that was unavailable in the earlier session.

## What changed since the predecessor design

The predecessor shipped the mechanism hardening (schema-derived `slot_overhead`, a
`SlotOverflowError` write-time guard, and an `est + slot_overhead ≥ real` property test)
but **deferred Layer 1** — the actual estimate correction — because it could not
reproduce the real-corpus divergence. Phase 0 concluded (findings §"Why the design doc's
hypothesized mechanism does not apply here"):

> every backend — including a SparseVar source — opens as `Haps` … SVAR2 is
> StreamingDataset-only (out of scope).

**That conclusion is incorrect for SVAR2-format (`format_version 2.0.0`) datasets.** It
was reached by testing only VCF / PGEN / old-format-SVAR fixtures, all of which *do* open
as `Haps`. The real Hartwig corpus is SVAR2-backed and is now available, which both
falsifies the assumption and supplies the missing fixture.

### Evidence

1. **The corpus is SVAR2-backed.** `data/corpus/hartwig/hartwig.gvl/metadata.json` has
   `svar2_link` set (`svar_link: null`), `format_version 2.0.0`, ploidy 2, and a
   `genotypes/svar2_ranges/` genotype store (`{dense,vk}_{snp,indel}_range.npy`,
   `svar2_meta.json`).

2. **`Svar2Haps` IS reachable from the released `to_dataloader` path.**
   `_dataset/_reconstruct.py:149` — `if isinstance(self.haps, Svar2Haps): return
   self._call_svar2(...)`; `_dataset/_impl.py:350` branches on `isinstance(self._seqs,
   Svar2Haps)` too. So `Dataset.open(...).to_dataloader(mode="double_buffered")` on this
   corpus reconstructs through `_call_svar2`, **not** the `Haps` path Phase 0 measured.

3. **The estimate under-counts massively on this corpus (the #315 overflow).** Re-running
   the exact reported config (40 regions × 7089 samples, flat `variant-windows`,
   `ref="window"`, `alt="allele"`, `unphased_union=True`, `jitter=0`, tracks OFF,
   `flank_length=128`, default `buffer_bytes=2 GiB`, `batch_size=4096`) against
   genvarloader **0.40.1** (≈ current `main`; includes the 2a `SlotOverflowError`):

   ```
   RuntimeError: ProducerError (SlotOverflowError): serialized chunk (n_instances=283560)
   does not fit the shared-memory slot of 18156032 bytes. The per-instance byte estimate
   under-sized this slot.
   ```

   **283560 = 40 × 7089** — i.e. the estimate packed the *entire* subset into **one**
   ~17 MB slot (~64 B/instance). At `flank_length=128` each emitted window is ≈257 ref
   tokens plus allele + offset bytes, so the true payload is tens of MB. The estimate is
   accounting ≈0 variant-data bytes per instance. This is a large **per-instance**
   under-count, not the per-chunk constant Phase 0 found on the `Haps` path.

4. **Phase 0 already flagged the mechanism.** Findings §"Why … does not apply here" notes
   `Svar2Haps` has separate variant-counting logic (`_svar2_haps.py`, `p_eff = 1 if
   unphased_union else P`) — it simply assumed that path was unreachable. It is reachable;
   that separate counting is the prime suspect for the divergence.

## Root-cause hypothesis (to be pinned, not assumed)

`Dataset._output_bytes_per_instance`'s `variant-windows` branch derives its per-instance
window count and allele bytes from the **`Haps`** quantities (`n_variants =
genotypes.lengths`; `to_packed().data`), for which Phase 0 proved the identity
`M == K == W`. On an SVAR2 dataset the reconstruction runs through `Svar2Haps` /
`_call_svar2`, whose emitted window count and/or `unphased_union` `p_eff` handling differ
from the `Haps` quantities the estimate uses — so `M` (estimate) < `W` (emitted), and the
estimate is no longer an upper bound. **The exact diverging term (window count vs `p_eff`
vs allele bytes) is pinned by Phase-0-on-real-corpus before any fix is written** — the
predecessor plan's rule "Layer 1 does not proceed on a guess" is retained.

## Goals / non-goals

**Goals**
- Make `double_buffered` variant-windows **work** on the real SVAR2 Hartwig corpus at the
  reported config — the reported command completes instead of raising `SlotOverflowError`.
- Restore `estimate + slot_overhead ≥ real serialized bytes` as a **provable upper bound
  on the SVAR2 read path**, defined operationally by the extended property test passing.
- Close the coverage gap **by test**: the slot-fit property test must exercise an SVAR2
  dataset through the released `to_dataloader`/reconstruct path, so this class of drift is
  caught in CI.

**Non-goals**
- Changes to `buffered` / `manual` modes (they don't serialize into fixed slots).
- Bug A (`realign_tracks` propagation) — remains its own issue + PR.
- The producer-side grow-or-split auto-recovery — still deferred; only the escalation
  branch if Layer 1 proves intractable (below).
- Re-deriving `slot_overhead` (Layer 2c) — already shipped and correct; unchanged.

## Design

### Phase 0-real — Pin the divergence (gate)

In the worktree, with the real corpus symlinked in (as done for the aster env) and Rust
built (`maturin develop --release`), run an instrumentation script (adapt the predecessor's
referenced `diag_315_realcorpus.py`; it is scratch, not committed) against the Hartwig
dataset at the failing config. For a sample of instances — and for at least one instance
from a chunk that overflows — capture and compare:

- `M` = the estimate's per-instance window/variant count (`_output_bytes_per_instance`
  internals) and its byte breakdown (ref_span, allele bytes, flank, offsets);
- `W` = the window count actually emitted by `_call_svar2` / the Rust svar2 window kernel
  (`len(seq_offsets) − 1` of the produced ref slot), and the real serialized payload from a
  direct `write_chunk`;
- the reconstructor class actually in use (assert `Svar2Haps`);
- `d(real − est)/d(inst)` across instance counts, to confirm the under-count is
  per-instance (expected from Evidence §3), not the per-chunk constant Phase 0 saw.

**Output:** the offending `(r_idx, s_idx)`, the diverging term, the variant record class
(ilen / ALT / genotype pattern / dense-vs-vk range) driving it, and whether it is
`unphased_union`-dependent. This dump is the recipe for the Layer-1 fix and the regression
fixture. Layer 1 does not start until this pins the term.

### Layer 1 — Make the estimate an upper bound on the SVAR2 path

Correct the `variant-windows` branch of `Dataset._output_bytes_per_instance`
(`_dataset/_impl.py`) so that, when the dataset reconstructs via `Svar2Haps`, its
per-instance window count / allele-byte accounting is derived from the **same quantity
`_call_svar2` emits** (mirroring `Svar2Haps`' counting, incl. its `p_eff = 1 if
unphased_union else P` rule), making `M ≥ W` and the estimate a provable upper bound for
that record class and backend. Prefer reusing `Svar2Haps`' own counting entry point over
duplicating its logic, so the estimate and reconstructor cannot drift again. "Correct" is
defined operationally: the Layer-2b-extended property test (below) passes on a fixture
reproducing the pinned record class.

### Layer 2b-svar2 — Extend the upper-bound property test (the durable fix)

Extend `tests/unit/test_slot_fit_property.py` to add an **SVAR2 dataset opened through the
released `Dataset.open` + `to_dataloader`/reconstruct path** (not the synthetic `Haps`
matrix it covers today) and assert, for every chunk,

```
sum(_output_bytes_per_instance(include_offsets=True)) + slot_overhead(schema)
    ≥ real write_chunk cursor − HEADER_RESERVED
```

across `ref`/`alt` ∈ {window, allele}, `unphased_union` on/off, `flank_length` ∈ {small,
128}, and the Phase-0-pinned record class. This is the coverage gap that let #315 through.

**Fixture source — decided by Phase 0, not now:**
1. *Preferred* — a small **synthetic SVAR2 fixture** reproducing the pinned record class,
   built through the project's generator + the SVAR2 write path, if that path can produce
   the divergence.
2. *Fallback* — a **tiny committed slice of the real corpus** (few regions × few samples,
   size-bounded, checked whether license/data policy permits committing) as a test asset.
3. *If neither is cheap* — document the backend-coverage gap explicitly in the test and
   spec (matching the predecessor's "note the gap rather than claim coverage" rule) and
   rely on Layer 2a to keep any residual overflow loud.

### Verify & report

- The reported config now completes: re-run the repro; assert it yields batches and does
  **not** raise `SlotOverflowError`.
- Full `pixi run -e dev pytest tests -q` + `cargo test`; rebuild Rust
  (`maturin develop --release`) before any pytest importing the extension.
- Update the predecessor spec's status table (Layer 1 → done) and **correct** the Phase 0
  findings' "SVAR2 unreachable / every backend opens as Haps" claim with a pointer here.
- Post the resolution to #315: corrected repro params, confirmed SVAR2 root cause, fix.

## Escalation branch (unchanged)

If Phase 0 shows the estimate genuinely cannot cheaply mirror `Svar2Haps`' emitted count
(e.g. it depends on data only known after reconstruction), fall back to the deferred
producer-side **grow-or-split** auto-recovery — a scope change requiring re-approval. The
Evidence above points to a clean counting mismatch, so Layer 1 is expected to be tractable.

## Risks & open questions

- **Fixture reproducibility.** Whether a *synthetic* SVAR2 fixture can reproduce the pin is
  itself a Phase-0 output; the fallback ladder above handles a negative result.
- **`Svar2Haps` counting reuse.** If `Svar2Haps` has no clean per-instance count entry
  point to reuse, Layer 1 must factor one out (small refactor) rather than duplicate the
  `p_eff` logic — duplication is what drifted in the first place.
- **Committing real data.** The fallback fixture must respect data-sharing policy and size
  limits; if disallowed, drop to option 3.
- **`p_eff` vs ploidy.** The corpus is ploidy 2 with `unphased_union=True` (→ `p_eff = 1`);
  the fix and test must cover both `unphased_union` values so the ploidy-collapse path is
  not the only one exercised.

## Docs & skill upkeep

No public-API change expected (internal estimate + test). Confirm `docs/source/*` and
`skills/genvarloader/SKILL.md` need no change; add a conventional-commit changelog entry.
```
