# Phase 0 findings — #315 double_buffered variant-windows slot under-count

**Date:** 2026-07-21
**Issue:** [#315](https://github.com/mcvickerlab/GenVarLoader/issues/315)
**Gate for:** Task 5 (Layer 1 — restore the estimate's upper-bound for the pinned case)

> **Correction (2026-07-23):** the "SVAR2 unreachable / every backend opens as `Haps`"
> claim below (see "Why the design doc's hypothesized mechanism does not apply here") is
> **wrong**. The real Hartwig corpus is SVAR2-format-2.0.0 and **does** reach `Svar2Haps`
> on the released `Dataset.open` + `to_dataloader` path — that is exactly what this doc's
> "not available in this environment" gap was hiding. Once the real corpus was symlinked
> in, `type(view._seqs).__name__ == "Svar2Haps"` was confirmed directly. The real
> divergence this doc couldn't pin turned out to be `Svar2Haps`-specific: its
> `n_vars_total`/`ref_span`/`alt_alleles` terms all read the permanently-empty
> SVAR1-shaped placeholders (`Svar2Haps.genotypes`/`.variants`) that exist only to satisfy
> `isinstance(_, Haps)` checks, so the estimate under-counts to a flat 32 B/instance
> regardless of real content. This is a genuinely separate root cause from anything on the
> `Haps` path analyzed below (which remains correct as measured — `M == K == W` still
> holds there). See
> [2026-07-23-phase0-realcorpus-findings.md](./2026-07-23-phase0-realcorpus-findings.md)
> for the full pin and
> [2026-07-23-svar2-variant-windows-slot-fit-design.md](./2026-07-23-svar2-variant-windows-slot-fit-design.md)
> for the fix design. The rest of this document (below) is preserved as originally
> written for the historical record of the `Haps`-path analysis, which is still accurate.

## Summary (the pin)

**The estimate `Dataset._output_bytes_per_instance` is a byte-exact per-instance
upper bound for every record type and storage backend reproducible with the
project's synthetic generator, on the released `Haps` read path — which is the only
path `Dataset.to_dataloader(mode="double_buffered")` uses.** The per-instance and
per-variant under-count slope is **exactly zero**; the only gap between the estimate
and the real serialized payload is a fixed **per-chunk** constant (~48–55 bytes) that
`slot_overhead_bytes` (Task 2) now covers with a 4096 floor.

**The reported real-corpus (Hartwig) divergence could NOT be reproduced** with any
available synthetic data. Pinning the exact offending record class therefore
**requires the real Hartwig dataset**, which is not present in this environment.
Task 5 cannot proceed on a synthetic reproduction; see "Required input to Task 5".

## What was measured

The real Hartwig corpus and reference are not available here, so Phase 0 was run
against the shared synthetic `session_document` (`tests/_builders/case.py`) — the
standardized 14-record VCF with **multiallelic** (`chr2:1110696 G>A,T`;
`chr2:1234567 A>GA,AC` microsatellite), **indel** (`chr1:1010696 GAGA…>G`,
`chr1:1110696 A>TTT`), **homozygous-ALT** (`1|1`, `1/1`, `2/2`), and **missing/half-call**
(`1|.`, `.|1`, `./.`) records — built through **all three variant backends** (VCF,
PGEN, SVAR) and opened with the synthetic reference.

Reported failing view encoded exactly: flat `variant-windows`, `ref="window"`,
`alt="allele"`, `unphased_union=True`, `jitter=0`, tracks OFF, `flank_length ∈ {8, 128}`.

Scripts (this session's scratch): `phase0_backend_sweep.py` (per-instance real vs
estimate), `phase0_scaling.py` (per-chunk vs per-instance decomposition, the decisive
one), `phase0_MKW.py` (variant-count identity).

### Result 1 — `real − est` is a per-CHUNK constant, not per-instance

Packing the full 8×3 grid into one chunk and tiling it to N = 24 → 192 instances:

| backend | uu | N=24 | N=48 | N=96 | N=192 | d(real−est)/d(inst) |
|---------|-----|------|------|------|-------|---------------------|
| vcf/pgen/svar | True/False | ~55 | ~54 | ~52 | ~48 | **≈ 0 (slightly negative)** |

`delta = real − est` stays flat (actually *decreases* slightly with N, from
`_align` padding amortizing). `est + slot_overhead_bytes(view) ≥ real` holds for
every backend, every N, both `unphased_union` values, at `flank_length=128`. A
per-instance under-count would grow `delta` linearly with N; it does not.

The earlier single-instance sweep (`phase0_backend_sweep.py`) reported a ~63 B
"under-count" on *every* instance of *every* backend — that was a **methodology
artifact**: a 1-instance chunk pays the full per-chunk overhead once, so it looks
per-instance. The multi-instance tiling above is the correct measurement and shows
the term is per-chunk (the known Bug B constant, now covered by Task 2).

### Result 2 — variant-count identity `M == K == W`

For every synthetic instance, every backend, `unphased_union` on/off (including the
homozygous sites):

- `M` = `n_variants()` folded (the estimate's flank-token multiplier `n_vars_total`)
- `K` = `len(to_packed(genotypes[r,s]).data)` (the estimate's `ref_span`/`alt` set)
- `W` = emitted window count (`len(ref-slot.seq_offsets) − 1`, the writer's truth)

**`M == K == W` in 100% of instances.** This is structural, not luck: on the `Haps`
path `self.n_variants = self.genotypes.lengths` (`_haps.py:298`) and `to_packed().data`
are the *same ragged array's* length, so `M == K` by construction, and the Rust window
kernel emits one window per packed v_idx, so `W == K`.

## Why the design doc's hypothesized mechanism does not apply here

The design doc (`…-double-buffered-vw-slot-fit-design.md`, "Conclusion") proposed the
divergence was the estimate's variant *selection* (`n_variants()` + `_allele_bytes_sum`)
disagreeing with the writer's `to_packed()` → `v_idxs`. On the `Haps` path this
disagreement is **structurally impossible** (Result 2): both derive from the same
`genotypes` ragged array. The `ref_span`/`alt_alleles` terms already consume
`to_packed()` directly (`_impl.py:1615`, `_haps.py:779`). So Task 5's suggested
"align the estimate with `to_packed()`" fix would be a no-op — they are already aligned.

`Svar2Haps` (the SVAR2 / StreamingDataset reconstructor) has separate variant-counting
logic (`_svar2_haps.py`, `p_eff = 1 if unphased_union else P`), but it is **not reachable**
from the released `Dataset.open` + `to_dataloader` path (#315's scope): every backend —
including a SparseVar source — opens as `Haps`, verified this session. SVAR2 is
StreamingDataset-only (out of scope per the design doc and CLAUDE.md).

> **This paragraph is superseded — see the 2026-07-23 correction note at the top of this
> document.** A SVAR2-format-2.0.0 dataset (the real Hartwig corpus) *does* open as
> `Svar2Haps` on this exact released path; "every backend opens as `Haps`" was true only
> of the backends available to this session (VCF/PGEN/SVAR1), not SVAR2, which was
> untested here for lack of a real corpus.

## Consequence for the plan

Because the estimate is *already* a correct per-instance upper bound for everything
reproducible, and the real overflow still occurs on Hartwig, the true cause is one the
synthetic generator cannot produce and this environment cannot pin. This is the plan's
anticipated escalation branch (design doc "Risks & open questions" → escalation path):
**the estimate cannot be corrected by a guess**, and the plan forbids guessing
("Layer 1 does not proceed on a guess").

Tasks 2 (slot_overhead), 3 (SlotOverflowError guard), 4 (upper-bound property test),
and 6 (verify/docs) are valid and valuable independent of the pin — they harden the
mechanism and turn any residual overflow (including Hartwig's) into an actionable
error. They are being completed. **Task 5 is blocked** pending one of the two inputs
below.

## Required input to Task 5 (one of)

1. **Real-corpus pin.** Run, on the actual Hartwig dataset + its reference, this
   session's `diag_315_realcorpus.py` (per-instance real-`write_chunk` vs estimate
   sweep) — OR the corrected `phase0_scaling.py` adapted to that dataset — and capture:
   the offending `(r_idx, s_idx)`, the `ilen`/ALT strings/genotype pattern of the
   variants in that instance, the backend, and whether `d(real−est)/d(inst) > 0`
   (per-instance under-count) vs a per-chunk constant that only breaches the slack at
   large chunk sizes. That dump is the fixture recipe Task 5's failing test encodes.

2. **Escalate to grow-or-split (scope change, needs user approval).** If the divergence
   cannot be modeled cheaply in the estimate, adopt the deferred producer-side
   grow-or-split auto-recovery (reallocate a larger slot / split the chunk on overflow),
   which guarantees correctness without an exact estimate. This is a current non-goal;
   re-approve scope with the user before implementing.

Until one of these arrives, the SlotOverflowError guard (Task 3) ensures the reported
Hartwig config fails with an actionable message ("lower batch_size or raise
buffer_bytes") rather than the cryptic `np.frombuffer` `ValueError` — a strict
improvement even without the estimate fix.
