# double_buffered variant-windows slot-fit (#315)

**Date:** 2026-07-21
**Status:** ✅ Fully implemented — Layers 2a/2b/2c + Phase 0 done; **Layer 1 (estimate
fix) done (2026-07-23)**, resolved via the real-corpus pin. See
[Phase 0 findings](./2026-07-21-phase0-findings.md) and
[2026-07-23-phase0-realcorpus-findings.md](./2026-07-23-phase0-realcorpus-findings.md).
**Issue:** [#315](https://github.com/mcvickerlab/GenVarLoader/issues/315)
**Branch/target:** `fix/315-double-buffered-vw-slot` → `main` (this is the *released*
`gvl.Dataset.to_dataloader` double-buffer path, **not** StreamingDataset work — see the
CLAUDE.md scope clarification in PR #318).

## Problem

`Dataset.to_dataloader(..., mode="double_buffered")` over flat **`variant-windows`**
output fails deterministically at scale with:

```
RuntimeError: ProducerError (ValueError): buffer is smaller than requested size
  at _shm_layout.py::_write_flat_variant_windows  (np.frombuffer offset+count > len(buf))
```

The identical view is fine under `mode="buffered"` and `mode="manual"`, and all three
are byte-identical in a per-variant parity check at small scale — so the **data is
correct**. This is purely a shared-memory **slot-sizing** bug on the `double_buffered`
path.

**Reported failing configuration** (released PyPI wheel **0.40.0**; corrected params from
the reporter — the issue's `flank_length=8` snippet was illustrative):

- flat `variant-windows`, `ref="window"`, `alt="allele"`, `unphased_union=True`, `jitter=0`
- **`flank_length=128`** (≈257-token windows), **tracks OFF** (trackless Hartwig corpus)
- `buffer_bytes=2 GiB` (→ `slot_bytes ≈ 1 GiB`), `batch_size=4096`, `shuffle=True`
- 40 regions × 7089 samples (region subset of full Hartwig via `restrict_footprint`)
- Reproduced twice. A 2×4 subset with `buffer_bytes=16 MiB` passes.

`v0.40.0` is the PR #312 merge (`0a6a0f0b`) plus a version-bump commit, so the failing
wheel **includes** #312's byte-accounting fixes — this is not a stale pre-#312 report.

## How slot sizing works today (and why it can overflow)

- `_torch.py::_resolve_buffered_inputs` builds a per-instance byte table
  `bpi = Dataset._output_bytes_per_instance(include_offsets=(mode=="double_buffered"))`.
- `_chunked.py::ChunkPlanner` packs epoch-ordered `(r,s)` instances into chunks so that
  `sum(bpi[r,s]) ≤ slot_bytes`, and exposes `peak_chunk_bytes` (max chunk estimate).
- `_double_buffered_loader.py:275` sizes each shm slot as
  `capacity = HEADER_RESERVED + peak_chunk_bytes + 4096`.
- The producer materializes `chunk = ds[r_idx, s_idx]` and serializes it with
  `_shm_layout.py::write_chunk` / `_write_flat_variant_windows`.

**The invariant that must hold:** for every chunk, the real serialized payload must be
`≤ peak_chunk_bytes + 4096`. Because `peak_chunk_bytes` is derived from the *same*
estimate the chunker packs with, the only headroom for any un-estimated byte is the fixed
**4096** slack. `buffered`/`manual` never serialize into fixed-size slots, so they never
hit this — which is why only `double_buffered` fails.

## Root-cause analysis

Two independent empirical passes (dummy dataset; real file-backed dense multi-variant /
dual-haplotype / indel data) plus a reading of the Rust window kernel establish:

1. **The estimate is byte-exact per-instance for every record type synthetic data can
   produce** — SNP, multiallelic, small/large ins+del (incl. 49 bp), contig-boundary
   indels (padded to full `2L+span`, not clipped), homozygous-on-both-haps, multi-variant
   groups — at **both `flank_length=8` and `128`** (172k variants → same fixed shortfall
   as 673). The per-instance/per-variant under-count slope is **exactly zero**.

2. **The only un-estimated cost is a fixed per-chunk constant** (~48–74 bytes): the
   estimate omits the `+1` terminator entry of each serialized offset array
   (`8 × n_offset_arrays`) and the `≤7 B`/array `_align` padding. This is real but bounded;
   it needs **>~508 `var_fields`** to breach the 4096 slack, so it does **not** explain the
   reported 40×7089 overflow on its own. It is nonetheless a correctness gap (Bug B below).

3. **The writer sizes windows purely from `ilen` and stored allele bytes**, and the
   estimate's *formulas* mirror it exactly (`src/variants/windows.rs`):
   - `ref_window` length per variant `= (end + L) − (start − L) = 1 + max(−ilen, 0) + 2L`
     — `end = start − min(ilen,0) + 1`. There is **no END/SVLEN path**; it is `ilen`-only.
     Matches the estimate's `ref_span = 1 + max(−ilen,0)` + `2L` term-for-term.
   - `alt` (bare, `alt="allele"`) length `=` gathered stored ALT bytes; `alt_window` length
     `= 2L +` stored ALT bytes. Matches `_allele_bytes_sum`.

4. **Symbolic / breakend ALTs are impossible in a written gvl dataset** — `gvl.write`
   *rejects* them (genoray 2.9.1 guard; spec `2026-06-07-reject-symbolic-breakend`). The
   Hartwig corpus was written successfully, so it contains none. (Earlier top suspect
   ruled out.)

**Conclusion.** Since the writer's length formulas and the estimate's formulas are
identical, the overflow **cannot** be a per-window-length error. It must be a mismatch in
**which / how many variants** each side accounts, because the two sides compute that
through *separate code paths*:

- **Estimate (Python):** `Dataset.n_variants()` + `Haps._allele_bytes_sum()` +
  `real_ploidy` grouping.
- **Writer (Rust):** `genotypes[r,s].to_packed()` → `v_idxs` → `gather_alleles()`.

On synthetic VCF data these agree byte-for-byte. On the real corpus they diverge. The
divergence is therefore **real-corpus-specific** — most probably backend-driven (Hartwig
may be PGEN or SVAR2; all synthetic tests used VCF) or a genotype-pattern / `unphased_union`
edge case the synthetic generator does not produce. **The exact term is not yet pinned**
(see Phase 0).

### Collateral bug found en route

**Bug A — `realign_tracks` not propagated to the producer (independent, high severity).**
`with_seqs("variant-windows")` + active tracks requires `realign_tracks=False` in the
parent, but `_build_producer_schema` omits `realign_tracks` from the schema dict and
`_producer._apply_schema` never sets it. The producer's fresh `Dataset.open()` defaults it
to `True`, so replaying the schema raises
`ValueError: with_seqs('variant-windows') with tracks requires with_settings(realign_tracks=False)`
at `_reconstruct.py:539`. ⇒ **variant-windows + any active track is completely broken in
`double_buffered` mode** — but it dies at schema replay with a *different* error, never
reaching `write_chunk`, so it is **not** the #315 overflow. Tracked and fixed separately
(its own issue + PR), not folded into this one.

## Goals / non-goals

**Goals**

- Make `double_buffered` variant-windows **work** on the real Hartwig corpus at the
  reported config (no `buffer is smaller than requested size`).
- Guarantee the slot-fit invariant `estimate + slot_overhead ≥ real serialized bytes`
  **by test**, across a record-type *and backend* matrix — so this class of drift is
  caught in CI, not in production.
- Turn any residual overflow into a **precise, actionable error** instead of a cryptic
  `np.frombuffer` failure.

**Non-goals**

- Auto-recovery by growing/splitting oversized chunks (the "robust structural rewrite").
  Deferred: viable only if Phase 0 shows the estimate cannot be made a clean upper bound
  (escalation path below).
- Any change to `buffered`/`manual` modes (they don't use fixed slots).
- Bug A's fix (separate PR).

## Design

### Phase 0 — Pin the divergence (gate)

Run the instrumentation script `diag_315_realcorpus.py` (a per-instance real-`write_chunk`
vs estimate sweep that flags any instance under-counting by >256 B and dumps that
instance's `ilen` / ALT strings / backend) against the actual Hartwig dataset at the
failing params. Output pins **which variant-selection or allele-byte quantity** the
estimate mis-counts relative to the writer, and on **which backend**.

- If the sweep shows **no** per-instance under-count, the overflow is chunk-packing-scale
  specific; re-capture the exact `(r_idx, s_idx)` of a failing chunk from a real
  `to_dataloader` run and feed those indices in.
- The pinned quantity determines Layer 1's exact correction. Layer 1 does **not** proceed
  on a guess.

### Layer 1 — Restore the upper-bound invariant in the estimate

Correct the pinned divergence in the `seq_kind == "variant-windows"` branch of
`Dataset._output_bytes_per_instance` (`_dataset/_impl.py`) so its variant count / allele
bytes are a **provable upper bound** on what the writer emits for that record class and
backend. "Correct" is defined operationally by the Layer 2b property test passing on a
fixture reproducing the pinned record class.

Likely shape (to be confirmed by Phase 0): align the estimate's variant selection with the
writer's `to_packed()` `v_idxs` for the failing backend, or add the missing backend-specific
allele-byte accounting (e.g. an SVAR2/PGEN path where stored ALT bytes differ from what
`_allele_bytes_sum` assumes). If Phase 0 reveals the estimate genuinely cannot be made a
tight upper bound cheaply, escalate (below) rather than bloat the estimate.

### Layer 2a — Producer write-time slot-fit guard (fail loud, fail safe)

Before `write_chunk` writes payload into a slot, compute (or accumulate) the real serialized
size and verify it fits the slot capacity. On overflow, raise a precise error naming: chunk
index, real bytes vs available bytes, the offending `(r_idx, s_idx)` (or the first field to
overflow), and the actionable remedy (*lower `batch_size` or raise `buffer_bytes`*). This
replaces the cryptic `np.frombuffer` ValueError and — critically — makes **any future
estimator drift fail loud and safe** instead of surfacing as a mysterious crash.

Implementation note: `write_chunk` already advances a `cursor`; add a size-only pre-pass (or
a bounds check against `len(buf)` before each `np.frombuffer` write) so the guard fires with
context *before* the raw overflow. Cheap; no format change.

### Layer 2b — Upper-bound property test (the durable fix)

Add a test asserting, for a **matrix of schemas × record types × backends**, that

```
sum(_output_bytes_per_instance(include_offsets=True)) + slot_overhead(schema)
    ≥ real write_chunk cursor − HEADER_RESERVED
```

for every chunk. Matrix must include: SNP, indel (incl. large), contig-boundary, multi-
allelic-atomized, homozygous, multi-variant groups, `unphased_union` on/off, `ref`/`alt` ∈
{window, allele}, `flank_length` ∈ {small, 128}, **VCF *and* PGEN *and* SVAR2 backends**,
and the Phase-0 record class once pinned. This is the test that would have caught #315 and
the #312 drift saga; it is the primary defense against recurrence.

### Layer 2c — Fold per-chunk overhead into slot capacity (Bug B)

Replace the magic `4096` slack with a **derived** per-chunk overhead bound: the offset-array
`+1` terminators (`8 × n_offset_arrays`) and `_align` padding (`8 × n_serialized_arrays`),
computable from the schema (field count, window-slot count). `slot_overhead(schema)` in
Layer 2b uses the same derivation, so the test and the runtime bound stay in lockstep. This
removes the last un-accounted term and makes `peak_chunk_bytes + slot_overhead` a true
upper bound independent of field count.

### Layer 3 — Bug A (`realign_tracks` propagation) — *separate issue + PR*

Add `schema["realign_tracks"] = ds.realign_tracks` in `_build_producer_schema` and thread it
through `_apply_schema`'s `settings_kwargs`, plus a regression test that
`double_buffered` + variant-windows + an active track iterates. Filed as its own issue,
cross-linked to #315; not in this PR.

## Testing strategy

- **Layer 2b property test** is the centerpiece (record-type × backend matrix).
- **Regression test** reproducing #315's shape: a variant-windows view at `flank_length=128`
  with the Phase-0 record class, packed into a slot sized to just fit the *estimate*, must
  serialize without overflow (i.e. estimate is a true upper bound).
- **Guard test** (Layer 2a): force an intentional under-count (monkeypatch the estimate low)
  and assert the producer raises the precise, actionable error — not the raw `np.frombuffer`
  ValueError.
- Before pushing: full tree (`pixi run -e dev pytest tests -q`) + `cargo test`, since this
  touches shared shm-layout / dataloader code. Rebuild Rust (`maturin develop --release`)
  before any pytest that imports the extension.

## Docs & skill upkeep (per CLAUDE.md gates)

No public-API surface change is expected (internal slot sizing + a producer guard). Confirm
`docs/source/*` and `skills/genvarloader/SKILL.md` need no update; if Layer 1 changes any
user-facing `double_buffered` guidance or `buffer_bytes` advice, update `dataset.md`/`faq.md`
accordingly. Add a `changelog`-worthy conventional-commit message (does **not** substitute
for prose docs).

## Risks & open questions

- **Phase 0 may not reproduce a per-instance under-count** on the corpus subset available to
  the maintainer. Mitigation: capture the exact failing `(r_idx, s_idx)` from a real run.
- **The divergence may be intractable to model cheaply in the estimate** (e.g. requires
  replicating `to_packed()`'s selection). → **Escalation path:** fall back to the deferred
  producer-side *grow-or-split* handling (reallocate a larger slot and signal the consumer,
  or split the chunk), which guarantees correctness without an exact estimate. This changes
  scope and would be re-approved with the user before implementing.
- **Backend coverage gap:** if the property-test matrix can't cheaply build a PGEN/SVAR2
  fixture matching Hartwig, note the gap explicitly rather than claim full coverage.

## Rollout

1. Phase 0 (instrumentation) → pin divergence.
2. Implement Layer 2b (property test, expected to fail on the pinned case) → Layer 1 (fix
   until it passes) → Layer 2c → Layer 2a.
3. Full-tree + cargo verification; update #315 with the corrected repro + confirmed root
   cause + fix.
4. Layer 3 (`realign_tracks`) as a separate issue + PR.

## Implementation status (2026-07-21)

| Layer | Status | Commit / note |
|-------|--------|---------------|
| **Phase 0** — pin divergence | ✅ done (negative) | Estimate is a **verified per-instance upper bound** for every synthetic record type × VCF/PGEN/SVAR on the released `Haps` path; `M == K == W`; `real − est` is a per-chunk constant covered by `slot_overhead`. Real-corpus divergence **not reproducible** synthetically. See [Phase 0 findings](./2026-07-21-phase0-findings.md). |
| **2c** — schema-derived slot overhead (drop magic 4096) | ✅ done | `slot_overhead_bytes(dataset)` in `_slot_overhead.py`, wired into `_double_buffered_loader.py`; floored at 4096. |
| **2a** — producer write-time guard | ✅ done | `SlotOverflowError(ValueError)` in `_shm_layout.py`; `write_chunk` translates the cryptic numpy overflow into an actionable "lower batch_size / raise buffer_bytes" error. |
| **2b** — upper-bound property test | ✅ done | `tests/unit/test_slot_fit_property.py`; asserts `est + slot_overhead ≥ real` across ref/alt × unphased_union × flank × {dummy,VCF,PGEN,SVAR} (44 views, all pass). |
| **1** — estimate upper-bound fix for the pinned case | ✅ **done (2026-07-23)** | Real-corpus pin (not synthetic): the Hartwig corpus is SVAR2-format and reconstructs via `Svar2Haps`, not `Haps` — the estimate's `n_vars_total`/`ref_span`/`alt_alleles` terms all read `Svar2Haps`'s permanently-empty SVAR1-shaped placeholders (`.genotypes`/`.variants`), collapsing to a flat 32 B/instance regardless of real content. Fixed via a shared `Svar2Haps.measure_variant_payload` entry point that re-sources all three terms from the same read-bound decode/fold the reconstructor itself uses, so estimator and reconstructor cannot drift apart again. Confirmed on the real corpus: `to_dataloader(mode="double_buffered")` over the exact reported 40×7089 config now yields batches with no `SlotOverflowError`. See [2026-07-23-svar2-variant-windows-slot-fit-design.md](./2026-07-23-svar2-variant-windows-slot-fit-design.md) and [2026-07-23-phase0-realcorpus-findings.md](./2026-07-23-phase0-realcorpus-findings.md). |
| **3** — `realign_tracks` propagation (Bug A) | ⬜ separate issue + PR | Out of scope for this PR. |

**Net effect shipped so far:** the reported Hartwig config no longer fails with a cryptic
`np.frombuffer` error — it now raises `SlotOverflowError` with an actionable remedy — and
the slot-fit invariant is locked in CI. Making that config *fit* (rather than fail loud)
still requires Layer 1, which is blocked as above.
