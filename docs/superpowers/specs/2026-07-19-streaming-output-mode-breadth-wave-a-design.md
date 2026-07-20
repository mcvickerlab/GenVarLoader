# Design: `StreamingDataset` — output-mode breadth, Wave A (length + jitter + annotated)

**Date:** 2026-07-19
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md` (Plan 5)
**Issue:** [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) — Wave A
**Splits with:** [#304](https://github.com/mcvickerlab/GenVarLoader/issues/304) — Wave B (variants-output surface)
**Folds in:** [#300](https://github.com/mcvickerlab/GenVarLoader/issues/300) — same-POS atom-ordering invariant
**Follows:** VCF/PGEN backends [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276)
(PR [#299](https://github.com/mcvickerlab/GenVarLoader/pull/299)) and the SVAR1 window engine
[#275](https://github.com/mcvickerlab/GenVarLoader/issues/275)
**Target branch:** `streaming` (via `spec/277-output-mode-wave-a`)

## Summary

Bring the write-free `StreamingDataset` up to a first slice of the written `Dataset`'s output
configuration surface, on the three merged backends (SVAR1, VCF, PGEN). Wave A covers the three
knobs that **reuse the existing streaming window buffer and the fused reconstruct kernels** —
plumbing, not new decode:

- **`with_len(length)`** — fixed-length or ragged haplotype/annotated output.
- **jitter** — read-time window jitter with a defined rng contract.
- **`with_seqs("annotated")`** — `AnnotatedHaps` (haplotypes + per-position `var_idxs` + `ref_coords`).

The expensive half — the variants-output surface (`with_seqs("variants")`/`"variant-windows"`,
`min_af`/`max_af`, `var_fields`), which needs *new per-variant channels* decoded into the window
buffer — is **Wave B (#304)**, picked up immediately after Wave A. The waves are sequential, not
parallel: both edit the same `StreamingDataset` config surface and backend classes in
`_streaming.py`, and Wave B builds on the `with_seqs` dispatch Wave A establishes.

### Why these three knobs are the "cheap" group

Wave A's knobs feed the **same** sparse inputs the haplotype kernels already consume
(`geno_offset_idx` + per-hap CSR + the static `v_starts`/`ilens`/`alt_alleles`/`alt_offsets`
table). Concretely:

- `with_len` is a single `output_length` argument already honored by
  `reconstruct_haplotypes_fused` / `reconstruct_annotated_haplotypes_fused`
  (`src/ffi/mod.rs`, `output_length >= 0 ? output_length : ref_len+diff`).
- jitter is a translation of the query window plus a wider read; it needs **no new per-variant
  data**.
- annotated uses `reconstruct_annotated_haplotypes_fused`, which takes **identical** sparse inputs
  to the plain haplotype kernel and only additionally emits the two annotation buffers.

Wave B's knobs, by contrast, need REF alleles, the `AF` INFO field, dosages/FORMAT fields, and
dataset-global variant ids decoded into `DecodedWindow` (`src/record_stream/transpose.rs`), none
of which `fill_decoded_window` extracts today. That is the seam the split follows.

### What ships (Wave A)

- `sds.with_len(200)` / `sds.with_len("ragged")` — fixed and ragged output for haplotypes and
  annotated.
- `sds.with_settings(jitter=128, rng=0)` — read-time jitter with a documented rng contract.
- `sds.with_seqs("annotated")` → `AnnotatedHaps` batches, **byte-identical** `var_idxs` and
  `ref_coords` vs `gvl.write()` + `Dataset[r, s]` on all three backends.
- Corrected #300 fixtures/docs + a genuine (non-coincidental) same-POS atom-ordering test.
- Docs + `skills/genvarloader/SKILL.md` updated for every new knob.

### Out of scope (Wave B / later)

`with_seqs("variants")`, `with_seqs("variant-windows")`, `min_af`/`max_af`, `var_fields`,
`with_seqs("reference")`, spliced streaming, tracks/intervals (#279), SVAR2 backend (#298 —
picks up this generic code when it merges), `num_workers>0`. `with_output_format` is not added:
`to_iter` always yields `Ragged`.

## Background: the current surface

`StreamingDataset` is a frozen, `slots=True` dataclass (`_streaming.py:48`), iterable-only via
`to_iter` (a fixed cartesian sweep of BED × samples, region-major). Today:

- `with_seqs` accepts **only** `"haplotypes"` (`_streaming.py:493`); everything else raises.
- `jitter` is already a constructor kwarg but raises `NotImplementedError` for any nonzero value
  (`_streaming.py:171`).
- There is no `with_len`, no `with_settings`, no `seed`/`deterministic`, and no `output_length`
  threading anywhere on the class.
- Three backends exist: `_Svar1Backend` (read/generate split + `Svar1StreamEngine`), `_VcfBackend`
  and `_PgenBackend` (both drive `RecordStreamEngine`). All three currently emit **haplotype bytes
  only** — a `Ragged[S1]` of shape `(batch, ploidy, ~length)`.

The written `Dataset` reference implementations Wave A mirrors:

- `Dataset.with_len` (`_impl.py:561`), `with_settings` (`_impl.py:209`), `with_seqs`
  (`_impl.py:642`).
- `Haps.__call__` / `get_haps_and_shifts` / `_prepare_request` (`_haps.py:590`/`634`/`690`) —
  `output_length` threading, `shifts` sampling, and the annotated dispatch to
  `_reconstruct_annotated_haplotypes` (`_haps.py:934`).
- Read-path jitter in `_query.py:166-172` (per-region `jitter_off` draw, window translation).

## Design

### 1. Config surface — mirror the written `Dataset`

Add three `with_*` methods, each returning a copied frozen dataclass (like the existing
`with_seqs`). Keep them thin: they only set state; all behavior lives in `_iter_batches`/the
backends.

- **`with_seqs(kind: Literal["haplotypes", "annotated"]) -> StreamingDataset`.** Extend the
  existing method's accepted literals. `"annotated"` sets an internal `_seq_kind` field; anything
  outside the two literals still raises `NotImplementedError` naming Wave B for
  `"variants"`/`"variant-windows"` and `"reference"`.
- **`with_len(length: int | Literal["ragged"]) -> StreamingDataset`.** Sets `_output_length`
  (int, or a sentinel for ragged). Validation mirrors the written path where it applies to
  streaming (fixed `length >= 1`; and, when `jitter>0`, the eff-length-fits-the-read check — see
  §2). **`"variable"` is intentionally not accepted:** on the written path it only selects the
  `ArrayDataset` vs `RaggedDataset` *container*, and `to_iter` always yields `Ragged`. One obvious
  way; users pad the ragged output themselves. `with_len` docstring states this explicitly.
- **`with_settings(*, jitter: int | None = None, rng: int | np.random.Generator | None = None,
  deterministic: bool | None = None) -> StreamingDataset`.** Mirrors `Dataset.with_settings`'s
  relevant subset **with the same parameter names** (`rng`, not `seed`). Sets `_jitter`, the rng,
  and `_deterministic`. Only the jitter/rng-relevant settings are exposed in Wave A
  (min_af/max_af/var_fields/splice/tracks settings are Wave B or later).

New internal frozen fields (defaults preserve today's behavior): `_seq_kind: type = RaggedSeqs`,
`_output_length: int | Literal["ragged"] = "ragged"`, `_jitter: int = 0`, `_rng: int |
np.random.Generator | None = None`, `_deterministic: bool = True`. The existing `jitter`
constructor kwarg is wired to `_jitter` (and the `NotImplementedError` guard at `_streaming.py:171`
is removed — see §2).

### 2. `with_len` and jitter (shared reconstruction plumbing)

Both knobs flow through one place: the per-window/per-batch reconstruction call in `_iter_batches`
and the backends' `generate_batch`/engine drive.

**`output_length` threading.** The fused kernels already accept `output_length` and honor
`output_length >= 0 ? fixed : ragged`. Wave A passes it through:

- ragged (default): pass the ragged sentinel (`-1`), current behavior — no change.
- fixed int `L`: pass `L`; the kernel emits exactly `L` bytes per hap. Output offsets become
  uniform, so batches are dense-paddable by the user but still delivered as `Ragged`.

For the SVAR1 backend this threads through `svar1_generate_batch`/`Svar1StreamEngine`; for VCF/PGEN
through `RecordStreamEngine`'s `generate_batch_core`. Each already calls the reconstruct kernel —
Wave A supplies a non-sentinel `output_length` and (for non-deterministic fixed output) per-hap
`shifts`.

**`shifts` (fixed-length, non-deterministic).** The written path samples a within-window shift for
over-length haplotypes only when `output_length` is a fixed int and not deterministic
(`_haps.py:732-742`); it is zero for ragged/`deterministic`. Wave A reproduces this exactly: a
per-`(row, ploid)` `shifts` array drawn from the run's `Generator` (see rng contract below),
`max_shift = diffs.clip(min=0) + (lengths - output_length).clip(min=0)`. When `deterministic`
(the default) or ragged, `shifts` is all zeros → byte-parity with `Dataset[r, s]` under matching
settings.

**Jitter.** The written path pre-extends stored regions by `max_jitter` at *write* time and
translates the window within those stored flanks (`_query.py:166`). `StreamingDataset` has no write
step and no stored flanks, so it jitters at **read** time:

1. **Widen the read window.** For each region, read `[start - jitter, end + jitter]` (clamped to
   `[0, contig_len]` using the `reference` the dataset already holds and the contig lengths). This
   pulls the extra variants and reference bytes a jittered window can land on. The read window is
   the existing granularity; widening it does not change the CSR/offsets machinery.
2. **Draw and translate.** Draw one `jitter_off ~ U[-jitter, +jitter]` per region (shared across
   that region's samples/ploidy, matching the written path), in **sweep order**, from the run's
   `Generator`. The reconstructed query window for the region becomes `[start + jitter_off, end +
   jitter_off]` (length preserved). Contig-boundary clamping guarantees the jittered+widened window
   stays in bounds; the `with_len` eff-length check (`output_length + 2*jitter <= region_len +
   2*jitter`, i.e. `output_length <= region_len`) is validated at `with_len`/`with_settings` time.

**rng contract (the load-bearing decision).**

- `jitter == 0` (default) → **deterministic, byte-identical** to `gvl.write()` + `Dataset[r, s]`.
  This is the existing parity gate and stays the primary correctness oracle.
- `jitter > 0` → **not** byte-identical to `Dataset[r, s]`, and Wave A does not pretend otherwise.
  The written path's per-region draw order is keyed to arbitrary `[r, s]` *query* order; the
  stream's order is a fixed cartesian *sweep*. There is no shared stored-flank contract either.
  So jitter is specified as a **documented, reproducible augmentation**, not a parity target:
  - A `StreamingDataset` set with `rng=k` produces the **same** jittered output on every sweep
    (reproducibility), drawing per region in sweep order from `np.random.default_rng(k)`
    (matching `Dataset.with_settings`'s `rng` handling).
  - Correctness is gated by properties, not byte-equality: (a) every output has the requested
    length; (b) with `jitter=0` vs `jitter>0` the emitted genomic coordinates differ by exactly
    the drawn per-region offset and nothing else (translation-only — the same variants apply,
    shifted); (c) all coordinates stay within the contig.
  - The rng contract (`rng` semantics, per-region-in-sweep-order draw, translation-only guarantee)
    is documented on `with_settings` and in `faq.md`/`dataset.md`, per the issue's "match or
    document" clause.

### 3. `with_seqs("annotated")`

Route annotated output through `reconstruct_annotated_haplotypes_fused` (already registered,
`src/ffi/mod.rs`), which consumes the **same** sparse inputs as the haplotype kernel and
additionally emits `annot_v_idxs` (variant index per output base, `-1` for reference) and
`annot_ref_pos` (reference coordinate per base). The Python output type is `AnnotatedHaps`
(`_types.py:27`): `haps` (S1) + `var_idxs` (i32) + `ref_coords` (i32), assembled into ragged
batches the same way haplotype batches are today.

Per backend:

- **SVAR1.** `geno_v_idxs` is `Svar1Reader::variant_idxs()` — **already dataset-global** ids.
  Annotated `var_idxs` is byte-parity essentially for free; add the annotated code path to
  `svar1_generate_batch`/`Svar1StreamEngine` (emit the two extra buffers) and to `_Svar1Backend`.
- **VCF / PGEN.** The kernel writes whatever `geno_v_idxs` values it is given into `annot_v_idxs`.
  Streaming's `geno_v_idxs` are **window-local column indices** (`transpose.rs` pushes column `v`),
  so annotated `var_idxs` must be mapped to dataset-global ids: `global_id = window_global_base +
  local_v`. This requires plumbing the **window's global variant base** into the window buffer:
  - PGEN already computes a global `var_start` per window (`src/record_stream/pgen.rs`) — thread it
    through to the annotated emission.
  - VCF needs the `.gvi`-row (file-order) global index of the window's first decoded atom exposed
    from the decode; add it to `DecodedWindow`/the engine's per-window metadata.

  The mapping is applied either in the kernel (add the base to the emitted annotation, cheapest) or
  in a post-pass on `annot_v_idxs` (no kernel change). The choice is an implementation-plan detail;
  parity is identical either way.

Byte-parity for annotated is gated on **all three** backends for `haps`, `var_idxs`, and
`ref_coords`, under `jitter=0`.

#### 3a. #300 resolution (folded in — it is a prerequisite for VCF/PGEN `var_idxs` parity)

Investigation established that #300's premise is **false at the pinned genoray rev**: the streaming
`ChunkAssembler`'s tie-break `seq` is **not** lexicographic ALT bytes — it is a file-order counter
`record_seq << 32 | atom_ix` (genoray `chunk_assembler.rs:346`, min-heap keyed `(pos, seq)`). The
write oracle orders by `.gvi` row = VCF/PVAR **file-row order** (`_write.py` applies no variant
sort; `_var_ranges.py` returns variants in `.gvi`-row order). And `gvl.write` **rejects
multiallelic records** (`_write.py:606`), so every accepted record yields exactly one atom
(`atom_ix ≡ 0`) and the streaming key collapses to `(pos, record-order)` = the oracle's key.

**Therefore the two global orderings match by construction, not coincidentally** — for the only
inputs `gvl.write` accepts (coordinate-sorted, pre-split biallelic). Full VCF/PGEN `var_idxs`
byte-parity is achievable with **no genoray change and no rev bump**: `window_global_base +
local_v` is the file-order global id on both sides. The numba-oracle-bug-policy's "canonicalize in
genoray" path does **not** trigger — file-row order is `gvl.write`'s defined contract (it demands
sorted, pre-normalized input), so it is the intended order, not arbitrary garbage.

#300's deliverables, now lightweight:

1. **Correct the wrong docs.** Fix the `vcf_snp_ins_del_multi` fixture docstring
   (`tests/dataset/conftest.py:494-505`, and its PGEN twin), the `test_streaming_vcf_parity.py`
   module caveat (`:21-27`), and any comment that calls `seq` "lexicographic ALT order" — state the
   true file-order invariant and that both paths honor it by construction.
2. **Lock the invariant genuinely (not coincidentally).** Add a same-POS fixture whose two
   pre-split biallelic rows are in file order **T-then-G** (file order ≠ lexicographic), and a
   **pre-split "triallelic"** (three same-POS biallelic rows) in non-lexicographic file order.
   These prove both decoders still agree because both are file-order — the exact case the old
   docstring feared but never actually exercised. Wire them into the VCF (and PGEN-derived) parity
   tests.
3. **Prove `var_idxs` byte-parity** for all three backends via the annotated parity tests below,
   including on the new same-POS fixtures.

### 4. Testing / parity plan

Oracle: `gvl.write()` + `Dataset.open()[r, s]` under matching settings (the standing contract).

- **`with_len` parity.** Byte-identical haplotypes and annotated for `with_len(L)` (fixed) and
  `with_len("ragged")` vs `Dataset.with_len(L)` / `.with_len("ragged")`, `jitter=0`,
  `deterministic=True`, all three backends.
- **Annotated parity.** Byte-identical `haps`, `var_idxs`, `ref_coords` vs
  `Dataset.with_seqs("annotated")[r, s]`, all three backends, `jitter=0`. Includes the new #300
  same-POS / triallelic fixtures for VCF and PGEN.
- **Jitter (property tests, not byte-parity).** (a) requested-length invariant; (b)
  translation-only vs `jitter=0` (same variants applied, coordinates shifted by exactly the drawn
  per-region offset); (c) in-contig-bounds; (d) reproducibility across two sweeps with the same
  `rng`; (e) two different `rng` values differ.
- **#300 invariant.** The new same-POS/triallelic fixtures pass annotated `var_idxs` parity,
  demonstrating file-order agreement on a case where file ≠ lexicographic order.
- **Regression.** Existing haplotype parity + scale/engine tests stay green (Wave A must not
  perturb the `jitter=0` haplotype path).

Rust changes (annotated emission for SVAR1 engine; VCF `.gvi` global-base in `DecodedWindow`)
require `pixi run -e dev maturin develop --release` before pytest, per CLAUDE.md.

## Decomposition (parallelizable within Wave A)

These sub-pieces are largely independent once the config surface (below) lands, and can be
implemented by parallel agents per the project's SDD conventions:

1. **Config surface (serial prerequisite).** `with_len`, `with_settings`, extended `with_seqs`,
   the new frozen fields, and the `output_length`/`shifts` threading skeleton in
   `_iter_batches`/backends (ragged path unchanged; fixed path wired but exercised by piece 2).
2. **`with_len` + fixed-length `shifts`** across all three backends + parity tests.
3. **Jitter** (read-window widening, per-region draw, translation) + rng-contract docs + property
   tests.
4. **Annotated** across all three backends (SVAR1 free; VCF/PGEN global-base plumbing) + parity
   tests.
5. **#300** doc corrections + new same-POS/triallelic fixtures + invariant tests. (Depends on
   piece 4's VCF/PGEN global-base plumbing for the `var_idxs` assertions.)
6. **Docs + skill** — `README`/`faq.md`/`dataset.md`/`api.md` + `skills/genvarloader/SKILL.md` for
   every new knob; roadmap + board update.

Pieces 2/3/4 all touch `_streaming.py`'s backends and `_iter_batches`; sequence 1 first, then run
2/3/4 with awareness of the shared file (or land 1→2→(3‖4)). 5 depends on 4. The implementation
plan (writing-plans) will finalize the exact parallel/serial split per CLAUDE.md's
dispatching-parallel-agents guidance.

## Risks

- **Jitter is the riskiest piece.** Read-window widening interacts with contig boundaries and the
  `max_mem` window sizing. Mitigation: clamp early, keep jitter orthogonal to the offsets buffer
  (it widens the query, not the CSR machinery), and gate correctness with the translation-only
  property test rather than byte-parity.
- **VCF global-base for annotated.** Exposing the `.gvi`-row global index of a window's first atom
  from the Rust decode is the one genuinely new bit of plumbing. Mitigation: PGEN's `var_start`
  already proves the shape; the VCF filler decodes a known contig range, so the base is the count
  of prior-contig-order variants — available from the same `.gvi`/CSI machinery the range read uses.
- **#300 being "already correct" is load-bearing.** If a future genoray rev changes the
  `ChunkAssembler` tie-break away from file-order, the invariant breaks. Mitigation: the new
  non-lexicographic same-POS test *is* the guard — it fails loudly if the ordering ever diverges,
  and it is pinned to the current rev's documented contract.

## References

- Written output-mode surface: `_impl.py` (`with_len`/`with_settings`/`with_seqs`), `_haps.py`
  (`__call__`/`get_haps_and_shifts`/`_prepare_request`/`_reconstruct_annotated_haplotypes`),
  `_query.py:166` (jitter), `_types.py:27` (`AnnotatedHaps`).
- Streaming surface: `_streaming.py` (`StreamingDataset`, `_Svar1Backend`, `_VcfBackend`,
  `_PgenBackend`), `src/record_stream/` (`engine.rs`, `transpose.rs`, `vcf.rs`, `pgen.rs`),
  `src/ffi/mod.rs` (`reconstruct_annotated_haplotypes_fused`).
- #300 mechanism: genoray `chunk_assembler.rs:62-66,346-348`, `_write.py:606,726-782`,
  `tests/dataset/conftest.py:469-543`, `tests/dataset/test_streaming_vcf_parity.py`.
- Conventions: `docs/archive/roadmaps/rust-migration.md` (byte-identical parity contract),
  numba-oracle-bug-policy.
