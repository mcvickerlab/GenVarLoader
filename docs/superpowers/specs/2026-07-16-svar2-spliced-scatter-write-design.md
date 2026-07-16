# SVAR2 spliced haplotypes: scatter-write the kernel output (no Python re-order)

> **Status:** design approved · **Date:** 2026-07-16 · **Issue:** #273 ·
> **Base:** `main` @ `09d5055` (includes #272)
>
> Follow-up to #272. Touches `src/` → the rust-migration roadmap gate
> (`docs/roadmaps/rust-migration.md`) and its byte-identical parity contract apply.

## 1. Motivation

SVAR1 reconstructs spliced haplotypes with no post-kernel re-order. SVAR2 has no
equivalent: `Svar2Haps.__call__` reconstructs in region order and then re-orders
the *output bytes* in Python to reach spliced order.

#272 replaced a byte-level fancy-index with a per-row slice + `np.concatenate`
(189 ms → 23 ms). This removes the remaining pass entirely.

### How SVAR1 actually does it

Worth stating plainly, because the issue's framing ("fused Rust spliced kernel")
suggests something more exotic than what is there. `reconstruct_haplotypes_spliced_fused`
(`src/ffi/mod.rs:1701`) does **not** know about splicing. It permutes the small
per-element *inputs* in Python (regions, shifts, geno_offset_idx — O(rows)) and
passes `plan.permuted_out_offsets` as the kernel's output offsets. The kernel then
writes each element at its final spliced address. The trick is **reordering
metadata instead of bytes**.

The index spaces already line up: `build_splice_plan` permutes `k = query * E + e`
with `E = ploidy` for haplotypes (`_splice.py:87`), and SVAR2's kernel row index is
already `k = q * P + p`. `plan.permutation` indexes SVAR2 hap rows with no
translation.

The blocker is the contig-group loop: SVAR2 calls the kernel once per contig, each
call allocating its own contiguous buffer, while splice order interleaves contigs.
A group's rows therefore land at **non-contiguous** destinations, and the core's
parallel writer (`src/reconstruct/mod.rs:740`) requires monotone, gap-free
`out_offsets` for its `split_at_mut` chain.

## 2. Measurement (Phase 0 spike — completed 2026-07-16)

chr22 spliced `ds[:, :]`: 165 transcripts × 5 samples × ploidy 2 = 6600 elements,
13.2 MB output. Both backends timed in one process. The node is too noisy for
medians (svar1's median swung to 45 ms against a 25 ms min), so these are
**minimums over 25 reps**.

| segment | ms | share |
|---|---|---|
| svar1 spliced (baseline) | 25.15 | — |
| **svar2 spliced** | **35.33** | **1.41×; gap 10.2 ms** |
| `_ragged_arange_gather` (the concatenate) | **13.73** | **39%** |
| `get_haps_and_shifts` (gather #2 + kernel) | 6.41 | 18% |
| `plan_sizing` (gather #1) | 3.44 | 10% |
| `reverse_masked` (RC pass) | 1.70 | 5% |
| unattributed (plan build, regroup, glue — shared with svar1) | 9.62 | 28% |

**The single re-order pass exceeds the whole svar1 gap.** The mechanism matters:
a microbenchmark of the same shape gives **8.53 ms** for per-row slice+concatenate
vs **0.96 ms** for a flat memcpy of the same 13.2 MB. The 13.7 ms is therefore
**~9× the memory-bandwidth floor** — ~1.3 µs/row of numpy per-slice dispatch across
6600 rows, not bytes moved. #272 removed the index materialization; per-row dispatch
is what remains.

A scatter write deletes that cost rather than moving it to Rust: the kernel writes
those bytes once regardless, just at a different address.

Reproduce: build the datasets via `tests/benchmarks/data/build_svar_splice.py`
(needs `tests/data/fasta/hg38.fa.bgz` from `pixi run -e dev gen`, `plink2`, and the
uncommitted `tests/benchmarks/data/chr22_5s_hapsafe.pgen`), then time `ds[:, :]` for
both backends in one process, taking minimums. Attribution was done by wrapping
`haplotype_lengths_for_plan`, `get_haps_and_shifts`, `_assemble_haps`,
`_ragged_arange_gather`, and `_Flat.reverse_masked` with timers.

## 3. Design

One lever: **scatter write**.

Python allocates the full output buffer once (`total = plan.permuted_out_offsets[-1]`).
Each per-contig kernel call receives that buffer as an out-param plus an
`(n_group_rows, 2)` int64 array of **global destination bounds**. Row `k` writes to
`[permuted_out_offsets[j], permuted_out_offsets[j+1])` where `j = dest_rank[k]`, the
inverse permutation. No re-order, no concatenate, no RC post-pass.

### 3.1 Rust

**Core** (`src/reconstruct/mod.rs`, `reconstruct_haplotypes_from_svar2`): generalize
`out_offsets: ArrayView1<i64>` to per-row `out_bounds: ArrayView2<i64>` `(n_work, 2)`.
The parallel path currently carves chunks with a `split_at_mut` chain that assumes
each row starts where the previous ended; it must instead carve in **ascending-start
order** (argsort the bounds, O(n_work) rows — negligible) so interleaved
destinations with gaps are legal. Gaps are rows belonging to *other* contig groups
and are simply skipped; each call writes only its own rows. The serial path already
uses raw pointers for disjoint sub-ranges and needs no ordering change.

Contract, to be asserted: bounds are pairwise disjoint and within the buffer. The
existing gap-free monotone case remains valid input, so the existing allocate-and-
return FFI keeps working by building `(start, end)` pairs from its computed offsets.

**FFI** (`src/ffi/mod.rs`): new out-param entry alongside
`reconstruct_haplotypes_from_svar2_readbound`, taking `out: PyReadwriteArray1<u8>`,
`out_bounds`, and an optional per-row `to_rc` mask; returns nothing. This follows
the established house pattern — `reconstruct_haplotypes_from_sparse`
(`src/ffi/mod.rs:560`) already takes an out-param and releases the GIL via
`py.detach`, so the aliasing/GIL question is settled precedent, not new ground.
In-kernel per-row RC reuses `crate::reverse::rc_flat_rows_inplace`, as the SVAR1
fused entry does.

### 3.2 Python (`python/genvarloader/_dataset/_svar2_haps.py`)

`Svar2Haps.__call__` gains a spliced branch that:

1. computes `dest_rank` (inverse of `plan.permutation`) and per-row destination
   bounds — O(rows);
2. allocates `out = np.empty(total, np.uint8)`;
3. loops contig groups, slicing bounds for rows `k = qsel[:, None] * P + arange(P)`
   and calling the out-param FFI;
4. wraps with `_Flat.from_offsets(out, (n_perm, None), plan.permuted_out_offsets)`.

`to_rc` un-permutes to per-row order (`_getitem_spliced` builds it permuted) and is
passed to the kernel; the `reverse_masked` post-pass goes away.

The reconstruct call is currently reached via `get_haps_and_shifts`, which raises on
`splice_plan is not None`. Give the spliced path its own `_reconstruct_spliced`
method rather than parameterizing `get_haps_and_shifts` by destination bounds: the
two paths call *different* FFI entries (allocate-and-self-size vs scatter-into) and
differ on shifts (spliced is always zero — `_getitem_spliced` asserts `jitter == 0`
and `deterministic`) and on sizing. What they genuinely share — the per-group cache
slice and reference slice — is already factored into `_gather_inputs` and
`_ref_for_contig`, so a shared loop helper would add indirection over three lines.

Deletes: the splice-path use of `_ragged_arange_gather` (`_svar2_haps.py:384`).
The helper itself stays — the unspliced and variants paths still use it.

## 4. Out of scope

- **The unspliced path.** Its ragged reads deliberately let the kernel self-size;
  preallocating would force an extra diffs gather and could regress. `_assemble_haps`
  stays and measured 0.06 ms thanks to #272's single-contig fast path.
- **Gather reuse (the "L2" lever).** The spliced path gathers twice — once to size
  the plan, once to reconstruct. Measured 3.44 ms, and not all of it is excess
  (svar1 computes diffs before its plan too). It needs an opaque gather-handle type
  across the FFI: real complexity for a second-order win that this change may make
  irrelevant. File as a follow-up **only if** post-change numbers justify it.
- **A single Rust call looping contigs internally** (the literal "fused svar2
  analog"). Buys one FFI crossing over N, duplicates the group loop that
  `_haplotype_diffs`/tracks/variants already share, and the crossing is not the cost.
- Annotated haps, tracks, and variants splicing — all still `NotImplementedError`.

## 5. Testing & acceptance

**Byte-identity is the contract.** Must stay green:

- `tests/dataset/test_svar2_dataset.py::test_svar2_spliced_*`
- `tests/benchmarks/test_e2e_svar_splice.py::test_svar1_svar2_spliced_parity`
- full tree (`pixi run -e dev pytest tests -q`) + `cargo test` — shared code changes,
  so `tests/unit/` must be covered, not just `tests/dataset/`.

New coverage:

- Rust unit test: scattered/interleaved `out_bounds` (a multi-contig destination
  pattern with gaps) produces the same bytes as the contiguous path, serial and
  parallel.
- Python test: a **multi-contig** spliced read. The chr22 benchmark is single-contig
  and so exercises only the gap-free case; the gap-carving logic is exactly what a
  single-contig test cannot reach.

**Perf gate** (per the shared-node protocol): same-session before/after on the chr22
spliced benchmark, comparing **minimums**, both backends in one process. Absolute
wall-clock across sessions is not a valid signal on this node.

Target: svar2/svar1 ≈ 1.0× or better (projection ≈ 0.8×, i.e. ~20 ms vs 25 ms) with
no per-getitem Python re-order pass.

**Rebuild the extension before testing** (`pixi run -e dev maturin develop --release`)
— pytest imports the stale `.so` otherwise, and parity tests would silently validate
the old binary.

## 6. Risks

- **The ≈0.8× projection assumes the unattributed 9.62 ms is genuinely shared with
  svar1.** It is plan-build + regroup + indexing glue that both backends run, but
  svar1's own breakdown was not profiled. If svar2's share there is inflated, the
  result lands nearer 1.0× than 0.8× — still meeting the issue's target.
- **Scattered writes and cache locality.** Rows are ~2 KB and written once; the
  concern is theoretical. The perf gate would catch it.
- **`out_bounds` disjointness is now a caller-enforced invariant.** Previously the
  gap-free offsets array made overlap unrepresentable. Mitigate with a debug-mode
  disjointness assert in the carve, in the spirit of the existing monotonicity
  `debug_assert`.

## 7. Docs

No public API change: `Dataset`/`gvl.write` signatures, `__all__`, and on-disk format
are untouched — this is an internal read-path optimization. So no `skills/genvarloader/SKILL.md`
or `docs/source/api.md` update is triggered. Update `docs/roadmaps/rust-migration.md`
(tick the task, record before/after measurements under the relevant checkpoint, set the
phase marker + PR link) as the roadmap gate requires.
