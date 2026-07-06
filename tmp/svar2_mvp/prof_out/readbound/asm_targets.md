# SVAR2 read-bound: cargo-asm work-list (post B1-B3)

Captured 2026-07-06 via `tmp/svar2_mvp/prof_perf.sh` after rebuilding with
`RUSTFLAGS="-C force-frame-pointers=yes"` on top of commits through
`a297d24` (B2: pre-size split_to_flat/decode_variants_from_split). Source
data: `tmp/svar2_mvp/prof_out/readbound/native_baseline.md` (fresh,
post-B1-B3). Pre-B1-B3 numbers preserved in
`tmp/svar2_mvp/prof_out/readbound/native_after_b1b3.md` (misleadingly named —
it holds the OLD/pre-B1-B3 capture).

Sanity re-check (A3-style): grep for `overlap_batch`/`dense_union` in the
fresh `native_baseline.md` returned **no matches** — union oracle confirmed
absent from the read-bound path. `SearchTree::build` tops out at 1.54%
(haplotypes_germline), consistent with the benign per-region `find_ranges`
search, not a regression.

## Per-mode top-5 native symbols (including excluded numpy/libc/kernel), with asm-fixable-Rust characterization

### haplotypes_germline (K=191)
| self% | symbol | owner |
|---|---|---|
| 12.31% | `[k] 0xffffffffb1a0f327` | kernel (unresolved, page-fault/mmap path) |
| 11.64% | `mapiter_trivial_get` | numpy |
| 11.11% | `LONG_add_AVX2` | numpy |
| 10.86% | `LONG_subtract_AVX2` | numpy |
| 3.86% | `genoray_core::query::gather_haps_readbound` | genoray (Rust, asm-fixable) |

Characterization: dominated by numpy int64 add/sub kernels + kernel-side
paging; only `gather_haps_readbound` (3.86%) + `SearchTree::build` (1.54%,
outside top-5) clear our cutoff → **~5.4% of self-time is asm-fixable Rust**
in this mode. B1 (skip redundant haplotype gather) appears to have already
squeezed most of the Rust cost out of the haplotypes path.

### variants_germline (K=7143)
| self% | symbol | owner |
|---|---|---|
| 18.31% | `genoray_core::query::gather_haps_readbound` | genoray (Rust, asm-fixable) |
| 6.79% | `PyArray_Repeat` | numpy |
| 5.43% | `genvarloader::svar2::decode_variants_from_split` | gvl (Rust, asm-fixable) |
| 5.21% | `genvarloader::svar2::split_to_flat` | gvl (Rust, asm-fixable) |
| 4.65% | `_int_free` | libc |

Characterization: this is the mode with the most asm-fixable Rust left —
`gather_haps_readbound` + `decode_variants_from_split` + `split_to_flat` +
`merge_keys` (2.06%) + `svar2_codec::decode_key` (2.33%) sum to **~33.3% of
self-time in gvl/genoray Rust**, the single largest optimization target of
the four modes.

### haplotypes_somatic (K=37)
| self% | symbol | owner |
|---|---|---|
| 11.02% | `mapiter_trivial_get` | numpy |
| 10.07% | `[k] 0xffffffffb1a0f327` | kernel |
| 9.09% | `PyUnicode_RichCompare` | python |
| 8.09% | `LONG_add_AVX2` | numpy |
| 8.07% | `LONG_subtract_AVX2` | numpy |

Characterization: essentially **no asm-fixable Rust remains** — only
`SearchTree::build` appears at all (0.55%, below cutoff). Entirely
numpy/python/kernel-structural at this point.

### variants_somatic (K=1792)
| self% | symbol | owner |
|---|---|---|
| 6.86% | `genoray_core::query::gather_haps_readbound` | genoray (Rust, asm-fixable) |
| 5.74% | `PyUnicode_RichCompare` | python |
| 4.41% | `[k] 0xffffffffb1a0f327` | kernel |
| 4.02% | `mapiter_get` | numpy |
| 3.89% | `_int_free` | libc |

Characterization: `gather_haps_readbound` + `decode_variants_from_split`
(2.56%) + `split_to_flat` (1.79%) + `merge_keys` (2.31%) sum to **~13.5% of
self-time in Rust** — about half of variants_germline's asm-fixable budget.

## Work-list: gvl/genoray native symbols with self-time ≥1.5% in ANY mode

| symbol | repo/file:line | max self% | modes (self%) | perf.data tag(s) |
|---|---|---|---|---|
| `genoray_core::query::gather_haps_readbound` | genoray `src/query.rs:1086` | 18.31% | haplotypes_germline (3.86%), variants_germline (18.31%), variants_somatic (6.86%) | `haplotypes_germline`, `variants_germline`, `variants_somatic` |
| `genvarloader::svar2::decode_variants_from_split` | gvl `src/svar2/mod.rs:269` | 5.43% | variants_germline (5.43%), variants_somatic (2.56%) | `variants_germline`, `variants_somatic` |
| `genvarloader::svar2::split_to_flat` | gvl `src/svar2/mod.rs:159` | 5.21% | variants_germline (5.21%), variants_somatic (1.79%) | `variants_germline`, `variants_somatic` |
| `svar2_codec::decode_key` | genoray `svar2-codec/src/lib.rs:237` | 2.33% | variants_germline (2.33%) | `variants_germline` |
| `genoray_core::spine::merge_keys` | genoray `src/spine.rs:63` | 2.31% | variants_germline (2.06%), variants_somatic (2.31%) | `variants_germline`, `variants_somatic` |
| `genoray_core::search::SearchTree::build` | genoray `src/search.rs:93` | 1.54% | haplotypes_germline (1.54%), haplotypes_somatic (0.55%, below cutoff) | `haplotypes_germline` |

**6 functions clear the ≥1.5% cutoff** → fan-out size of 6 for the parallel
cargo-asm pass. All are genoray-owned except `decode_variants_from_split` and
`split_to_flat` (gvl-owned, `src/svar2/mod.rs`).

Sub-cutoff, noted for completeness (do not fan out on these): `genvarloader::svar2::hap_diffs_svar2`
peaks at 0.58% (haplotypes_germline) — below the 1.5% bar in every mode.

## Excluded categories (structural, not cargo-asm-fixable)
- libc: `_int_malloc`/`_int_free`/`__memmove_avx_unaligned_erms`/`__memcmp_avx2_movbe`/`malloc`/`__libc_calloc`
- numpy: `_multiarray_umath` internals — `mapiter_trivial_get`/`mapiter_get`/`LONG_add_AVX2`/`LONG_subtract_AVX2`/`PyArray_Repeat`/`npyiter_buffered_iternext`/`_contig_to_contig`
- Python interpreter / GC: `_PyEval_EvalFrameDefault`, `gc_collect_main`, `deduce_unreachable`, `visit_reachable`, `dict_traverse`, `PyUnicode_RichCompare`, `PyObject_RichCompare(Bool)`, `list_contains`, `list_index`
- Rust std/alloc/toolchain (not gvl/genoray application code): `alloc::vec::Vec<T>::from_iter` variants, `__rustc::__rust_dealloc`, `__rustc::__rust_no_alloc_shim_is_unstable_v2`, `alloc::raw_vec::RawVec::grow_one`
- unresolved kernel samples: `[k] 0xffffffffb1a0f327`, `[k] 0xffffffffb14fa26d`, `[k] 0xffffffffb1c011e0`
