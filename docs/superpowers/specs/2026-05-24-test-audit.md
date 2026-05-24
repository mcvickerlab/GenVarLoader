# Test Audit (post-refactor)

**Date:** 2026-05-24
**Spec:** docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md
**Baseline:** docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt

This audit classifies every test function in `tests/integration/` (the
sole test tier as of this audit) into one of three buckets:

- **Delete** — tautological, fully duplicated by another test, or
  testing behavior removed in the refactor campaign.
- **Port** — valuable behavior, but the test is unnecessarily E2E and
  should move to `tests/unit/` once the relevant builder exists.
- **Keep-as-integration** — true regression coverage (write/read
  roundtrips, variant-source parity, golden 1kg checks).

A fourth section lists **polymorphism gaps**: code paths the existing
suite doesn't exercise, organized by axis.

## Method

For each file under `tests/integration/`:
1. Read every test function (and parametrize/case generator).
2. Cross-reference against the coverage baseline (per-module Cover% and
   missing-line ranges).
3. Assign one bucket. Note rationale in one line.

## Per-file classification

---

### `tests/integration/test_fasta.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_pad_right` | Port | Exercises `Fasta.read` boundary padding — no `Dataset` needed; pure `Fasta` unit test once a `make_fasta` builder exists. |
| `test_pad_left` | Port | Same: left-boundary padding of `Fasta.read`. |
| `test_no_pad` | Port | Same: `NoPadError` path of `Fasta.read`. |

All three belong together in `tests/unit/fasta/test_fasta.py`. Coverage
baseline: `_fasta.py` is at 51%; these tests drive only the happy-path
read + pad branches; the uncovered 49% (in-memory path, multi-fetch,
`__len__`, chunked reader) is unrelated to these three.

---

### `tests/integration/test_interval_track.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_bigwigs_satisfies_interval_track_protocol` | Delete | Tautological protocol conformance check — asserts `hasattr` on a class whose `__init__` we control; provides no regression value beyond what type-checking already guarantees. |

---

### `tests/integration/test_ref_ds.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_getitem` (cases: `case_ragged_regions`, `case_no_regions`) | Port | Exercises `RefDataset.__getitem__` via parametrized cases; `RefDataset` is independent of the write pipeline, so this is a component test once a `make_ref_dataset` builder exists. |
| `test_padded_slice` (4 cases: `no_pad`, `pad_left`, `pad_right`, `pad_both`) | Port | Exercises `_dataset/_utils.padded_slice` directly; no `Dataset` involved; pure unit test. |
| `test_refdataset_unspliced_defaults` | Delete | Asserts `ds.is_spliced is False` and `ds.splice_info is None` — these are fixture-setup assertions, not behavior tests; the state is set by the `RefDataset(reference, bed)` constructor with no optional args. |

---

### `tests/integration/test_ref_ds_splicing.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_spliced_single_col` | Keep-as-integration | End-to-end splice: constructs `RefDataset` with `splice_info`, indexes, verifies byte-for-byte concat against unspliced slices. |
| `test_spliced_two_col_reorders_exons` | Keep-as-integration | Verifies that `exon_number` column drives sort order before concat — a non-trivial data-flow check. |
| `test_spliced_mixed_strand` | Keep-as-integration | Negative-strand RC path through `RefDataset` splicing. |
| `test_with_settings_disable_splice` | Port | Tests `RefDataset.with_settings(splice_info=False)` state transition — once a builder for `RefDataset` state exists this is a unit test. |
| `test_with_settings_enable_splice` | Port | Same: `with_settings(splice_info=...)` enable path. |
| `test_with_settings_validation` | Port | Tests `RuntimeError` raises for incompatible jitter+splice and deterministic+splice combinations — input-validation logic, not data-path logic; no roundtrip needed. |
| `test_subset_to_transcripts` | Keep-as-integration | Exercises `RefDataset.subset_to` by transcript name, verifying the result via actual `__getitem__` output — cross-cutting state check. |
| `test_spliced_output_length_variable` | Port | Tests `with_len("variable")` output shape on spliced output; can be unit-tested against an in-memory `RefDataset`. |
| `test_spliced_rejects_fixed_length` | Port | Tests that spliced + fixed-len raises `RuntimeError`; pure validation logic. |

---

### `tests/integration/test_table.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_table_init_from_long_df` | Port | Exercises `Table.__init__` with long-form DataFrame; no write/read roundtrip. |
| `test_table_init_missing_canonical_column_raises` | Port | Validation error path in `Table.__init__`. |
| `test_table_init_from_dict_of_dfs` | Port | `Table.__init__` with per-sample dict input. |
| `test_table_column_map_renames_long_form` | Port | `column_map` renaming in long-form input. |
| `test_table_column_map_per_sample_dict` | Port | `column_map` renaming in per-sample-dict input. |
| `test_table_from_path_long_form` (4 param: csv/tsv/parquet/arrow) | Port | `Table.from_path` file-format dispatch — no write pipeline involvement. |
| `test_table_from_path_per_sample_dict` | Port | `Table.from_path` with dict of paths. |
| `test_table_from_path_unknown_extension` | Port | `Table.from_path` unknown-extension error. |
| `test_table_count_intervals_matches_brute_force` | Port | Core interval-counting kernel versus brute-force O(n×m); pure `Table` unit test. |
| `test_table_count_intervals_unknown_contig_returns_zeros` | Port | Unknown-contig zero-return path. |
| `test_table_intervals_from_offsets_roundtrip` | Port | `Table._intervals_from_offsets` roundtrip; no write pipeline. |
| `test_table_count_intervals_normalizes_contig_names` | Port | `chr`-prefix normalization in `Table.count_intervals`. |

Note: `_table.py` is at 92% coverage; the remaining 8% (lines 106, 164, 183, 229→293 chain) is the empty-contig guard and the `per_sample` dict-of-DataFrames internal path — none covered by the existing integration tests either.

---

### `tests/integration/test_utils.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_bed_to_regions_categorical_strand_returns_int32` | Port | Exercises `bed_to_regions` with a Categorical strand column — regression for a specific numba typing issue; pure utility unit test. |
| `test_bed_to_regions_utf8_strand_still_works` | Port | Sanity for Utf8 strand path in `bed_to_regions`. |
| `test_bed_to_regions_no_strand_defaults_to_plus` | Port | Default strand=1 path in `bed_to_regions`. |
| `test_splits_sum_le_value` | Port | `splits_sum_le_value` utility; pure array-math unit test. |
| `test_normalize_contig_name` (5 cases: match/add/strip/no-match/list) | Port | `normalize_contig_name` — pure utility, no Dataset involved. |

---

### `tests/integration/dataset/test_build_reconstructor.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_haps_only_returns_haps`, `test_ref_only_returns_ref`, `test_tracks_only_returns_tracks`, `test_haps_and_tracks_returns_haps_tracks`, `test_ref_and_tracks_returns_ref_tracks`, `test_neither_raises_value_error`, `test_haps_with_kind_reference_returns_ref`, `test_haps_with_kind_reference_no_reference_raises`, `test_ref_with_haps_kind_raises`, `test_seqs_kind_none_with_haps_storage_returns_tracks_only`, `test_tracks_inactive_with_seqs_returns_seqs_only`, `test_both_inactive_raises` | Port | Factory unit tests using `Mock` — no `Dataset.open` or write pipeline at all; these are already the cleanest tests in the suite and belong in `tests/unit/dataset/test_build_reconstructor.py` verbatim. |

---

### `tests/integration/dataset/test_dataset.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_ds_indexing` (3 seq_types × 7 idx variants = 21 cases) | Keep-as-integration | Smoke-tests `Dataset.__getitem__` across all output modes with real VCF data; catches reconstructor dispatch regressions. |
| `test_rs_indexing` (3 seq_types × 7 r_idx × 7 s_idx = 147 cases) | Delete | Fully subsumed by `test_ds_indexing` above plus `test_subset.py`; doesn't assert on output values — asserts only that indexing doesn't raise. The cross-product parametrize produces 147 session-scope expensive invocations for no additional value. |

---

### `tests/integration/dataset/test_ds_haps.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_ds_haps` (3 variant sources: vcf/pgen/svar) | Keep-as-integration | Golden-file parity test: haplotypes verified byte-for-byte against `pysam`-generated consensus FASTA. This is the canonical regression net for VCF/PGEN/SVAR parity at reconstruction time. |

Note: `rc_neg=False` only; `rc_neg=True` and `annotated`/`variants` output modes are not covered by this file.

---

### `tests/integration/dataset/test_ds_haps_1kg.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_ds_haps_1kg` (3 variant sources: bcf/pgen/svar) | Keep-as-integration | 1kGP-scale golden regression; marked `@pytest.mark.slow`; exercises real population data and multi-sample index at scale. Keep as the integration canary; it is the only test hitting the 1kg fixture. |

The `test_ds_haps.py` + `test_ds_haps_1kg.py` pair is correct: small dataset for fast CI, large dataset as a slow regression gate.

---

### `tests/integration/dataset/test_get_splice_bed.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_default_keeps_only_t1`, `test_zero_based_start`, `test_chrom_end_unchanged`, `test_dropped_non_cds_rows`, `test_sorted_output`, `test_multiple_of_3_filter_off_keeps_t2`, `test_tsl_none_keeps_t3`, `test_tsl_explicit_value`, `test_contigs_filter`, `test_gene_name_nulls_preserved`, `test_dtypes` | Port | All test `gvl.get_splice_bed` against a synthetic in-memory GTF; no Dataset write/read involved. These are pure unit tests against the GTF-parsing function. Move to `tests/unit/test_get_splice_bed.py`. |

---

### `tests/integration/dataset/test_indexing.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_subset` (5 cases) | Port | Tests `DatasetIndexer.subset_to` directly with a synthetic indexer; no `Dataset.open` or write pipeline. |
| `test_repeated_subset` (7 cases) | Port | Chained `DatasetIndexer.subset_to` idempotency; same pattern. |
| `test_subset_string_regions_and_samples` (5 cases) | Port | String-region path of `DatasetIndexer.subset_to`. |
| `test_subset_string_samples_only_matches_int` | Port | String vs. int equivalence. |
| `test_subset_string_regions_and_samples_matches_int` | Port | Same. |
| `test_chained_subset_region_by_name_not_in_subset_raises` | Port | KeyError on out-of-subset name. |
| `test_repeated_subset_strings` (7 cases) | Port | String-arg chained subset idempotency. |
| `test_subset_to_full` | Port | `DatasetIndexer.to_full_dataset()` restore. |
| `test_parse_idx` (5 cases + 1 xfail) | Port | `DatasetIndexer.parse_idx` covering scalar/list/slice/2D/missing-sample cases; pure indexer unit test. |

All belong in `tests/unit/dataset/test_indexing.py`.

---

### `tests/integration/dataset/test_insertion_fill.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_lower_all_strategies`, `test_lower_empty`, `test_constant_default_is_nan`, `test_flank_sample_negative_width_rejected`, `test_interpolate_order_capped`, `test_lower_unknown_class_raises`, `test_insertion_fill_base_not_instantiable` | Port | Pure unit tests of `InsertionFill` subclasses and `lower()` serializer; no `Dataset` at all. |
| `test_kernel_repeat_5p_default`, `test_kernel_repeat_5p_normalized`, `test_kernel_constant_nan`, `test_kernel_flank_sample_pool_membership`, `test_kernel_flank_sample_deterministic`, `test_kernel_interpolate_linear`, `test_kernel_interpolate_cubic_passes_through_anchors`, `test_kernel_flank_sample_edge_clamp`, `test_kernel_flank_sample_query_hap_affects_hash` | Port | Direct kernel invocation via `shift_and_realign_track_sparse`; no `Dataset` scaffolding. Move to `tests/unit/tracks/test_insertion_fill_kernel.py`. |
| `test_end_to_end_set_insertion_fill` | Keep-as-integration | Exercises the full `with_insertion_fill` plumbing on `get_dummy_dataset()` — verifies that `_recon` wiring is correct end-to-end. |
| `test_dummy_dataset_with_default_insertion_fill_does_not_crash` | Keep-as-integration | Regression: `Tracks` with empty `insertion_fill` dict must not `KeyError` during `__call__`; tied to real reconstruction path. |
| `test_with_insertion_fill_rejects_when_no_tracks_active` | Port | Input-validation path (`with_tracks(False)` → `with_insertion_fill` raises); no reconstruction needed. |
| `test_with_insertion_fill_single_applies_to_all`, `test_with_insertion_fill_dict_partial_falls_back`, `test_with_tracks_prunes_insertion_fill` | Port | Tests `Tracks.with_insertion_fill` method directly on a synthetic `Tracks` object constructed without any `Dataset.open`; pure component tests. |

---

### `tests/integration/dataset/test_issue_153.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_issue_153_hap_lengths` | Keep-as-integration | Filed regression (GH #153): spanning-deletion (`*` allele) ilen accounting bug produced undersized output buffer. Requires real VCF data; validates exact haplotype lengths. Keep as a permanent regression fixture. |

---

### `tests/integration/dataset/test_issue_191_var_fields.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_dosage_absent_when_not_requested` | Keep-as-integration | Regression (GH #191): dosage leaked into output even when not in `var_fields`; requires real SVAR + write pipeline. |
| `test_dosage_present_when_requested` | Keep-as-integration | Positive case for the same bug. |
| `test_available_var_fields_includes_dosage_when_present` | Keep-as-integration | Schema-peek roundtrip with a real SVAR + dosages memmap. |
| `test_available_var_fields_excludes_dosage_when_absent` | Keep-as-integration | Negative case using the canonical SVAR fixture. |
| `test_available_info_fields_lists_numeric_columns_without_loading` | Port | Exercises `_Variants.available_info_fields` on a SVAR path; no `Dataset.open` required — pure `_Variants` component test once the SVAR fixture is accessible as a path fixture. |
| `test_from_table_info_fields_filter` | Port | `_Variants.from_table(info_fields=pick)` — component test; no write pipeline needed. |
| `test_from_table_info_fields_none_loads_all` | Port | Back-compat path of `_Variants.from_table`. |
| `test_load_info_extends_info_dict` | Port | `_Variants.load_info` merging; component test. |
| `test_load_info_idempotent_for_already_loaded_fields` | Port | Idempotency of `load_info`; component test. |
| `test_haps_from_path_filters_info_loading` | Keep-as-integration | Verifies that `Dataset.open` with default `var_fields` does NOT load extra info columns or memmap dosages — touches the full `open` → `Haps.from_path` chain. |
| `test_haps_available_var_fields_from_schema` | Keep-as-integration | `available_var_fields` reflects file schema, not loaded state — requires real SVAR written by `gvl.write`. |
| `test_dataset_open_accepts_var_fields` | Keep-as-integration | `Dataset.open(var_fields=...)` end-to-end routing. |
| `test_dataset_open_default_var_fields_is_minimum_useful_set` | Keep-as-integration | Canonical default `var_fields` contract. |
| `test_with_settings_lazily_loads_new_info_field` | Keep-as-integration | `with_settings(var_fields=[..., new_field])` lazy-load path; requires real SVAR. |
| `test_with_settings_lazily_loads_dosages` | Keep-as-integration | `with_settings(var_fields=[..., 'dosage'])` lazy-memmap path. |

---

### `tests/integration/dataset/test_jitter.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_jitter` (3 variant sources: vcf/pgen/svar) | Keep-as-integration | Validates jitter mechanics end-to-end: reconstructs annotated haplotypes and checks that `ref_coords` reflects the expected random offset; requires real data and the full reconstruction pipeline. Exercises `annotated` output mode. |

---

### `tests/integration/dataset/test_open_vs_settings_parity.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_open_vs_with_settings_parity_state` | Keep-as-integration | Regression for #176 Bug 2: `with_settings(var_filter=...)` was silently dropped from `_recon`; probes internal state. |
| `test_open_vs_with_settings_parity_output` | Keep-as-integration | Same regression — byte-for-byte output parity between `open(splice_info, var_filter)` and `with_settings(splice_info, var_filter)` paths. |

---

### `tests/integration/dataset/test_rc_packing.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_rc_returns_packed_buffer` (3 cases: all_false/all_true/mixed) | Port | Unit test of `ak.to_packed(ak.where(...))` packing invariant using synthetic `Ragged`; no `Dataset` at all. Move to `tests/unit/test_ragged.py`. |
| `test_unspliced_single_item_buffer_packed` | Keep-as-integration | E2E regression: buffer-size invariant after `_rc` with real VCF + write pipeline. Catches the doubled-buffer ak.where leak in the reconstruction path. |
| `test_spliced_reference_pos_strand_matches_fasta` | Keep-as-integration | Reference-mode spliced output matches FASTA byte-for-byte; validates `_getitem_spliced` → `_rc` for positive strand. |
| `test_spliced_reference_neg_strand_is_rc_of_fasta` | Keep-as-integration | Same for negative strand RC. |
| `test_multi_exon_spliced_buffer_packed` | Keep-as-integration | Multi-exon ploidy-interleaving regression: buffer-size invariant on spliced haplotype output with 2-exon transcripts. |
| `test_multi_exon_spliced_matches_fasta_concat` | Keep-as-integration | Multi-exon reference output matches FASTA concat per exon. |
| `test_cds_start_codon_is_atg_nearly_always` | Keep-as-integration | Real CDS dataset guard (skipped unless env vars set); production-level biological correctness check. |
| `test_cds_internal_stops_bounded` | Keep-as-integration | Same env-gated real-data test for stop codon count in reconstructed CDS. |
| `test_spliced_tracks_round_trip` | Keep-as-integration | Tracks splice path buffer-packing invariant; skipped if no tracks in fixture. |
| `test_haptracks_splicing_raises` | Keep-as-integration | `HapsTracks` + splice raises `NotImplementedError` — documents current limitation; skipped if no tracks in fixture. |

---

### `tests/integration/dataset/test_realign.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_sparse` (4 cases: snps/indels/spanning_del/shift_ins) | Port | Directly invokes `shift_and_realign_track_sparse` kernel; no `Dataset` scaffolding. Move to `tests/unit/tracks/test_realign.py`. |

---

### `tests/integration/dataset/test_splice_plan.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_plan_no_inner_axes`, `test_plan_ploidy_2`, `test_plan_multi_sample_ploidy_2`, `test_plan_total_bytes_consistent`, `test_plan_single_element_rows`, `test_plan_inner_fixed_size_3`, `test_plan_dtype_invariants` | Port | Pure unit tests of `build_splice_plan`; all inputs are synthetic numpy arrays; no `Dataset` involved. Move to `tests/unit/dataset/test_splice_plan.py`. |
| `test_ref_call_with_plan_writes_per_element_layout` | Port | Tests `Ref.__call__(splice_plan=...)` directly; constructs `Ref` from `Reference.from_path` — only needs the reference FASTA, not a GVL dataset. Move to unit once a FASTA builder/fixture is available. |
| `test_tracks_call_float32_splice_plan` | Port | Tests `Tracks._call_float32` with a `SplicePlan` directly; builds a synthetic `Tracks` in ~50 lines without `Dataset.open`. Move to `tests/unit/tracks/test_tracks_splice.py`. |

---

### `tests/integration/dataset/test_subset.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_subset` (9 region idx types × 12 sample idx types = 108 cases) | Keep-as-integration | Exercises `Dataset.subset_to` with every combination of index type (None, scalar, neg-scalar, slice variants, list, array, bool, str, list-of-strs, Series) against a real Dataset; validates that `_idxer._r_idx` and `samples` are correct after subsetting. This is the authoritative regression net for `subset_to` polymorphism. |

Note: `test_indexing.py` covers `DatasetIndexer` in isolation; `test_subset.py` covers the full `Dataset.subset_to` integration path — both are valuable and complementary.

---

### `tests/integration/dataset/test_svar_link.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_svar_link_roundtrip`, `test_svar_link_rejects_malformed_fingerprint` | Port | Pure Pydantic model serialization/validation tests; no disk I/O needed. |
| `test_metadata_version_parses_existing_strings`, `test_metadata_version_serializes_back_to_string`, `test_metadata_svar_link_defaults_to_none`, `test_semantic_version_ordering_for_one_based_dispatch` | Port | `Metadata` model unit tests; pure Pydantic. |
| `test_write_from_svar_records_svar_link_and_no_symlink` | Keep-as-integration | Write-time regression: `gvl.write` from SVAR must record `svar_link` in metadata and not create a legacy symlink. |
| `test_resolve_svar_prefers_override` | Port | `_resolve_svar` resolution logic; could be unit-tested with mock paths. |
| `test_resolve_svar_uses_relative_path` | Keep-as-integration | Requires a real written dataset to get the recorded relative path. |
| `test_resolve_svar_falls_back_to_sibling` | Port | Tests sibling-discovery fallback; can be done with `tmp_path` and no GVL write. |
| `test_resolve_svar_raises_when_not_found` | Port | Error path in `_resolve_svar`; no write needed. |
| `test_verify_fingerprint_mismatch_raises` | Keep-as-integration | Fingerprint mismatch check using a real SVAR; validates the write-time fingerprint. |
| `test_verify_fingerprint_ok` | Keep-as-integration | Positive fingerprint check using a written dataset's metadata. |
| `test_open_dataset_via_recorded_svar_link` | Keep-as-integration | Full open roundtrip: write SVAR → open via link → `__getitem__`. |
| `test_open_dataset_after_relocation_via_override` | Keep-as-integration | Relocation + override path through `Dataset.open`. |
| `test_open_dataset_mismatched_svar_raises` | Keep-as-integration | Fingerprint mismatch at open time. |
| `test_open_dataset_legacy_symlink_layout` | Keep-as-integration | Legacy symlink compat + DeprecationWarning. |
| `test_migrate_svar_link_upgrades_legacy_dataset` | Keep-as-integration | `migrate_svar_link` upgrades a legacy dataset and subsequent open emits no warning. |
| `test_migrate_svar_link_is_idempotent` | Keep-as-integration | Idempotency of migration. |
| `test_open_after_joint_relocation_preserves_relative` | Keep-as-integration | Joint relocation (gvl + svar move together) resolves via relative path. |
| `test_migrate_svar_link_refuses_dangling_symlink` | Keep-as-integration | `migrate_svar_link` raises `FileNotFoundError` on dangling symlink. |

---

### `tests/integration/dataset/test_with_settings_var_filter.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_with_settings_var_filter_propagates_to_recon` | Keep-as-integration | Regression for #176 Bug 1: `with_settings(var_filter=...)` must update `_recon`, not just `_seqs`; probes internal state on a written SVAR dataset. |
| `test_with_settings_var_filter_false_clears_recon` | Keep-as-integration | Negative-path regression: clearing `var_filter` must also clear `_recon.filter`. |

---

### `tests/integration/dataset/test_write_tracks.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_write_with_table_only_roundtrip` | Keep-as-integration | End-to-end write→disk validation: `gvl.write(tracks=Table)` without variants; verifies on-disk interval array structure. |
| `test_write_with_mixed_bigwigs_and_table` | Keep-as-integration | Multi-source track write: BigWigs + Table together; validates that both interval directories are created. |
| `test_write_with_variants_and_tracks` | Keep-as-integration | Mixed write: VCF variants + Table tracks; validates metadata sample union. |
| `test_write_duplicate_track_names_rejected` | Port | Input-validation error path in `gvl.write`; no disk I/O strictly needed — could be unit-tested against the write pipeline's validation layer. |

---

### `tests/integration/dataset/test_write.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_write` | Delete | Marked `@pytest.mark.skip`; effectively dead code. If re-enabled it would be Keep-as-integration, but currently it is not run and the coverage it would provide is already covered by `test_ds_haps.py`. |
| `test_write_errors_when_post_index_budget_too_small` | Keep-as-integration | Validates `max_mem` budget enforcement in `gvl.write`; uses `monkeypatch` to trigger the budget error; requires VCF + write pipeline. |
| `test_write_loads_lazy_vcf_index` | Keep-as-integration | Validates that `gvl.write` triggers lazy GVI index load for a `VCF(with_gvi_index=False)`; write→disk roundtrip. |
| `test_write_loads_lazy_pgen_index` | Keep-as-integration | Same for PGEN `load_index=False`. |

---

### `tests/integration/dataset/genotypes/test_choose_exonic_variants.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_choose_exonic_variants_1d_geno_offsets` | Port | Direct kernel invocation of `choose_exonic_variants` with synthetic arrays; pure unit test. |
| `test_choose_exonic_variants_2d_geno_offsets` | Port | Regression for 2-D SVAR offsets bug; direct kernel invocation; no `Dataset`. Move to `tests/unit/dataset/genotypes/test_choose_exonic_variants.py`. |

---

### `tests/integration/dataset/genotypes/test_filter_af.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_filter_af` | Delete | Marked `@pytest.mark.skip`; case generators (`case_filter_af_*`) are defined as module-level functions but none actually exist — the test has never run. Dead code; delete the file. |

---

### `tests/integration/dataset/genotypes/test_rag_variants.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_infer_germ_ccfs` (6 cases) | Port | Direct invocation of `_infer_germline_ccfs` with synthetic arrays; no `Dataset`. Move to `tests/unit/dataset/genotypes/test_rag_variants.py`. |
| `test_rc` (3 cases: no_rc/second_batch/all) | Port | Direct `RaggedVariants.rc_` method test; builds variants in-memory via awkward-array constructors; no `Dataset`. |

---

### `tests/integration/dataset/genotypes/test_reconstruct.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_sparse` (4 cases: snps/indels/spanning_del_pad/shift_ins) | Port | Direct invocation of `reconstruct_haplotype_from_sparse` numba kernel with synthetic inputs; no `Dataset`. Move to `tests/unit/dataset/genotypes/test_reconstruct.py`. |

---

### `tests/integration/tracks/test_annot_tracks.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_annot_tracks` | Keep-as-integration | Exercises `Dataset.write_annot_tracks` + `with_tracks("5ss", "tracks")` + `__getitem__` — requires a written GVL dataset and verifies the annotation track coordinate alignment in the output. |

---

### `tests/integration/tracks/test_i2t_t2i.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_intervals_to_tracks` (2 cases: simple/two_regions) | Port | Direct invocation of `intervals_to_tracks` kernel; synthetic `RaggedIntervals`; no `Dataset`. Move to `tests/unit/tracks/test_i2t_t2i.py`. |
| `test_tracks_to_intervals` (2 cases) | Port | Direct invocation of `tracks_to_intervals`; synthetic data; no `Dataset`. |

---

### `tests/integration/tracks/test_random_nonoverlapping.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_nonoverlapping_intervals` (4 cases) | Port | Tests `nonoverlapping_intervals` from `utils.py` (a test helper, not a library function); this is effectively a property-based test of a test utility. It imports from the sibling `utils.py` in the same directory. Move together with `utils.py` to `tests/unit/tracks/test_random_nonoverlapping.py`. |

---

### `tests/integration/variants/test_sites.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_sites` | Keep-as-integration | Tests `DatasetWithSites` end-to-end: `get_dummy_dataset()` → `DatasetWithSites` → `__getitem__`; exercises the sites-only reconstruction path (`_sitesonly.py`) which has only 62% coverage in the baseline. |

---

### `tests/integration/variants/test_variant_utils.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_path_is_pgen`, `test_path_is_vcf` | Port | Pure path-string predicates in `_variants/_utils.py`; no I/O at all. Move to `tests/unit/variants/test_variant_utils.py`. |

---

## Polymorphism gaps

### Output mode matrix

The axes are: output mode (`haplotypes` / `reference` / `annotated` / `variants`) × length mode (`ragged` / `variable` / `int`) × rc_neg (`True` / `False`).

**Uncovered combinations (referenced to coverage baseline missing lines):**

- **`variants` mode + `ragged`/`padded` `__getitem__`** — `_impl.py` lines 1048–1072 (`_getitem_unspliced` variants branch) and 1093–1124 (`_getitem_spliced` variants branch) are both in the missing list. `test_sites.py` exercises `DatasetWithSites` but the base `Dataset.__getitem__` with `with_seqs("variants")` is never called directly in any test.
- **`haplotypes` mode + `rc_neg=True`** — `test_ds_haps.py` uses `rc_neg=False` only; `test_jitter.py` uses `annotated` mode with `rc_neg` implicitly False. The RC branch inside `_query.py:reverse_complement_ragged` (`_impl.py` proxy at line 826) is only exercised by `test_rc_packing.py` tests in `reference` mode.
- **`annotated` mode + `padded` (fixed-length)** — `test_dataset.py` fires `with_seqs("annotated")` but uses the default output length; padded annotated mode (`ArrayDataset[AnnotatedHaps, ...]`) is not tested.
- **`reference` mode + `ragged`** — `test_dataset.py` sessions cover `reference` mode, and `test_ref_ds.py` covers `RefDataset`; but `Dataset.open(...).with_seqs("reference").with_len("ragged")` is not exercised as a combination in any test.
- **`haplotypes` mode + `padded`** — `test_ds_haps.py` uses `with_len("ragged")` only; `ArrayDataset[NDArrayBytes, ...].__getitem__` for haplotypes is untested (missing `_impl.py` lines 1260–1264).

### `with_settings` lazy reload

- **`jitter` kwarg triggering reload** — `test_jitter.py` calls `with_settings(jitter=..., rc_neg=False)` as part of setup, but doesn't verify that the same `_seqs` instance is reused (no reload test). The reload vs. reuse branches in `_open.py` lines 161, 166–168 are uncovered.
- **`rc_neg=True` via `with_settings`** — no test calls `with_settings(rc_neg=True)` and verifies output; `_impl.py` lines 582, 587 (rc_neg propagation in `with_settings`) are in the missing list.
- **`var_fields` triggering lazy reload vs. reuse** — `test_issue_191_var_fields.py` covers the happy path; the early-exit branch (fields already loaded, no reload needed) at `_open.py` line 70 is marked missing.
- **`deterministic=False`** — no test exercises the non-deterministic jitter path; `_impl.py` line 550 is missing.
- **`var_filter` via `with_settings` on PGEN/VCF sources** — `test_with_settings_var_filter.py` uses SVAR only; `_haps.py` lines 517–530 (filter path for VCF/PGEN geno_offsets layout) are likely untested.

### `subset_to`

- **`Dataset.subset_to` with `Series` of region IDs** — `test_subset.py` covers `pl.Series` for *sample* IDs (`smp_series` case), but no test uses a `pl.Series` of region IDs.
- **`Dataset.subset_to` on a spliced (sp_idxer) dataset** — `_impl.py` lines 954–955 (`sp_idxer.subset_to` path) are marked missing; no test calls `subset_to` after `with_settings(splice_info=...)`.
- **`DatasetIndexer.subset_to` with a Polars Series argument** — covered at the `Dataset` level via `test_subset.py` but not tested directly on `DatasetIndexer` in `test_indexing.py`.

### `__getitem__` polymorphism

The `__getitem__` return type varies by (output mode × len mode). Missing combinations from `_impl.py`/`_query.py` coverage:

- **`(haplotypes, padded)`** — `ArrayDataset.__getitem__` returning `NDArray[bytes_]`; lines 1260–1264 missing.
- **`(reference, ragged)`** — `RaggedDataset.__getitem__` returning `RaggedSeqs` in reference mode; not exercised (only `annotated`/`haplotypes` ragged modes are).
- **`(variants, ragged)` and `(variants, padded)`** — `RaggedVariants` / `NDArrayVariants` return shapes; lines 1048–1072, 1093–1124 missing.
- **`(tracks-only, spliced)`** — `test_rc_packing.py:test_spliced_tracks_round_trip` is a skipped-if-no-tracks test and the fixture has no tracks; effectively uncovered.
- **`(HapsTracks, spliced)` success path** — `test_haptracks_splicing_raises` verifies the `NotImplementedError`; the actual success path (if ever enabled) is obviously uncovered.
- **2-D sample index (reshape)** — `DatasetIndexer.parse_idx` covers the `getitem_reshape` case in `test_indexing.py` but `Dataset.__getitem__` with a 2-D sample list is not exercised end-to-end; `_query.py` lines 344–349 and 359–369 are missing.

### Splicing / GTF

- **Multi-exon haplotypes (non-reference mode)** — `test_rc_packing.py:test_multi_exon_spliced_matches_fasta_concat` uses `reference` mode; `haplotypes` mode multi-exon is not verified byte-for-byte.
- **Negative-strand haplotype splicing** — spliced negative-strand is only exercised in `reference` mode (`test_spliced_reference_neg_strand_is_rc_of_fasta`); `_getitem_spliced` for haplotypes + negative strand is uncovered.
- **`SplicePlan` with `annotated` mode** — `build_splice_plan` + `annotated` output is entirely untested.
- **`SpliceMap.subset_to`** — `_impl.py` lines 954–955 missing (see subset_to section above).
- **`gvl.get_splice_bed` with `strand` column output format** — `test_get_splice_bed.py` covers GTF parsing extensively but does not verify that the output `strand` column is the correct sign convention for downstream `Dataset.open(splice_info=...)`.

### Insertion-fill strategies

All five strategies (`Repeat5p`, `Repeat5pNormalized`, `Constant`, `FlankSample`, `Interpolate`) are tested at the **kernel level** in `test_insertion_fill.py`. The missing coverage is:

- **End-to-end `with_insertion_fill` with any strategy other than `Constant`** — `test_end_to_end_set_insertion_fill` uses `Constant(nan)` only; `Repeat5p`, `Repeat5pNormalized`, `FlankSample`, `Interpolate` are never exercised through the full `Dataset.__getitem__` → `HapsTracks.__call__` path.
- **`with_insertion_fill` on a real (non-dummy) written dataset** — both E2E insertion-fill tests use `gvl.get_dummy_dataset()`; the actual `_tracks.py` re-alignment path with real BigWig intervals + real indel variants is not exercised by any test.
- **`_tracks.py` lines 267–400** (the full `write_transformed_track`-era dead code was deleted; what remains at those lines is the actual `_call_float32` scatter loop) — covered at 41% overall; the indel-aware re-alignment kernel path for every strategy except `Repeat5p` default is not end-to-end verified.

### VCF/PGEN/SVAR parity

- **VCF/PGEN/SVAR parity for `annotated` mode** — `test_ds_haps.py` confirms parity only for `haplotypes` mode; `annotated` mode output (which adds `ref_coords` and `v_idxs` arrays) is never compared across sources.
- **VCF/PGEN/SVAR parity for `variants` mode** — similarly untested across sources.
- **VCF/PGEN parity for track-attached datasets** — `test_ds_haps.py` uses `with_tracks(False)`; no cross-source track parity test exists.
- **PGEN-only: multiallelic normalization** — `_write.py` has a PGEN-specific multi-allelic handling path; no test specifically exercises a PGEN file with multiallelic variants to verify normalization produces VCF-equivalent output.

### Jitter and rc_neg

- **`rc_neg=True` with haplotypes mode** — `test_jitter.py` uses `annotated` mode; no test uses `with_seqs("haplotypes")` with `rc_neg=True` and verifies that negative-strand regions produce RC'd sequences.
- **`rc_neg=True` with reference mode** — `test_spliced_reference_neg_strand_is_rc_of_fasta` in `test_rc_packing.py` is the closest test, but that uses `RefDataset.with_seqs("reference")` + `splice_info`, not `Dataset.open(...).with_seqs("reference")` + `rc_neg=True` + unspliced output.
- **Jitter + splicing interaction** — `test_ref_ds_splicing.py:test_with_settings_validation` asserts that `jitter=1` + `splice_info` raises `RuntimeError`; the valid `jitter=0` + splice combination is tested, but no test verifies jitter behavior on a `haplotypes`-mode non-spliced dataset with `rc_neg=True`.
- **Jitter range boundary (max_jitter)** — `test_jitter.py` uses `jitter=dataset.max_jitter`; the case where `jitter == 0` explicitly (no jitter applied) is not a separate test.

---

## Summary

| Bucket | Count |
|---|---|
| Delete | 5 |
| Port | 77 |
| Keep-as-integration | 55 |
| **Total** | **137** |

**Count methodology:** parametrized test functions are counted once per test function (not per case), and grouped rows are counted as one row per logical test function.

Breakdown by file:

| File | Delete | Port | Keep |
|---|---|---|---|
| test_fasta.py | 0 | 3 | 0 |
| test_interval_track.py | 1 | 0 | 0 |
| test_ref_ds.py | 1 | 2 | 0 |
| test_ref_ds_splicing.py | 0 | 5 | 4 |
| test_table.py | 0 | 12 | 0 |
| test_utils.py | 0 | 5 | 0 |
| test_build_reconstructor.py | 0 | 12 | 0 |
| test_dataset.py | 1 | 0 | 1 |
| test_ds_haps.py | 0 | 0 | 1 |
| test_ds_haps_1kg.py | 0 | 0 | 1 |
| test_get_splice_bed.py | 0 | 11 | 0 |
| test_indexing.py | 0 | 9 | 0 |
| test_insertion_fill.py | 0 | 13 | 2 |
| test_issue_153.py | 0 | 0 | 1 |
| test_issue_191_var_fields.py | 0 | 5 | 10 |
| test_jitter.py | 0 | 0 | 1 |
| test_open_vs_settings_parity.py | 0 | 0 | 2 |
| test_rc_packing.py | 0 | 1 | 9 |
| test_realign.py | 0 | 1 | 0 |
| test_splice_plan.py | 0 | 3 | 0 |
| test_subset.py | 0 | 0 | 1 |
| test_svar_link.py | 0 | 8 | 12 |
| test_with_settings_var_filter.py | 0 | 0 | 2 |
| test_write_tracks.py | 0 | 1 | 3 |
| test_write.py | 1 | 0 | 3 |
| genotypes/test_choose_exonic_variants.py | 0 | 2 | 0 |
| genotypes/test_filter_af.py | 1 | 0 | 0 |
| genotypes/test_rag_variants.py | 0 | 2 | 0 |
| genotypes/test_reconstruct.py | 0 | 1 | 0 |
| tracks/test_annot_tracks.py | 0 | 0 | 1 |
| tracks/test_i2t_t2i.py | 0 | 2 | 0 |
| tracks/test_random_nonoverlapping.py | 0 | 1 | 0 |
| variants/test_sites.py | 0 | 0 | 1 |
| variants/test_variant_utils.py | 0 | 2 | 0 |
| **Total** | **5** | **77** | **55** |

---

## Recommendations for Phase 4 (delete pass)

Ordered by risk and dependency:

1. **Delete `genotypes/test_filter_af.py` immediately** — the file is entirely `@pytest.mark.skip` with no runnable tests and missing case generators. Zero risk.

2. **Delete `test_dataset.py:test_rs_indexing`** — 147-case cross-product that only checks "doesn't raise"; `test_ds_indexing` (21 cases) plus `test_subset.py` (108 cases) provide strictly stronger coverage. High ROI deletion.

3. **Delete `test_interval_track.py:test_bigwigs_satisfies_interval_track_protocol`** — tautological `hasattr` check against a class we control. Zero risk.

4. **Delete `test_ref_ds.py:test_refdataset_unspliced_defaults`** — asserts fixture-construction defaults, not behavior. Zero risk.

5. **Delete `test_write.py:test_write` (the skipped one)** — dead code already marked `@pytest.mark.skip`; if the test is ever revived it should be rewritten against current data structures.

**Before deleting Port tests:** Port tests should only be deleted from `tests/integration/` after their unit equivalents exist in `tests/unit/`. The highest-priority ports (least dependency on builders) are:

- `test_build_reconstructor.py` — already uses only `Mock`; can be moved verbatim today.
- `test_indexing.py` — uses only synthetic `DatasetIndexer`; can be moved verbatim today.
- `test_splice_plan.py` — synthetic numpy inputs only (except `test_ref_call_with_plan_writes_per_element_layout` which needs the FASTA path fixture); 6 of 8 tests can move today.
- `test_realign.py` and `genotypes/test_reconstruct.py` — direct numba kernel tests; can move today.
- `test_variant_utils.py` — pure path-string predicates; trivial.

**Tests that need a `Dataset` builder before porting:**
- `test_insertion_fill.py:test_with_insertion_fill_rejects_when_no_tracks_active` and `test_dataset.py:test_ds_indexing` require a real `Dataset`; these become unit tests only after `make_dataset` or `make_haps_dataset` builders land.
