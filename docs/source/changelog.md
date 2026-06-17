# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


# Changelog


## v0.32.0 (unreleased)

### Feat

- **write**: `gvl.write` gains `annot_tracks=` — sample-independent annotation tracks written to `<path>/annot_intervals/<name>/`
  - accepts a path to an interval table or bigWig, or a polars DataFrame/LazyFrame with BED-like columns (`chrom`, `chromStart`, `chromEnd`, `score`)
  - DataFrame/LazyFrame and table-file sources use the polars-bio overlap backend and require the `table` extra (`pip install genvarloader[table]`); bigWig path sources do not
- **write**: `gvl.write` parallelizes over write categories — variants run first (serially), then per-sample `tracks` and `annot_tracks` run concurrently (joblib loky); `max_mem` is divided across the concurrently-running categories
- **update**: new `gvl.update(dataset, tracks=None, annot_tracks=None, *, overwrite=False, max_mem="4g")` adds tracks to an existing on-disk dataset
  - accepts a path or an opened `Dataset`; per-sample `tracks` require exact sample-set agreement (reordered to dataset order automatically); `annot_tracks` are sample-independent
  - each track is published atomically; a live dataset can be read during update but won't see the new track until reopened
  - `overwrite=True` replaces a same-named existing track; default raises `FileExistsError`

### Refactor

- **update**: `Dataset.write_annot_tracks` removed; use `gvl.update(dataset, annot_tracks={"name": source})` or pass `annot_tracks=` to `gvl.write` at creation time

## v0.31.0 (2026-06-15)

### Feat

- **table**: ship polars-bio as the `table` extra
- **table**: make gvl.Table an opt-in experimental feature

### Fix

- **write**: raise clear ValueError when variant/track sample intersection is empty (#225)

## v0.30.0 (2026-06-14)

### Feat

- **dataset**: fold haplotypes into ploidy-1 union in variant decode (#222)
- **dataset**: reject haplotypes/annotated output under unphased_union (#222)
- **dataset**: report ploidy=1 and fold n_variants under unphased_union (#222)
- **dataset**: add unphased_union flag on Haps + with_settings (#222)

### Fix

- **dataset**: keep n_variants int32 under unphased_union fold (#222)

### Refactor

- **variants**: drop retired order-dependent germline-CCF inference (#222)

### Perf

- **threads**: tolerate malformed GVL_NUM_THREADS at import (#221)
- **variant-windows**: single fused fetch for both-window decode (#221)
- **flanks**: fuse 3 ref-window fetches into 1 via flank slicing (#221)
- **reference**: dispatch get_reference kernel serial/parallel (#221)
- **reference**: dispatch fetch kernel serial/parallel by per-thread bytes (#221)
- **threads**: cap numba workers to cgroup cores + add dispatch predicate (#221)

## v0.29.0 (2026-06-13)

### Feat

- **flat**: dummy padding for flank_tokens and variant-windows in get_variants_flat
- **flat**: thread unknown_token onto Haps for dummy token fill
- **flat**: fill_empty_groups fills flank_tokens with unknown_token
- **flat**: _FlatVariantWindows.fill_empty_groups (all-unknown_token dummy)
- **flat**: _fill_empty_fixed kernel for flank_tokens empty-row dummy fill
- **flat**: double_buffered carries flat buffers without awkward re-wrap
- **flat**: slice_chunk handles flat containers
- **flat**: shm write/read flat types (kind 1/2/3) without awkward
- **flat**: instance-axis __getitem__ on flat containers
- **flat**: with_settings(dummy_variant=...) + export DummyVariant + non-variant guard
- **flat**: Haps.dummy_variant + apply empty-group fill in get_variants_flat
- **flat**: DummyVariant + empty-group fill kernels + fill_empty_groups
- **flat**: export FlatVariantWindows and VarWindowOpt
- **flat**: VarWindowOpt-driven per-allele variant-windows (window|allele matrix)
- **flat**: VarWindowOpt + per-allele window/allele computation + optional window fields
- **flat**: register variant-windows kind end-to-end
- **flat**: attach flank_tokens / emit windows from get_variants_flat
- **flat**: ref/alt window assembly + tokenization
- **flat**: _FlatWindow + _FlatVariantWindows two-level token types
- **flat**: compute ride-along flank tokens from reference
- **flat**: thread flank_length + token LUT settings onto Haps
- **flat**: byte->int token LUT builder for flank tokenization
- **flat**: A flat variant decode (get_variants_flat) with no awkward
- **flat**: A0 flat passthrough for seqs/haps/annot/reference outputs
- **flat**: output_format field + with_output_format + QueryView.flat_output
- **flat**: export FlatRagged/FlatAnnotatedHaps/FlatVariants/FlatAlleles
- **flat**: _FlatVariants/_FlatAlleles types with to_ragged()

### Fix

- **torch**: reject variant-windows output over buffered transport up front
- **flat**: allow dummy_variant with variant-windows output kind
- **flat**: _fill_empty_seq preserves input dtype for token windows
- **flat**: reject flat+buffered variants with flank tokens at construction
- **flat**: harden _FlatVariants slice for flank_tokens; test polish
- **flat**: with_settings(dummy_variant=False) is a no-op on non-variant datasets
- **flat**: narrow _seqs to Haps before flank-field replace (pyrefly)
- **flat**: clear error for variant-windows + active tracks; drop dead alias
- **flat**: clear error for VarWindowOpt(ref='allele') on REF-less dataset
- **flat**: clearer splice-unsupported messages for variant-windows
- **flat**: windows reshape must keep ploidy dim ([1:-1] not [1:-2])
- **flat**: guard flank-without-LUT, document Haps fields, test validation paths
- **flat**: shape-driven _FlatAlleles.to_ragged fixes scalar-scalar squeeze byte-identity
- **flat**: _FlatAlleles.reshape re-appends ragged axis; docstrings + 2D-index test
- **flat**: add _FlatAlleles.squeeze so _FlatVariants.squeeze works

### Refactor

- **flat**: clarify flat-variant reader names; annotated shm round-trip test
- **flat**: retire awkward _get_variants; ragged variants decode via flat path
- **flat**: normalize _FlatVariants.reshape shape arg; drop redundant import

## v0.28.0 (2026-06-08)

### Feat

- **variants**: numba allele-pack kernel + layout decomposition for non-canonical views
- **write**: reject symbolic/breakend variants from SVAR inputs
- **write**: reject symbolic/breakend variants from PGEN inputs
- **write**: reject symbolic/breakend variants from VCF inputs
- **write**: add consolidated unsupported-variant validator

### Fix

- **variants**: rc_() on sliced/reordered views via materialized copy
- **variants**: to_packed() handles sliced/reordered alt/ref via numba kernel
- **open**: default to variants when genotypes have no reference
- **write**: validate VCF variants before creating output directory

## v0.27.0 (2026-06-05)

### Feat

- **_open**: validate dataset format version + integrity on open
- **_write**: atomic dataset creation + format_version in Metadata
- **_fasta_cache**: publish cache atomically via atomic_dir + locked double-check
- **_atomic**: add atomic_dir directory-publish primitive
- **fasta**: use .gvlfa cache module and accept .gvlfa input
- **_fasta_cache**: add ensure_cache orchestrator and dispatch
- **_fasta_cache**: migrate legacy .gvl caches by reusing bytes
- **_fasta_cache**: add build, load, and validity guards
- **_fasta_cache**: add source hints and three-way resolution
- **_fasta_cache**: add FastaCache models and fingerprint

### Fix

- **torch**: warn when BatchSampler overrides explicit batch_size
- **torch**: buffered modes honor drop_last=False
- **torch**: do not forward drop_last to DataLoader in default mode
- **chunked**: keep trailing partial batch in ChunkPlanner
- **test_fasta**: move mid-file imports to top (E402, CI lint)
- **_fasta_cache**: guard legacy migration against stale/truncated bytes
- **_fasta_cache**: raise on format-too-new sibling cache instead of silent downgrade

### Refactor

- **_write**: use plain with-atomic_dir; restore warnings filter; add atomicity + format_version on-disk tests
- **reference**: build cache via ensure_cache, accept .gvlfa
- **_fasta_cache**: fix progress bar advance and tighten load status type

## v0.26.0 (2026-06-01)

### Feat

- **query**: flat-aware getitem boundary (legacy Ragged still supported)
- **flat**: masked reverse/RC on flat buffers
- **flat**: _Flat numpy ragged transport (no awkward kernels)
- **write**: add _window_to_sparse dense->sparse dispatch helper
- **bench**: log CPU/system/microarch info for all benchmarks
- **bench**: add MiB/s bandwidth plot; trim 1KG regression test data
- **bench**: add 3x4 small-multiples results plot
- **bench**: add bench.py thread-pinned orchestration
- **bench**: add CSV header/append helpers
- **bench**: add per-cell measurement protocol
- **bench**: add exact output-bytes table helper
- **bench**: add BED resize + per-region-length dataset prep
- **bench**: enumerate deduped dataloader bench cells
- **bench**: scaffold dataloader bench axis constants
- mode='double_buffered' dataloader happy path
- **producer**: subprocess entrypoint for double_buffered mode
- **shm**: Ragged and RaggedVariants serialization
- **shm**: hand-rolled slot header + dense round-trip
- mode='buffered' dataloader
- **chunked**: ChunkPlanner and slice_chunk
- **dataset**: _output_bytes_per_instance tracks branch
- **dataset**: _output_bytes_per_instance variants branch with var_fields
- **dataset**: _output_bytes_per_instance annotated branch
- **dataset**: _output_bytes_per_instance reference + haplotypes
- **haps**: add _allele_bytes_sum for exact variant footprint
- **dataset**: with_settings lazily loads new var_fields
- **open**: Dataset.open accepts var_fields, forwards to Haps.from_path
- **haps**: from_path honors var_fields for lazy info+dosage loading
- **haps**: _Variants.load_info lazily extends info dict
- **haps**: _Variants.from_table accepts info_fields filter
- **haps**: add _Variants.available_info_fields schema peek
- **reconstruct**: promote view-state to explicit _seqs_kind field
- **reconstruct**: add _build_reconstructor factory
- **splice**: Tracks._call_float32 accepts SplicePlan
- **splice**: Haps._get_haplotypes accepts SplicePlan
- **splice**: Ref.__call__ accepts optional SplicePlan for zero-copy spliced layout
- **splice**: add SplicePlan + build_splice_plan helper
- **refds**: support with_settings(splice_info) and spliced subset_to
- **refds**: implement spliced __getitem__ via SpliceMap + _cat_length
- **refds**: add splice_info field and is_spliced/spliced_regions
- **splice**: add sample-agnostic SpliceMap

### Fix

- **write**: handle empty regions in no-extend VCF path; assert chunk/index alignment
- **table**: temporarily disable gvl.Table to avoid polars-bio segfault
- **double_buffered**: release producer+shm per loader, not at process exit
- **double_buffered**: serialize RaggedAnnotatedHaps (annotated output)
- **double_buffered**: size shm slots for serialized ragged footprint
- **bench**: open datasets with hg38 reference; revert out-of-scope _open.py change
- **double_buffered**: replay all Dataset settings in producer subprocess
- **sitesonly**: use correct genoray VCF.get_record_info kwargs (fields= not attrs=)
- **ragged**: to_padded numeric branch handles clip=True RegularArray output
- **torch**: forward generator from DataLoader through to RandomSampler for reproducible shuffle
- **dataset**: with_settings(rng=...) maps to _rng dataclass field
- inline offset computation in filter_af 2-D path; reorder test_torch imports
- **rag_variants**: accept positional axis arg in RaggedVariants.squeeze
- **haps**: gate dosage output by var_fields (#191)
- **utils**: copy read-only inputs in idx_like_to_array
- **types**: annotate Fasta.pad as bytes | None
- **Dataset.open**: promote to haplotypes before with_settings(splice_info)
- **choose_exonic_variants**: use (2, n_slices) indexing for 2-D offsets
- **with_settings**: propagate var_filter to _recon, preserving kind

### Refactor

- **variants**: remove orphaned _rc_helper/_rc_numba_helper
- **haps**: extract _build_allele_layout/_alt_layout_parts helpers
- **query**: overloads for reverse_complement_ragged/pad match flat runtime contract
- **flat**: document + guard reverse_masked comp as DNA-RC mode selector
- **write**: assemble full PGEN windows, dispatch via _window_to_sparse, fix max_ends
- **write**: assemble full VCF windows, dispatch via _window_to_sparse, fix max_ends
- trim dead code and over-commenting per CLAUDE.md
- **types**: remove stale type:ignore and annotate the rest
- **naming**: rename SplicePlan.perm -> SplicePlan.permutation
- **naming**: standardize geno_offset_idxs -> geno_offset_idx
- **naming**: rename rsp_idx -> region_sample_ploid_idx
- **dataset**: split _reconstruct.py + extract _query.py via QueryView (PR6)
- **reconstruct**: delete dead body of write_transformed_track + add roadmap
- **reconstruct**: ReconstructionRequest + restructure _get_haplotypes
- **open**: extract OpenRequest + decompose Dataset.open
- **write**: extract shared phased-chunked writer for VCF/PGEN
- **impl**: migrate _impl.py from attrs to stdlib dataclass
- **reconstruct**: migrate _reconstruct.py from attrs to stdlib dataclass
- **reference**: migrate _reference.py from attrs to stdlib dataclass
- **indexing**: migrate _indexing.py from attrs to stdlib dataclass
- **splice**: migrate _splice.py from attrs to stdlib dataclass
- **insertion-fill**: migrate _insertion_fill.py from attrs to stdlib dataclass
- **ragged**: migrate _ragged.py from attrs to stdlib dataclass
- **variants**: migrate _records.py from attrs to stdlib dataclass
- **types**: migrate _types.py from attrs to stdlib dataclass
- **impl**: route remaining _recon construction/checks through view-state
- **impl**: sequence_type returns the _seqs_kind field directly
- **impl**: collapse with_settings _recon propagation via factory
- **impl**: simplify with_tracks via factory + active_tracks check
- **impl**: simplify with_seqs via _seqs_kind + factory
- **impl**: route Dataset.open construction through factory
- **splice**: Dataset spliced path uses SplicePlan
- **splice**: RefDataset spliced path uses SplicePlan
- **splice**: compose SpliceIndexer from SpliceMap + DatasetIndexer
- **splice**: annotate SpliceMap method signatures
- **splice**: clean up _splice imports per review
- **splice**: move _cat_length helpers to _splice module
- **Dataset.open**: delegate splice_info/var_filter to with_settings

### Perf

- **haps**: seqpro to_packed for _allele_bytes_sum gather
- **variants**: field-wise RaggedVariants.to_packed (seqpro + layout rebuild)
- **variants**: flat in-place rc_ via seqpro reverse_complement_masked
- **variants**: seqpro to_packed for allele/genotype/dosage gathers
- **getitem**: remove dead legacy branches + guard awkward out of hot path + A/B write-up
- **variants**: documented awkward-native limit for RaggedVariants (Task 10)
- **splice**: flat-buffer spliced reconstruction + _regroup (Task 9)
- **reconstruct**: _Flat tracks in HapsTracks; RefTracks flat via leaves
- **ref**: return _Flat from reference reconstruction
- **haps**: return _Flat / _FlatAnnotatedHaps from reconstruction
- **tracks**: return _Flat from float32 reconstruction
- **bench**: 10x profiling iterations (N_BATCHES=2000) + sudo py-spy script
- **ragged**: delegate to_padded to seqpro 0.13 flat-buffer kernel
- **rc**: flat-buffer reverse-complement via seqpro 0.12.1
- **bench**: open hg38 reference as memmap (in_memory=False)

## v0.25.0 (2026-05-21)

### Feat

- add migrate_svar_link for upgrading legacy datasets
- **dataset**: wire svar_link resolver into Haps.from_path; Dataset.open(svar=)
- **dataset**: add _resolve_svar and _verify_fingerprint
- **write**: record SvarLink in metadata, drop link.svar symlink
- **dataset**: add SvarLink / SvarFingerprint pydantic models
- **write**: subtract genoray nbytes from max_mem; warn when index dominates

### Fix

- ndim guard on geno_offsets in choose_exonic_variants second loop
- **write**: clarify max_mem docstring and skip index-accounting log for SparseVar
- **write**: eager-load variant index for accurate max_mem accounting

### Refactor

- **dataset**: switch Haps.from_path version compare to SemanticVersion
- **dataset**: use SemanticVersion in Metadata, add svar_link field

## v0.24.1 (2026-05-13)

### Fix

- bump genoray, VCF bug

## v0.24.0 (2026-05-12)

### Feat

- Dataset.with_insertion_fill + public API exports
- route per-track insertion fill into HapsTracks kernel call
- per-track insertion-fill on Tracks reconstructor
- kernel-level insertion-fill strategy dispatch
- add InsertionFill strategy classes and lowering helper

### Fix

- bump genoray
- **insertion-fill**: strengthen tests, Self return type, clearer error
- **insertion-fill**: lazy fallback to Repeat5p for unpopulated insertion_fill
- **insertion-fill**: derive base_seed from full idx array, use full uint64 range
- **insertion-fill**: require params, document fallback, broaden flank-sample tests
- **insertion-fill**: non-instantiable base, tighter MAX_PARAMS, add test coverage

## v0.23.1 (2026-05-11)

### Fix

- **perf**: benchmarks
- **perf**: bench gvl.Table query algs
- **perf**: use single polars-bio overlap (no xprod) in gvl.Table
- types

### Refactor

- **perf**: vectorize scatter, use replace_strict and lexsort in gvl.Table

## 0.23.0 (2026-05-09)

### Feat

- generalize gvl.write() to accept Table tracks
- rename write() param bigwigs= -> tracks=, support mixed sequences
- Table._intervals_from_offsets via polars_bio.overlap
- Table.count_intervals via polars_bio.count_overlaps
- add Table.from_path for csv/tsv/parquet/arrow files
- add Table skeleton with long-form DataFrame init
- add IntervalTrack Protocol for unified track sources
- export get_splice_bed from package root
- add get_splice_bed for GTF→splicing-BED conversion

### Fix

- normalize contig names in Table; correct unavail set warning
- cast chrom and strand to Utf8 in get_splice_bed

### Refactor

- rename _write_bigwigs -> _write_track
- tighten IntervalTrack Protocol annotation and docstring

## 0.22.3 (2026-05-08)

### Fix

- **ci**: exclude _impl.py from debug-statements hook (match syntax)
- **deps**: bump seqpro to 0.11.0 and genoray to 2.3.0
- **_utils**: handle Categorical strand in bed_to_regions

## 0.22.2 (2026-04-28)

### Fix

- make tbb and pyomp optional dependencies

## 0.22.1 (2026-04-22)

### Fix

- skip overlapping variants in get_diffs_sparse to match reconstruction logic

## 0.22.0 (2026-04-20)

### Feat

- SVAR support passes all tests
- SVAR support passes all tests
- add members to conveniently inspect dataset splicing info.
- move indices and transformation to torch dataset/dataloader API since these are generally never needed outside that context. feat: fully functional zero-copy splicing mechanics. fix: bug in rev and rev comp causing garbage output.
- **wip**: testing spliced return values
- initial prototype for splicing.

### Fix

- scope RUSTFLAGS rpath to osx-arm64 only
- support osx-arm64, add contributing instructions
- enable cargo test for PyO3 extension crate
- correct ploidy interleaving in _cat_length
- pack ak.where results in _rc to prevent buffer leak
- make sure exonic filter gets applied. style: adhere to pre-commit
- exons are already in reverse order for negative stranded genes
- virtual indexing for splice indexer
- data corruption when rc_helper is parallelized
- map contig names appropriately for bounds checking on ds regions + ref
- continue migrating to seqpro Ragged, enable logger at module level, add warnings about potential reference genome mismatches
- add spanning dels to test and fix hap ilens for this case
- variant index -> variant info mapping
- add spanning dels to test and fix hap ilens for this case
- parsing splice info and returning single item instead of list
- update spliced_bed in with_settings for splice_info
- __getitem__ type annotations for StrIdx
- spliced i2d_map

## 0.21.4 (2026-03-26)

### Fix

- chained subset_to y str names

## 0.21.3 (2026-03-25)

### Fix

- repeated subset_to

## 0.21.2 (2026-03-17)

### Fix

- repeated subset_to
- bump genoray

## 0.21.1 (2026-03-09)

### Fix

- bump genoray

## 0.21.0 (2026-02-26)

### Feat

- add option to extend lengths or not for gvl.write() fix: contig normalization

### Fix

- add cli as optional dependency

## 0.20.0 (2026-02-05)

### Feat

- support python 3.13
- upgrade to genoray 2

### Fix

- numba weirdness

## 0.19.1 (2025-12-20)

### Fix

- **perf**: upgrade genoray for (much) faster writes. test: idempotent fixutres in test_ds_haps

## 0.19.0 (2025-12-03)

### Feat

- allow subsetting by region name. fix: convert eligible ak.Array to sp.rag.Ragged for gvl.RaggedVariants whenever possible

## 0.18.3 (2025-11-09)

### Fix

- **perf**: faster reverse complementing

## 0.18.2 (2025-11-03)

### Fix

- correctly parse and load variant fields (skip duplicates)
- use ak.str.length instead of ak.num to get ref and alt lengths. docs: shape docstring

## 0.18.1 (2025-10-23)

### Fix

- track file format version

## 0.18.0 (2025-10-22)

### Feat

- make RaggedVariants an Awkward Array subclass supporting arbitrary additional fields.

## 0.17.0 (2025-08-22)

### Feat

- use new seqpro.Ragged interface
- method to obtain the number of variants per region,sample,haplotype.
- support for reverse complementing alt alleles of RaggedVariants
- ragvariant methods for pytorch, fix squeeze for scalar indexing, upstream genoray fix for extending genotypes from PGEN, (perf) batched contig normalization
- subsetting for RefDataset

### Fix

- right shape for double slice indexing
- shape/broadcasting bug for reverse complementing variants
- **perf**: breaking changes to ragged data layout from seqpro to eliminate copies when converting non-contiguous ragged data to/from awkward arrays.
- when ref not passed to Dataset.open, emit a warning instead of error and let ds return raggedvariants
- refdataset transforms with indices

## 0.16.0 (2025-06-05)

### Feat

- make DatasetWithSites return both wild-type and mutant haplotypes

### Fix

- transform not applied when dataset returns single item. docs: add basenji2 evaluation
- use genoray>=0.12. docs: basenji2 eval
- ensure samples are re-ordered by subset_to if necessary
- Let GVL recognize bgz-compressed VCFs
- pad ref_coords with max value for dtype to ensure ref coords are sorted
- remove transform arg for dummy dataset
- finish deprecating the transform setting on Dataset, which was moved to dataloading functionality
- PR #101, ensure variable length output corresponds to ArrayDataset
- update genoray pixi version
- permit ragged output for dataloading, emitting a warning instead of raising an error
- constrain genoray for breaking changes
- torch dataset issues
- bump seqpro to 0.4.2

## 0.15.0 (2025-05-23)

### Feat

- add reference property to Dataset and add path attribute to Reference
- add RefDataset to work with one or more reference genomes. Also change internal indexing to never materialize a full dataset index, dramatically reducing memory usage to support datasets with 1M+ regions. BREAKING CHANGE: move returning indices and transforms to torch API, since these features are generally unnecessary for non-dataloading contexts.
- sites-only changes for QoL. fix: consider output length < region length for sites overlap
- allow tracks to pass-through dataset with sites since SNPs have no affect on them
- **wip**: pad ragged annotated ref coords with max dtype value. pass sanity checks.
- **wip**: change ref_coord annotation so that right-pad values have position MAX_I32
- type-safe Dataset, passes all tests.
- refactor Dataset implementation to be (almost) fully type-safe.
- **wip**: sites-only variants
- **wip**: use sp.bed functions.
- **wip**: small updates
- apply sites-only SNPs, filtering non-SNPs out from VCFs.
- sites-only classes, intersecting them with Datasets, and obtaining information necessary to apply variants.
- add annot track to dummy dataset
- **wip**: initial implementation for read/write annotation tracks, incorporating them along the track dimension.
- deprecate unphased variants
- **wip**: dosages/CCFs on ragged variants
- prototype of returning ragged variants from Dataset

### Fix

- shape of single item from RefDataset
- update init for seqpro bump
- bump seqpro to 0.4.0 which includes basic gtf ops
- update dummy dataset for changes to Reference. docs: add more docstrings
- jittering by folding it into data (re)construction
- contig offset mapping for in-memory reference and incrementing offset when writing cache
- bump genoray version to handle unsorted PVAR contigs
- bump genoray version for filtered PGEN fix
- ensure annot tracks match on-disk ordering
- update for internal breaking changes
- check for SNPs
- bump genoray so bioconda pgenlib is valid
- pass all tests
- internal breaking changes
- treat POS as 1-based to match VCF spec
- type annotations
- type annotation
- contig naming for reference fasta.
- contig normalization
- pass all tests.
- pass tests.
- pass tests.
- Dataset.open returns highest complexity ds by default (haps + all tracks, sorted).
- use pandera polars not pandas
- correct manipulation of active tracks
- dummy dataset
- wrong germline ccfs for 3rd germline variant and beyond
- wrong geno path
- parsing SVAR metadata, bump genoray

### Refactor

- use genoray

## 0.14.4 (2025-05-12)

### Fix

- data corruption when rc_helper is parallelized

## 0.14.3 (2025-05-10)

### Fix

- bump genoray to 0.10.3
- bounds checking on ds regions vs. reference contig lengths

## 0.14.2 (2025-05-08)

### Fix

- unpack tuple for contig names when writing PGEN
- **wip**: handle queries on contigs that do not exist in source variants
- remove print statement gvl write

### Refactor

- remove dead code

## 0.14.1 (2025-05-02)

### Fix

- wrong assumed shapes when checking regions with no variants

## 0.14.0 (2025-05-01)

### Feat

- expose the Reference class and allow it to be passed to Dataset.open to avoid data duplication. feat: begin work for returning variant info
- SVAR support passes all tests
- expose the Reference class and allow it to be passed to Dataset.open to avoid data duplication. feat: begin work for returning variant info

### Fix

- dummy data
- dummy data with Ragged sparsegenotypes
- continue migrating to seqpro Ragged, enable logger at module level, add warnings about potential reference genome mismatches
- add spanning dels to test and fix hap ilens for this case
- dummy data
- dummy data with Ragged sparsegenotypes

## 0.13.0 (2025-04-30)

### Feat

- SVAR support passes all tests

### Fix

- continue migrating to seqpro Ragged, enable logger at module level, add warnings about potential reference genome mismatches
- add spanning dels to test and fix hap ilens for this case

## 0.12.0 (2025-04-18)

### Feat

- remove Variants from public API. lets variants be a path and automatically infer if it is a VCF or PGEN
- automatically write or update genoray VCF index to be GVL compatible during write
- use genoray for variant I/O

## 0.11.0 (2025-04-09)

### Feat

- change format of variant file indices so they can be one file, change naming to add .gvi extension to source.

### Refactor

- let starts and ends be optional for variant queries

## 0.10.8 (2025-04-08)

### Fix

- reverse helper wrong when parallel=True
- reverse helper wrong when parallel=True

## 0.10.7 (2025-04-04)

### Fix

- pass all tests.

## 0.10.6 (2025-04-04)

### Fix

- bug in rev and rev comp causing garbage output for negative stranded data.

## 0.10.5 (2025-04-01)

### Fix

- (breaks experimental API) rename dosage to CCF and adjust alg to handle missing CCF from e.g. germline variants.

## 0.10.4 (2025-04-01)

### Fix

- hotfix for indexer usage

## 0.10.3 (2025-04-01)

### Fix

- shape of dataset return values.

## 0.10.2 (2025-04-01)

### Fix

- pass relevant tests.
- faster VCF reading.

## 0.10.1 (2025-03-25)

### Fix

- bump polars version, breaking changes upstream.

## 0.10.0 (2025-03-21)

### Feat

- type-safe Dataset, passes all tests.
- refactor Dataset implementation to be (almost) fully type-safe.

### Fix

- Dataset.open returns highest complexity ds by default (haps + all tracks, sorted).

## 0.9.0 (2025-03-06)

### Feat

- option to return ragged data from gvl.Dataset. output_length is set dynamically. fix: hap reconstruction matches bcftools. change default for Dataset.deterministic from False to True. change track output from a list of arrays to having a track dimension i.e. from shape (b [p] l) to (b t [p] l). docs: add dataset.md, faq.md and overhaul geuvadis.ipynb to be simpler and reflect changes in API.

## 0.8.1 (2025-02-12)

### Fix

- incorrect mask from get_keep_mask_for_length (#37)

## 0.8.0 (2025-02-05)

### Feat

- sequence annotations

## v0.7.3 (2025-01-27)

### Feat

- allow subset_to() to accept boolean masks and polars Series
- allow subset_to() to accept boolean masks and polars Series

### Fix

- add test for subset_to
- add test for subset_to
- update tests to match internal API changes
- update tests to match internal API changes
- bug in mark_keep_variants with spanning deletions.

## v0.7.2 (2025-01-26)

### Fix

- change loop order to only open files once.
- respect memory limits when writing bigwig data.
- online docs notebook syntax highlightning
- better docs.

## v0.7.1 (2025-01-17)

### Fix

- bump version
- scalar dataset indexing, region_indices order, updated docs, hotfixes

## v0.7.0 (2025-01-17)

### Feat

- indexing matches input bed file. make sel() a private method pending better API design. fix: pass tests for indexing and separate indexing and subsetting logic into DatasetIndexer.
- write input regions to disk with a column mapping each to a row in the sorted dataset regions

### Fix

- passing tests
- passing tests
- passing tests

## v0.6.4 (2024-12-16)

### Fix

- update rust dependencies.

## v0.6.3 (2024-12-16)

### Fix

- unintended torch requirements

## v0.6.2 (2024-12-16)

### Fix

- update version
- StratifiedSampler requires torch. fix: remove deprecated conda env files.

## v0.6.1 (2024-11-25)

### Fix

- handle empty genotypes during gvl write. fix: PgenGenos sample_idx should be sorted when compared to current sample_idx.

## v0.6.0 (2024-09-03)

### Feat

- bump version
- geuvadis tutorial.
- tutorial notebook, pooch dependency.

### Fix

- update available tracks after writing transformed ones to disk.

## v0.5.6 (2024-08-07)

### Fix

- bump version
- offsets can overflow int32, use int64 instead.

## v0.5.5 (2024-08-02)

### Fix

- make Records.vars_in_range... functions fallible by returning None instead of "empty" RecordInfo instances. This fixes downstream behavior of the Variants.read.. methods when there are no variants in the query. feat: when reading VCFs for the first time and no index is found, try to index them first before raising an error. fix: better docstrings on attributes of private API.
- add build number to replace yanked release

## v0.5.4 (2024-07-05)

### Fix

- fix breaking changes from polars 1.0
- fix breaking changes from polars 1.0

## v0.5.3 (2024-07-05)

### Fix

- fix breaking changes from polars 1.0
- fix breaking changes from polars 1.0

## v0.5.2 (2024-07-05)

### Fix

- typo in pyproject causing dependencies to be ignored.
- typo in pyproject causing dependencies to be ignored.

## v0.5.1 (2024-06-29)

### Feat

- prep for readthedocs
- prepare for online documentation.

### Fix

- add favicon
- documentation formatting
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- readthedocs dependencies
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config

## v0.5.0 (2024-06-13)

### Feat

- bump version
- multiprocess reading of genotypes, both VCF and PGEN. fix: bug in reading genotypes from PGEN

## v0.4.1 (2024-06-11)

### Fix

- bump version
- got number of regions from wrong array in get_reference

## v0.4.0 (2024-06-05)

### Feat

- deprecate old loader, worse performance. reorganize code.

### Fix

- better documentation in README. feat!: rename write_transformed_tracks to write_transformed_track. feat: more ergonomic indexing.

## v0.3.3 (2024-06-01)

### Fix

- bump version
- wrong max_ends from SparseGenotypes.from_dense_with_length due to data races/incorrect parallel semantics for numba
- diffs need to be clipped and negated when computing shifts

### Perf

- pad haplotypes on-the-fly to avoid extra copying of reference subsequences

## v0.3.2 (2024-04-29)

### Feat

- can convert Records back to a polars DataFrame with minimal copying via conversion of VLenAlleles to pyarrow buffers
- make open_with_settings the standard open function. fix: recognize .bgz extension for fasta files

### Fix

- remove dynamic versioning table
- move cli to __main__ feat: generalize Variants to automatically identify whether vcf or pgen is passed
- move cli to script in python source directory, maturin limitation?
- wrong implementation of heuristic for extending genotypes.

### Perf

- faster sparsifying genotypes. feat: log level for cli. fix: clip missing lengths for appropriate end extension.

## v0.3.1 (2024-04-16)

### Feat

- benchmark interval decompression on cpu with numba vs. cpu with taichi vs. gpu with taichi
- optionally decompress intervals to tracks on gpu
- initial support for stranded regions
- option to cache fasta files as numpy arrays.
- implement BigWig intervals as Rust extension.
- finishing touches on multi-track implementation. Block is cryptic issue where writing genotypes is somehow preventing joblib from launching new processes.
- stop overwriting by default, add option.
- transforms directly on tracks. feat: intervals as array of structs for better data locality.
- let extra tracks get added via paths
- let extra tracks get added via paths
- initial support for indels in tracks and WIP on also returning auxiliary genome wide tracks.
- initial sparse genos -> haplotypes and sparse hap diffs.
- wip sparse genotypes.
- properties for getting haplotypes, references, or tracks only.
- properties for getting haplotypes, references, or tracks only.
- encourage num_workers <= 1 with GVL dataloader.
- freeze gvl.Dataset to prevent user from accidentally introducing invalid states. feat: warn if any query contigs have either no variatns or intervals associated with them.
- warn instead of error when no reference passed and genos present.
- disable overwriting by default, have no args be help.
- also report number of samples.
- add .from_table constructor for BigWigs.
- move CLI to script, include in package.
- use a table to specify bigwigs instead. fix: jittering.
- add script to write datasets to disk.
- more quality of life improvements. relax dependency version constraints.
- with_seed method
- quality of life methods for subsetting and converting to dataloaders.
- torch convenience functions fix: ensure genotypes and intervals written in sorted order wrt the BED file.
- pre-computed implementation.

### Fix

- dependency typo
- remove taichi interval to track implementation since it did not improve performance, even on GPU
- need to subset arrays to be reverse complemented
- change argument order of subset_to to match the rest of the API. fix: simplify subset implementation.
- remove python 3.10 type hints
- dimension order on subsets.
- make variant indices absolute on write.
- sparse genotypes layout
- sparse genotypes layout
- wrong layout out genotypes and wrong max ends computation.
- ragged array layouts for correct concatenation when writing datasets one contig at a time.
- bug where init_intervals would not initialize all available tracks.
- track_to_intervals had wrong n_intervals and thus, wrong offsets.
- track_to_intervals had wrong n_intervals and thus, wrong offsets.
- bug in computing max ends.
- match serde for genome tracks.
- bug in open state management.
- bug when writing genotypes where the chromosome of the requested regions is not present in the VCF.
- bug getting intersection of samples available.
- bug getting intersection of samples available.
- sum wrong axis in adjust multi index.
- make GVLDataset __getitem__ API  match torch Dataset API (i.e. use raveled index)
- QOL improvements.
- incorrect genotypes returned from VCF when queries have overlapping ranges.
- wrong shape.
- wrong shape.

### Refactor

- move construct virtual data to loader so utils import faster.
- move construct virtual data to loader so utils import faster.
- rename util to utils.
- rename util to utils.
- move write under dataset directory. perf?: move indexing operations into numba.
- move cli to script outside package, faster help message.
- break up dataset implementation into smaller files. refactor!: condense with_ methods into single with_settings() methods. feat: sel() and isel() methods for eager retrieval by sample and region.

### Perf

- when opening witih settings and providing a reference, but return_sequences is false, don't load the reference into memory.

## v0.3.0 (2024-03-15)

### Feat

- write ZarrTracks in smaller chunks.
- write ZarrTracks in smaller chunks.

### Fix

- remove wip vidx feature.
- relax numba version constraint
- rounding issues for setting fixed lengths on BED regions.
- more informative vcf record progress bar.

## v0.3.0rc6 (2024-03-11)

### Feat

- improve record query performance by allowing nearest_nonoverlapping index adjustment to be computed on-the-fly in the weighted activity selection algorithm and thus also benefit from early stopping.
- more descriptive progress bar for constructing ZarrGenos from another file.
- add progress bar for reading VCF records.

### Fix

- pylance update, catch possibly unbound variables.
- instead of failing, raise warning when encountering non-SNP, non-INDEL variants and skip them.

## v0.3.0rc5 (2024-03-04)

### Fix

- more descriptive pbar when writing ZarrTracks from another reader.
- BigWigs, only keep contigs that are shared across all bigwigs.
- better error messages and catching cases for non-SNP, non-INDEL variants.
- avoid segfault caused when a TensorStore is forked to new processes.
- make ZarrTracks implement Reader protocol. feat: add NumpyGenos for in-memory representation. feat: better ZarrGenos.from_recs_genos progress bar.

## v0.3.0rc4 (2024-02-29)

### Fix

- naming of .ends.gvl.arrow to .gvl.ends.arrow so file suffix parsing works correctly.

## v0.3.0rc3 (2024-02-29)

## v0.3.0rc2 (2024-02-29)

### Fix

- remove pyd4 dependency, had unspectacular performance.

## v0.3.0-rc.1 (2024-02-28)

### Feat

- add ZarrTracks for much faster performance than D4.
- finish deprecating parallel GVL.

### Fix

- implementation of Haplotypes with re-alignment of tracks, no runtime errors. Pending unit tests.
- implementation of Haplotypes with re-alignment of tracks, no runtime errors. Pending unit tests.
- deprecate vcf, tiledb, and zarr readers and associated types.

## v0.2.5 (2024-02-26)

### Fix

- raise informative error for unnormalized VCFs.
- remove print statement and add zarr dependency.

## v0.2.4 (2024-02-25)

### Fix

- VCFGenos needed offset information.
- bug in construct_haplotypes, updated shift too early. feat: support for VCF and Zarr/Tensorstore for parallel access to genotypes.

## v0.2.3 (2024-02-12)

### Fix

- wrong syntax with in_memory FASTA and cast queries to 1d for FastaVariants.

## v0.2.2 (2024-02-07)

## v0.2.1 (2024-02-02)

### Fix

- update seqpro version.
- update seqpro version.

## v0.2.0 (2023-12-29)

### Feat

- rename RLE table to intervals. WIP generalizing to arbitrary groupings and value columns.
- rename RLE table to intervals. WIP generalizing to arbitrary groupings and value columns.

### Fix

- include tqdm dependency
- non-contiguous genotypes array when using multiple regions in pgen.
- update tests to use relative paths and reflect changes to API
- handle contig normalization when file has mixed contigs prefixes.
- bug in Buffer class
- bug in Buffer class
- infer contig prefix for FASTA.
- return NDArray from Reader instead of DataArray for greater portability and performance.
- only slice batch_dims when fetching buffers.
- faster FASTA reads, skip sequences that aren't needed.
- broadcast track to have ploid dimension.

## v0.1.18 (2023-12-20)

### Feat

- **api**: remove jitter_bed from GVL.set and GVLDataset.set

### Fix

- big speed up of GVL initialization, especially when BED file has many unique contigs.
- make jitter_bed jitter each instance independently.
- speed up partition_bed when # of unique contigs is very large by parallelizing execution across contigs.

## v0.1.17 (2023-12-17)

### Feat

- pass all tests with weighted activity selection implementation.
- lazily open Fasta file handle, keeping it open after first read.
- lazily open Fasta file handle, keeping it open after first read.
- generic type annotation for random_chain.
- automatically infer torch DDP usage.
- add transform to map-style dataset. fix: remove region from GVL.sizes, should only have batch_dims excluding region.
- better protocol typing for Reader. feat: random_chain utility function to facilitate randomly chaining GVL loaders.
- experimental map-style torch dataset.
- support torch DDP by specifying distributed framework to GVL. fix: work-in-progress on proper max_end calculation.
- minimum batch dim sizes when shuffle=True (i.e. for training). feat: parallel processing of query regions in pgen and fastavariants. fix: compute max deletion lengths with weighted activity selection, remark on intractable aspects of problem and when heuristic fails. Handle failure in construct haplotypes function. feat[wip]: optionally converting PGEN genotypes to an N5 store, currently segfaults for unknown reasons. Gets further with longer sleep cycles. feat: add chunked attribute to readers so that GVL can attempt to respect chunked layouts. fix: negative indices when slicing VLenAlleles. feat: concat VLenAlleles.

### Fix

- reset partition counters on iteration start.
- randomly sample keys in random_chain.

## v0.1.16 (2023-11-28)

### Fix

- batch dimension can end up in wrong axis after vectorized indexing depending on batch_dims.

## v0.1.15 (2023-11-20)

### Fix

- in_memory Fasta holds wrong data.

## v0.1.14 (2023-11-20)

## v0.1.13 (2023-11-17)

## 0.1.12 (2023-11-16)

## v0.1.11 (2023-11-14)

## v0.1.9 (2023-11-06)

### Feat

- do not pre-compute diffs since it is only used once and this reduces memory usage.
- initial implementation of generalized haplotype construction for re-aligning tracks.
- switch buffer slicing alg depending on batch size.
- change type annotation to Mapping for covariance.
- better docstring.
- add method to get PyTorch dataset from GVL class.

### Fix

- better docstring.
- return region index with return_index. feat: specify order of arrays in return_tuples.
- increment buffer idx_slice by amount actually copied from buffer instead of batch_size, which is sometimes too large.
- readers with no non-length dimensions. feat: allow Fasta to be in-memory.

## v0.1.10 (2023-11-08)

### Feat

- improve speed for short seq lengths, large batch sizes. fix: bug in reverse complementing.

### Fix

- not all batch dims need be in every loader, handle this case.
- work-in-progress on wrong output lengths from fasta_variants.
- buffer length axis slicer was wrong length, should be computed by total length of *merged* regions.
- buffer length axis slicer was wrong length, should be computed by total length of regions.
- splice utils.
- attribute error.
- comment why versions are 0.0.0.
- dynamic versioning config.

## 0.1.9 (2023-11-06)

### Feat

- update README.

### Fix

- relax virtual data alignment constraints from exact join to inner join. feat: use dynamic versioning.
- make splicing util funcs generalize to n-dim arrays with length as the final axis.

## 0.1.7 (2023-11-03)

### Fix

- improve perf by not having batches as xr.Datasets.

## 0.1.6 (2023-10-31)

### Fix

- partitioned the wrong bed, jittered bed at wrong point in loop.
- forgot to cache jit function.

## 0.1.5 (2023-10-31)

### Fix

- batch_idx generation.

## 0.1.4 (2023-10-31)

### Fix

- computing relative starts for slicing buffers.

## 0.1.3 (2023-10-31)

### Fix

- computing max_ends.

## 0.14.2 (2025-05-07)

### Fix

- **wip**: handle queries on contigs that do not exist in source variants
- remove print statement gvl write

### Refactor

- remove dead code

## 0.14.1 (2025-05-02)

### Fix

- wrong assumed shapes when checking regions with no variants

## 0.14.0 (2025-05-01)

### Feat

- expose the Reference class and allow it to be passed to Dataset.open to avoid data duplication. feat: begin work for returning variant info
- SVAR support passes all tests

### Fix

- dummy data
- dummy data with Ragged sparsegenotypes
- continue migrating to seqpro Ragged, enable logger at module level, add warnings about potential reference genome mismatches
- add spanning dels to test and fix hap ilens for this case

## v0.13.0 (2025-04-30)

### Feat

- SVAR support passes all tests

### Fix

- continue migrating to seqpro Ragged, enable logger at module level, add warnings about potential reference genome mismatches
- add spanning dels to test and fix hap ilens for this case

## v0.12.0 (2025-04-18)

### Feat

- remove Variants from public API. lets variants be a path and automatically infer if it is a VCF or PGEN
- automatically write or update genoray VCF index to be GVL compatible during write
- use genoray for variant I/O

## v0.11.0 (2025-04-09)

### Feat

- change format of variant file indices so they can be one file, change naming to add .gvi extension to source.

### Refactor

- let starts and ends be optional for variant queries

## v0.10.8 (2025-04-08)

### Fix

- reverse helper wrong when parallel=True
- reverse helper wrong when parallel=True

## v0.10.7 (2025-04-04)

### Fix

- pass all tests.

## v0.10.6 (2025-04-04)

### Fix

- bug in rev and rev comp causing garbage output for negative stranded data.

## v0.10.5 (2025-04-01)

### Fix

- (breaks experimental API) rename dosage to CCF and adjust alg to handle missing CCF from e.g. germline variants.

## v0.10.4 (2025-04-01)

### Fix

- hotfix for indexer usage

## v0.10.3 (2025-04-01)

### Fix

- shape of dataset return values.

## v0.10.2 (2025-04-01)

### Fix

- pass relevant tests.
- faster VCF reading.

## v0.10.1 (2025-03-25)

### Fix

- bump polars version, breaking changes upstream.

## v0.10.0 (2025-03-21)

### Feat

- type-safe Dataset, passes all tests.
- refactor Dataset implementation to be (almost) fully type-safe.

### Fix

- Dataset.open returns highest complexity ds by default (haps + all tracks, sorted).

## v0.9.0 (2025-03-06)

This is a breaking change for GVL. Users should view the ["What's a `gvl.Dataset`?"](https://genvarloader.readthedocs.io/en/latest/dataset.html) page in the documentation for details, but major breaks include:

- removed the `length` argument from `gvl.write()`. Regions/BED files are now used as-is. If you want uniform length regions centered on inputs/peaks as before, preprocess your BED file with `gvl.with_length`.
- changed `Dataset.output_length` from a property to a dynamic setting with behavior describe in the "What's a gvl.Dataset?" page.
- changed track output shape to have a track axis.
- Datasets are now deterministic by default.

As a result of these changes, GVL seamlessly supports ragged length output and also paves the way for on-the-fly splicing. Since many changes were made, I wouldn't be surprised if a few bugs crop up despite my best efforts -- please leave issues if so!

### Feat

- option to return ragged data from gvl.Dataset. output_length is set dynamically. fix: hap reconstruction matches bcftools. change default for Dataset.deterministic from False to True. change track output from a list of arrays to having a track dimension i.e. from shape (b [p] l) to (b t [p] l). docs: add dataset.md, faq.md and overhaul geuvadis.ipynb to be simpler and reflect changes in API.

## v0.8.1 (2025-02-12)

### Fix

- incorrect mask from get_keep_mask_for_length (#37)

## v0.8.0 (2025-02-05)

### Feat

- sequence annotations

## v0.7.3 (2025-01-27)

### Feat

- allow subset_to() to accept boolean masks and polars Series
- allow subset_to() to accept boolean masks and polars Series

### Fix

- add test for subset_to
- add test for subset_to
- update tests to match internal API changes
- update tests to match internal API changes
- bug in mark_keep_variants with spanning deletions.

## v0.7.2 (2025-01-26)

### Fix

- change loop order to only open files once.
- respect memory limits when writing bigwig data.
- online docs notebook syntax highlightning
- better docs.

## v0.7.1 (2025-01-17)

### Fix

- bump version
- scalar dataset indexing, region_indices order, updated docs, hotfixes

## v0.7.0 (2025-01-17)

### Feat

- indexing matches input bed file. make sel() a private method pending better API design. fix: pass tests for indexing and separate indexing and subsetting logic into DatasetIndexer.
- write input regions to disk with a column mapping each to a row in the sorted dataset regions

### Fix

- passing tests
- passing tests
- passing tests

## v0.6.4 (2024-12-16)

### Fix

- update rust dependencies.

## v0.6.3 (2024-12-16)

### Fix

- unintended torch requirements

## v0.6.2 (2024-12-16)

### Fix

- update version
- StratifiedSampler requires torch. fix: remove deprecated conda env files.

## v0.6.1 (2024-11-25)

### Fix

- handle empty genotypes during gvl write. fix: PgenGenos sample_idx should be sorted when compared to current sample_idx.

## v0.6.0 (2024-09-03)

### Feat

- bump version
- geuvadis tutorial.
- tutorial notebook, pooch dependency.

### Fix

- update available tracks after writing transformed ones to disk.

## v0.5.6 (2024-08-07)

### Fix

- bump version
- offsets can overflow int32, use int64 instead.

## v0.5.5 (2024-08-02)

### Fix

- make Records.vars_in_range... functions fallible by returning None instead of "empty" RecordInfo instances. This fixes downstream behavior of the Variants.read.. methods when there are no variants in the query. feat: when reading VCFs for the first time and no index is found, try to index them first before raising an error. fix: better docstrings on attributes of private API.
- add build number to replace yanked release

## v0.5.4 (2024-07-05)

### Fix

- fix breaking changes from polars 1.0
- fix breaking changes from polars 1.0

## v0.5.3 (2024-07-05)

### Fix

- fix breaking changes from polars 1.0
- fix breaking changes from polars 1.0

## v0.5.2 (2024-07-05)

### Fix

- typo in pyproject causing dependencies to be ignored.
- typo in pyproject causing dependencies to be ignored.

## v0.5.1 (2024-06-29)

### Feat

- prep for readthedocs
- prepare for online documentation.

### Fix

- add favicon
- documentation formatting
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- rtd config
- readthedocs dependencies
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config
- readthedocs config

## v0.5.0 (2024-06-13)

### Feat

- bump version
- multiprocess reading of genotypes, both VCF and PGEN. fix: bug in reading genotypes from PGEN

## v0.4.1 (2024-06-11)

### Fix

- bump version
- got number of regions from wrong array in get_reference

## v0.4.0 (2024-06-05)

### Feat

- deprecate old loader, worse performance. reorganize code.

### Fix

- better documentation in README. feat!: rename write_transformed_tracks to write_transformed_track. feat: more ergonomic indexing.

## v0.3.3 (2024-06-01)

### Fix

- bump version
- wrong max_ends from SparseGenotypes.from_dense_with_length due to data races/incorrect parallel semantics for numba
- diffs need to be clipped and negated when computing shifts

### Perf

- pad haplotypes on-the-fly to avoid extra copying of reference subsequences

## v0.3.2 (2024-04-29)

### Feat

- can convert Records back to a polars DataFrame with minimal copying via conversion of VLenAlleles to pyarrow buffers
- make open_with_settings the standard open function. fix: recognize .bgz extension for fasta files

### Fix

- remove dynamic versioning table
- move cli to __main__ feat: generalize Variants to automatically identify whether vcf or pgen is passed
- move cli to script in python source directory, maturin limitation?
- wrong implementation of heuristic for extending genotypes.

### Perf

- faster sparsifying genotypes. feat: log level for cli. fix: clip missing lengths for appropriate end extension.

## v0.3.1 (2024-04-16)

### Feat

- benchmark interval decompression on cpu with numba vs. cpu with taichi vs. gpu with taichi
- optionally decompress intervals to tracks on gpu
- initial support for stranded regions
- option to cache fasta files as numpy arrays.
- implement BigWig intervals as Rust extension.
- finishing touches on multi-track implementation. Block is cryptic issue where writing genotypes is somehow preventing joblib from launching new processes.
- stop overwriting by default, add option.
- transforms directly on tracks. feat: intervals as array of structs for better data locality.
- let extra tracks get added via paths
- let extra tracks get added via paths
- initial support for indels in tracks and WIP on also returning auxiliary genome wide tracks.
- initial sparse genos -> haplotypes and sparse hap diffs.
- wip sparse genotypes.
- properties for getting haplotypes, references, or tracks only.
- properties for getting haplotypes, references, or tracks only.
- encourage num_workers <= 1 with GVL dataloader.
- freeze gvl.Dataset to prevent user from accidentally introducing invalid states. feat: warn if any query contigs have either no variatns or intervals associated with them.
- warn instead of error when no reference passed and genos present.
- disable overwriting by default, have no args be help.
- also report number of samples.
- add .from_table constructor for BigWigs.
- move CLI to script, include in package.
- use a table to specify bigwigs instead. fix: jittering.
- add script to write datasets to disk.
- more quality of life improvements. relax dependency version constraints.
- with_seed method
- quality of life methods for subsetting and converting to dataloaders.
- torch convenience functions fix: ensure genotypes and intervals written in sorted order wrt the BED file.
- pre-computed implementation.

### Fix

- dependency typo
- remove taichi interval to track implementation since it did not improve performance, even on GPU
- need to subset arrays to be reverse complemented
- change argument order of subset_to to match the rest of the API. fix: simplify subset implementation.
- remove python 3.10 type hints
- dimension order on subsets.
- make variant indices absolute on write.
- sparse genotypes layout
- sparse genotypes layout
- wrong layout out genotypes and wrong max ends computation.
- ragged array layouts for correct concatenation when writing datasets one contig at a time.
- bug where init_intervals would not initialize all available tracks.
- track_to_intervals had wrong n_intervals and thus, wrong offsets.
- track_to_intervals had wrong n_intervals and thus, wrong offsets.
- bug in computing max ends.
- match serde for genome tracks.
- bug in open state management.
- bug when writing genotypes where the chromosome of the requested regions is not present in the VCF.
- bug getting intersection of samples available.
- bug getting intersection of samples available.
- sum wrong axis in adjust multi index.
- make GVLDataset __getitem__ API  match torch Dataset API (i.e. use raveled index)
- QOL improvements.
- incorrect genotypes returned from VCF when queries have overlapping ranges.
- wrong shape.
- wrong shape.

### Refactor

- move construct virtual data to loader so utils import faster.
- move construct virtual data to loader so utils import faster.
- rename util to utils.
- rename util to utils.
- move write under dataset directory. perf?: move indexing operations into numba.
- move cli to script outside package, faster help message.
- break up dataset implementation into smaller files. refactor!: condense with_ methods into single with_settings() methods. feat: sel() and isel() methods for eager retrieval by sample and region.

### Perf

- when opening witih settings and providing a reference, but return_sequences is false, don't load the reference into memory.

## v0.3.0 (2024-03-15)

### Feat

- write ZarrTracks in smaller chunks.
- write ZarrTracks in smaller chunks.

### Fix

- remove wip vidx feature.
- relax numba version constraint
- rounding issues for setting fixed lengths on BED regions.
- more informative vcf record progress bar.

## v0.3.0rc6 (2024-03-11)

### Feat

- improve record query performance by allowing nearest_nonoverlapping index adjustment to be computed on-the-fly in the weighted activity selection algorithm and thus also benefit from early stopping.
- more descriptive progress bar for constructing ZarrGenos from another file.
- add progress bar for reading VCF records.

### Fix

- pylance update, catch possibly unbound variables.
- instead of failing, raise warning when encountering non-SNP, non-INDEL variants and skip them.

## v0.3.0rc5 (2024-03-04)

### Fix

- more descriptive pbar when writing ZarrTracks from another reader.
- BigWigs, only keep contigs that are shared across all bigwigs.
- better error messages and catching cases for non-SNP, non-INDEL variants.
- avoid segfault caused when a TensorStore is forked to new processes.
- make ZarrTracks implement Reader protocol. feat: add NumpyGenos for in-memory representation. feat: better ZarrGenos.from_recs_genos progress bar.

## v0.3.0rc4 (2024-02-29)

### Fix

- naming of .ends.gvl.arrow to .gvl.ends.arrow so file suffix parsing works correctly.

## v0.3.0rc3 (2024-02-29)

## v0.3.0rc2 (2024-02-29)

### Fix

- remove pyd4 dependency, had unspectacular performance.

## v0.3.0-rc.1 (2024-02-28)

### Feat

- add ZarrTracks for much faster performance than D4.
- finish deprecating parallel GVL.

### Fix

- implementation of Haplotypes with re-alignment of tracks, no runtime errors. Pending unit tests.
- implementation of Haplotypes with re-alignment of tracks, no runtime errors. Pending unit tests.
- deprecate vcf, tiledb, and zarr readers and associated types.

## v0.2.5 (2024-02-26)

### Fix

- raise informative error for unnormalized VCFs.
- remove print statement and add zarr dependency.
- VCFGenos needed offset information.
- bug in construct_haplotypes, updated shift too early. feat: support for VCF and Zarr/Tensorstore for parallel access to genotypes.

## v0.2.3 (2024-02-12)

### Fix

- wrong syntax with in_memory FASTA and cast queries to 1d for FastaVariants.

## v0.2.2 (2024-02-07)

## v0.2.1 (2024-02-02)

### Fix

- update seqpro version.
- update seqpro version.

## v0.2.0 (2023-12-29)

### Feat

- rename RLE table to intervals. WIP generalizing to arbitrary groupings and value columns.
- rename RLE table to intervals. WIP generalizing to arbitrary groupings and value columns.

### Fix

- include tqdm dependency
- non-contiguous genotypes array when using multiple regions in pgen.
- update tests to use relative paths and reflect changes to API
- handle contig normalization when file has mixed contigs prefixes.
- bug in Buffer class
- bug in Buffer class
- infer contig prefix for FASTA.
- return NDArray from Reader instead of DataArray for greater portability and performance.
- only slice batch_dims when fetching buffers.
- faster FASTA reads, skip sequences that aren't needed.
- broadcast track to have ploid dimension.

## v0.1.18 (2023-12-20)

### Feat

- **api**: remove jitter_bed from GVL.set and GVLDataset.set

### Fix

- big speed up of GVL initialization, especially when BED file has many unique contigs.
- make jitter_bed jitter each instance independently.
- speed up partition_bed when # of unique contigs is very large by parallelizing execution across contigs.

## v0.1.17 (2023-12-17)

### Feat

- pass all tests with weighted activity selection implementation.
- lazily open Fasta file handle, keeping it open after first read.
- lazily open Fasta file handle, keeping it open after first read.
- generic type annotation for random_chain.
- automatically infer torch DDP usage.
- add transform to map-style dataset. fix: remove region from GVL.sizes, should only have batch_dims excluding region.
- better protocol typing for Reader. feat: random_chain utility function to facilitate randomly chaining GVL loaders.
- experimental map-style torch dataset.
- support torch DDP by specifying distributed framework to GVL. fix: work-in-progress on proper max_end calculation.
- minimum batch dim sizes when shuffle=True (i.e. for training). feat: parallel processing of query regions in pgen and fastavariants. fix: compute max deletion lengths with weighted activity selection, remark on intractable aspects of problem and when heuristic fails. Handle failure in construct haplotypes function. feat[wip]: optionally converting PGEN genotypes to an N5 store, currently segfaults for unknown reasons. Gets further with longer sleep cycles. feat: add chunked attribute to readers so that GVL can attempt to respect chunked layouts. fix: negative indices when slicing VLenAlleles. feat: concat VLenAlleles.

### Fix

- reset partition counters on iteration start.
- randomly sample keys in random_chain.

## v0.1.16 (2023-11-28)

### Fix

- batch dimension can end up in wrong axis after vectorized indexing depending on batch_dims.

## v0.1.15 (2023-11-20)

### Fix

- in_memory Fasta holds wrong data.

## v0.1.14 (2023-11-20)

## v0.1.13 (2023-11-17)

## v0.1.12 (2023-11-16)

## v0.1.11 (2023-11-14)

## v0.1.9 (2023-11-06)

### Feat

- do not pre-compute diffs since it is only used once and this reduces memory usage.
- initial implementation of generalized haplotype construction for re-aligning tracks.
- switch buffer slicing alg depending on batch size.
- change type annotation to Mapping for covariance.
- better docstring.
- add method to get PyTorch dataset from GVL class.

### Fix

- better docstring.
- return region index with return_index. feat: specify order of arrays in return_tuples.
- increment buffer idx_slice by amount actually copied from buffer instead of batch_size, which is sometimes too large.
- readers with no non-length dimensions. feat: allow Fasta to be in-memory.

## v0.1.10 (2023-11-08)

### Feat

- improve speed for short seq lengths, large batch sizes. fix: bug in reverse complementing.
- update README.
- bump version.
- build 0.1.1
- draft support for PGEN split by contig.
- bump version to 0.1.0
- tests for indel support.
- option to disable jittering of haplotypes that are longer than query regions.
- implement stranded regions.
- add option to jitter bed regions.
- passing strand info to read(), 1 for forward and -1 for reverse.
- initial (buggy) implementation drafting spliced, multiregion `read` functions.
- split non-overlapping ROIs into separate partitions.
- organize kwargs docstring.
- make SyncBuffer work. Several errors in construct haplotypes with indels.
- enable make concurrent reads again, but without single buffer allocation per process. This remains WIP. Comm overhead might make this a bad idea anyway.
- allocate a buffer once and only once for each call to iter(). Pass a sliced view of this buffer to readers to fill up. Not implemented for multi-process work.
- bump version.
- infer contig prefix for TileDB-VCF. Also deprecate TileDB-VCF for now given issues creating TileDB-VCF datasets (segfaults) and no implementation for indels from TileDB-VCF (yet).
- bump version for bugfix.
- add license info to pyproject.toml

### Fix

- not all batch dims need be in every loader, handle this case.
- work-in-progress on wrong output lengths from fasta_variants.
- buffer length axis slicer was wrong length, should be computed by total length of *merged* regions.
- buffer length axis slicer was wrong length, should be computed by total length of regions.
- splice utils.
- attribute error.
- comment why versions are 0.0.0.
- dynamic versioning config.
- relax virtual data alignment constraints from exact join to inner join. feat: use dynamic versioning.
- make splicing util funcs generalize to n-dim arrays with length as the final axis.
- improve perf by not having batches as xr.Datasets.
- partitioned the wrong bed, jittered bed at wrong point in loop.
- forgot to cache jit function.
- batch_idx generation.
- computing relative starts for slicing buffers.
- computing max_ends.
- dim idx iteration.
- make strand column optional.
- forgot to pass contig to end_to_var_idx in Pgen read_for_hap.
- init Fasta.rev_strand_fn when alphabet is str. fix: using sample subsets with Pgen.
- uppercase alphabet when passed as a string.
- pass all tests! allow ploid kwarg in pgen reader, fix bugs with variant searching and max_end and end_idx calculation.
- pass all tests! allow ploid kwarg in pgen reader, fix bugs with variant searching and max_end and end idx calculation.
- consts for tracking buffer_idx column meanings. TODO reverse complement (or just reverse) data when slicing it from the buffer. Reverse complementing while constructing buffers requires partitions to be broken up more since only regions on the same strand can be merged.
- add pre-commit to dev dependencies.
- no variants in any query regions.
- ignore false positive dask.empty typing error.
- forgot to jit construct_haplotypes_with_indels().
- forgot comma.
- handle overlapping variants (i.e. genotype == ALT at same position in same sample) by only applying the first encountered.
- move ref_idx for deletions. feat: expose seed arg for fastavariants for determinism.
- checking that sample subset is all in pgen file.
- VLenAlleles slicing with None start. feat: cache pvar as arrow file.
- slicing VLenAlleles with start=None

## v0.0.2 (2023-10-11)

### Feat

- update README
- prepare to publish on pypi.
- reorganize loader code and add `set` method to update parameters that dont require re-initializing Ray Actors. fix: clean up BigWig docstring, plan for deprecation in favor of a method to convert to RLE table for big performance boost.
- reorganize code, minor changes.
- tested that ray-based loader runs.
- initial concurrent implementation with Ray.
- concurrent.futures based async buffering. Unfortunately, benchmarking shows this is slower than a single-threaded implementation.
- optimize buffer slicing. fix: setting uniform length.
- minor updates, increase fudge factor for memory usage.
- make libraries for different variant formats optional.
- reorganize, move loader into separate file.
- add pgen reader.
- optional lazy loading for RLE table.
- RLE table reader, corresponding to the BED5+ format.
- better docstring on Zarr reader.
- initial Zarr reader.
- more docstrings. fix: dtype conversion in bigwig.
- view_virtual_data to preview dimensions from combined readers and test that they are compatible. feat: weighted upsampling for entries that appear in batch dimensions.
- include GVL in __all__ imports. fix: Reader docstring.
- comments on how to implement async reads.
- return batch dim indices.

### Fix

- change shuffle of GVL.partitioned_bed to respect deprecation of random.shuffle's second argument. feat: make GVL.readers a dict.
- add license, description, repo link.
- prep for poetry to pypi.
- poetry build issues.
- wrong dtype for Fasta without padding. feat: optimize construct haplotypes with indels, parallel helps. fix: pgen position from 1-based to 0-based. fix: indexing bugs from converting code for 1 contig to multiple contigs in pgen.read().
- dtype of variant sizes.
- make construct_haplotypes_with_indels jittable.
- relax ray version constraint by not using subscript ObjectRef type.
- passing sample subsets correctly and tracking buffer use.
- update bnfo-environment.yml.
- wrong col names.
- export view_virtual_data in __all__. fix: switch bigwig to shared memmap and joblib, ray docs on shared memmory were less clear.
- partial batches.
- make GVL a proper iterator.
- accessing and padding for out of bounds regions.
- batch_dim issues, return_index issues, drop_last issues.
