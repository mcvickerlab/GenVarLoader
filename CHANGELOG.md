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
