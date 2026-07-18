//! `PgenWindowFiller` — decode one PGEN window into a `DecodedWindow` (issue #276 task
//! 10).
//!
//! Mirrors `crate::record_stream::vcf::VcfWindowFiller` (read that module doc first —
//! this one only calls out where PGEN diverges): drives genoray's
//! `PgenRecordSource -> ChunkAssembler` pipeline for a single job's region set on one
//! contig, then runs the same Task 2 transpose (`fill_decoded_window`). `DenseChunk` is
//! source-agnostic, so the transpose is byte-for-byte the same code VCF uses.
//!
//! ## Why PGEN needs its own POS->range machinery (VCF doesn't)
//!
//! A VCF is region-seekable via its tabix/CSI index (`VcfRecordSource::new` takes a
//! `{contig}:{start}-{end}` query directly). A `.pgen`/`.pvar` pair is not: genotypes
//! are addressed by a *global variant index*, not by position, and `PvarReader` (the
//! `.pvar` text scanner genoray's PGEN source drives in lockstep) has no seek — it can
//! only skip forward from byte 0. So before any window can be decoded, the filler needs
//! a POS->variant-index mapping. `PgenWindowFiller::new` builds that mapping *once*, by
//! scanning the whole `.pvar` for each contig's `[lo, hi)` variant-index range (see
//! `contig_var_ranges` below) — deliberately coarse, not narrowed to the window. See
//! "Coarse `var_start`" below for why narrowing is unsafe.
//!
//! ## pgenlib reader construction (the `Py<PyAny>` handle)
//!
//! Built once, at `PgenWindowFiller::new`, exactly the way genoray's own
//! `_svar2.py::from_pgen` builds it (`.pixi/envs/dev/lib/python3.10/site-packages/
//! genoray/_svar2.py:644-654`):
//!
//! ```python
//! import pgenlib
//! r = pgenlib.PgenReader(bytes(pgen_path), n_samples_full)
//! ```
//!
//! `n_samples_full` is the **full, on-disk `.psam` cohort size** — never a job's local
//! subset; subsetting happens later via `change_sample_subset` (see below). It is derived
//! from the `.psam` row count read at construction (the same `.psam` the public→physical
//! `phys` map below is built from), not passed in by the caller.
//!
//! **`allele_idx_offsets` is omitted (`None`).** genoray's own constructor call passes
//! it because SVAR2 conversion must support multiallelic PGEN. This filler does not:
//! `gvl.write`'s existing PGEN path and the streaming design both require
//! split/biallelic input (bcftools-norm'd upstream of `plink2 --make-pgen`), so every
//! variant has exactly 1 ALT and the trivial `[0, 2, 4, ...]` offsets are exactly what
//! `pgenlib` already assumes when the argument is omitted — confirmed empirically
//! (`pixi run -e dev python`, constructing `pgenlib.PgenReader(path, n_samples)` with no
//! `allele_idx_offsets` and reading a biallelic `.pgen` back correctly). **Multiallelic
//! PGEN is out of scope for v1** and would misdecode silently if fed in with no
//! validation — so `contig_var_ranges` (below), which already parses every `.pvar` row
//! for its POS->range scan, also checks each row's ALT column for a comma
//! (multiallelic) and errors out at `PgenWindowFiller::new` construction time, before
//! any pgenlib decode. A caller must pre-split with `bcftools norm -m-` /
//! `plink2 --make-pgen` from split input, same requirement `gvl.write` already
//! documents.
//!
//! ## Sample subsetting (`change_sample_subset`) — public sorted-name → physical `.psam`
//!
//! **The public `sample_idx` contract is lexicographically-sorted-name order** — the
//! order `gvl.write` stores genotypes in (`_write.py`'s unconditional `samples.sort()`)
//! and the order `gvl.Dataset.open()[r, s]` returns them in. A job's `[s_lo, s_hi)`
//! indexes into that *sorted-name* order, NOT the physical `.psam` column order (which
//! plink2 preserves from the source VCF and is arbitrary). These differ for any cohort
//! whose `.psam` sample order is not already sorted (e.g. `.psam` = `S10, S2, S1` →
//! sorted public = `S1, S10, S2`, since `"S10" < "S2"`), so mapping a public index
//! straight through to a physical pgenlib column would silently return the *wrong
//! sample's* haplotypes.
//! `PgenWindowFiller::new` therefore reads the `.psam` to learn the physical order and
//! builds `phys` — the public→physical map `phys[k] = position of the k-th sorted-name
//! sample within the `.psam`. This mirrors SVAR1's `_phys_sample_idx`
//! (`_dataset/_streaming.py::_Svar1Backend`) and genoray's `PGEN.set_samples`
//! (`_pgen.py`), which resolve names→physical indices then un-permute output back to the
//! requested order.
//!
//! `pgenlib.PgenReader.change_sample_subset` takes a **sorted-ascending** uint32
//! *physical* index array and returns genotype columns in that sorted-physical order
//! thereafter. So for a job's public slice `[s_lo, s_hi)` we (1) gather its physical
//! indices `phys[s_lo..s_hi]`, (2) sort a copy → `sorted_phys`, pass that to
//! `change_sample_subset`, and (3) build `sample_perm` as the *un-sorter*:
//! `sample_perm[out]` = the position of `phys[s_lo + out]` within `sorted_phys`, i.e.
//! which sorted-physical pgenlib column public output column `out` must read from
//! (`PgenRecordSource` applies `gt[out] = host_buf[.. sample_perm[out] ..]`, see
//! `pgen_reader.rs`). This reproduces `genoray/_pgen.py::set_samples`'s
//! `argsort`/`_s_unsorter` logic exactly. Physical indices are unique (one per `.psam`
//! name) so `sorted_phys` has no duplicates and `binary_search` gives each output
//! column's exact source. `fill` recomputes and re-applies the subset on *every*
//! invocation (never memoized against the previous job) so a stale subset can never leak
//! into the next window.
//!
//! There is deliberately **no `None`/identity fast-path**: `change_sample_subset(None)`
//! resets pgenlib to the full cohort in *physical* order, which is only correct when
//! `phys` is already the identity (an already-sorted `.psam`) — the exact case this fix
//! exists to stop assuming. An explicit `sorted_phys` array is always passed instead;
//! for an already-sorted full cohort it is `[0, n_full)` and behaves identically to
//! `None`, so correctness never depends on the `.psam` happening to be sorted.
//!
//! Mutating the shared `Py<PyAny>` reader between calls is safe because
//! `RecordBackend`'s single producer thread calls `WindowFiller::fill` strictly
//! sequentially, one job at a time (never concurrently — see `engine.rs`'s and
//! `stream_core`'s module docs); there is no `&mut self` on `fill`, but none is needed
//! since the mutation happens Python-side through the `Py<PyAny>` handle, not through any
//! Rust-owned field.
//!
//! ## Per-contig variant ranges: a from-scratch `.pvar` scan (not genoray's private
//! `_pvar_contig_ranges`)
//!
//! `PvarReader` (the Rust `.pvar` reader `PgenRecordSource` itself drives) parses only
//! POS/REF/ALT — it does not track `#CHROM` at all, since SVAR2 conversion is always
//! told its contig list up front by the (Python) caller. This filler has no such
//! caller-supplied contig list, so `contig_var_ranges` (below) does its own single
//! forward pass over the `.pvar`, tracking `#CHROM` alongside a running variant index,
//! to build the `{contig -> [lo, hi)}` map this module needs. This deliberately does
//! *not* reuse genoray's private Python `_svar2._pvar_contig_ranges` — that helper also
//! computes `allele_idx_offsets` (irrelevant here, since v1 is biallelic-only and omits
//! that argument entirely) and pulling it in would mean depending on a private function
//! *and* importing `polars` into the streaming hot path for no benefit. The Rust scan
//! below covers exactly what this filler needs and nothing more.
//!
//! Plain-text `.pvar` only (matching this repo's committed fixture and `plink2
//! --make-pgen`'s default). A `.pvar.zst` input is rejected with a clear error rather
//! than silently mis-scanned — `PvarReader` itself supports `.zst` (via the `zstd`
//! crate, a `genoray_core`-only dependency this crate does not otherwise need); wiring
//! that up is a small, isolated follow-up if a `.zst` fixture is ever needed.
//!
//! ## Coarse `var_start`: correct, but a known v1 perf limitation
//!
//! `fill` always decodes a contig from its **first** variant (`var_start = ` the whole
//! contig's `lo`), never narrowed to the window's own regions. This is required for
//! correctness, not laziness: a spanning upstream deletion (POS before the window but
//! whose extent reaches into it) must still be visible to `PgenRecordSource::next_record`
//! (whose region filter uses `OverlapMode::Variant`/`extent_overlaps`, exactly like the
//! VCF path — see that module's doc comment on why `Variant` overlap is required). Since
//! `PgenRecordSource` walks strictly forward from `var_start` with no way to "look
//! behind" once started, `var_start` must be low enough to have seen every variant that
//! could possibly span into the window — and without a `max_v_len`-style padding bound
//! (out of scope for v1, same caveat the VCF path's window construction carries), the
//! only *safe* lower bound is the whole contig's start. `var_end` is similarly the whole
//! contig's end; the `regions`/`OverlapMode::Variant`/region-exhaustion filter inside
//! `PgenRecordSource::next_record` still narrows the *output* to exactly the window's
//! overlapping variants, so this is correctness-preserving.
//!
//! **This is a real, documented performance cost**, not silently swept under the rug:
//! every window on a contig re-decodes (and region-filters) that contig from its start,
//! and `PgenRecordSource::new` re-opens the `.pvar` from byte 0 every call (`PvarReader`
//! has no seek), skipping `var_start` lines textually. For the small fixtures this repo
//! tests against that's negligible; at chromosome scale with many windows per contig it
//! is O(prefix) decode + O(var_start) `.pvar`-skip **per window**. Task 13's benchmark is
//! expected to surface this; the fix (narrow `var_start` with `max_v_len` padding, and/or
//! add a seek to genoray's `PvarReader` and bump the pinned `rev`) is deliberately
//! deferred — v1 prioritizes correctness/parity with the VCF path over throughput.
//!
//! ## Config parity with VCF (and with `gvl.write`'s PGEN path)
//!
//! `fasta_path: None`, `check_ref: CheckRef::Exclude` (inert without a FASTA — see
//! `vcf.rs`'s doc comment), `fields: &[]`. **`ploidy` is hardwired to 2** (not
//! threaded as a parameter): PGEN is diploid-only by format, matching
//! `_svar2.py::from_pgen`'s doc ("PGEN is diploid, so there is no `ploidy` parameter").
//!
//! ## GIL discipline
//!
//! `PgenRecordSource::refill` (inside genoray) already wraps its own `read_alleles_range`
//! call in `Python::attach` — the producer thread that calls `fill` holds no GIL between
//! chunks. This filler only touches Python explicitly for (a) one-time reader
//! construction in `new`, and (b) the `change_sample_subset` call at the top of each
//! `fill`; neither wraps the `ChunkAssembler`/`PgenRecordSource` decode loop itself.

use std::collections::HashMap;

use anyhow::{ensure, Context};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::normalize::CheckRef;
use genoray_core::pgen_reader::PgenRecordSource;
use genoray_core::svar2_view::OverlapMode;
use genoray_core::types::{BitGrid3, DenseChunk};

use crate::record_stream::engine::{ContigRef, RecordJob, WindowFiller};
use crate::record_stream::{fill_decoded_window, DecodedWindow};

/// Upper bound on a single window's variant count. See `vcf.rs`'s "Single-chunk
/// invariant" doc section — identical reasoning applies here.
const DEFAULT_CHUNK_SIZE: usize = 1 << 20;

/// PGEN is diploid-only by format (no ploidy field on disk) — see the module doc's
/// "Config parity" section.
const PGEN_PLOIDY: usize = 2;

/// Scan `pvar_path` once, tracking `#CHROM` alongside a running 0-based variant index,
/// to build each contig's half-open `[lo, hi)` variant-index range. See the module doc's
/// "Per-contig variant ranges" section for why this doesn't reuse genoray's private
/// Python `_pvar_contig_ranges`.
///
/// Plain-text `.pvar` only; `.pvar.zst` is rejected (see module doc).
///
/// Requires each contig's rows to be contiguous (plink2 always writes `.pvar` grouped by
/// contig) — a non-contiguous file is an error, not a silently-wrong range.
fn contig_var_ranges(pvar_path: &str) -> anyhow::Result<HashMap<String, (usize, usize)>> {
    use std::io::BufRead;

    ensure!(
        !pvar_path.ends_with(".zst"),
        "PgenWindowFiller: {pvar_path} is zstd-compressed; only plain-text .pvar is \
         supported in v1 (see the module doc's \"Per-contig variant ranges\" section)"
    );

    let file =
        std::fs::File::open(pvar_path).with_context(|| format!("opening pvar {pvar_path}"))?;
    let mut lines = std::io::BufReader::new(file).lines();

    let (chrom_col, pos_col, alt_col) = loop {
        let line = lines.next().with_context(|| {
            format!("{pvar_path}: reached EOF without a '#CHROM' header line")
        })??;
        if line.starts_with("##") {
            continue;
        }
        anyhow::ensure!(
            line.starts_with("#CHROM"),
            "{pvar_path}: expected a '#CHROM' header line, found '{line}'"
        );
        let cols: Vec<&str> = line.split('\t').collect();
        let find = |name: &str| {
            cols.iter()
                .position(|c| *c == name)
                .with_context(|| format!("{pvar_path}: header is missing a {name:?} column"))
        };
        break (find("#CHROM")?, find("POS")?, find("ALT")?);
    };

    let mut ranges: HashMap<String, (usize, usize)> = HashMap::new();
    for (vidx, line) in lines.enumerate() {
        let line = line.with_context(|| format!("reading pvar {pvar_path}"))?;
        let fields: Vec<&str> = line.split('\t').collect();
        let chrom = *fields
            .get(chrom_col)
            .with_context(|| format!("{pvar_path}: variant {vidx} is missing its #CHROM field"))?;
        let pos = *fields
            .get(pos_col)
            .with_context(|| format!("{pvar_path}: variant {vidx} is missing its POS field"))?;
        let alt = *fields
            .get(alt_col)
            .with_context(|| format!("{pvar_path}: variant {vidx} is missing its ALT field"))?;
        // PgenWindowFiller omits pgenlib's `allele_idx_offsets`, which is only correct
        // for biallelic PGEN (see the module doc's "pgenlib reader construction"
        // section) -- a multiallelic ALT (comma-separated) would silently misdecode
        // downstream with no error, so reject it loudly here, at construction, before
        // any decode happens.
        anyhow::ensure!(
            !alt.contains(','),
            "PgenWindowFiller: multiallelic variant at {chrom}:{pos} (ALT='{alt}'); PGEN \
             streaming v1 requires biallelic (split) input -- run `bcftools norm -m -` / \
             `plink2 --make-pgen` on split records first."
        );
        let chrom = chrom.to_string();

        match ranges.get_mut(&chrom) {
            Some((_, hi)) => {
                anyhow::ensure!(
                    *hi == vidx,
                    "{pvar_path}: contig {chrom:?} is not contiguous (SVAR2/PGEN \
                     streaming requires a .pvar grouped by contig, as plink2 always \
                     writes it)"
                );
                *hi = vidx + 1;
            }
            None => {
                ranges.insert(chrom, (vidx, vidx + 1));
            }
        }
    }

    Ok(ranges)
}

/// Read a plink2 `.psam`'s sample-ID (IID) column, in physical (on-disk) order.
///
/// The `.psam` is plain text with a `#`-prefixed header line (after optional `##`
/// pragma lines). The sample ID is the `IID` column; plink2 writes either a bare
/// `#IID ...` header or a `#FID\tIID\t...` header, so this locates the column named
/// `IID` (matching with the leading `#` stripped) rather than assuming a fixed position.
/// Whitespace-delimited to tolerate either tab- or space-separated `.psam` variants.
///
/// This physical order is what `PgenWindowFiller::new` maps the caller's sorted-name
/// `sample_idx` order onto (see the module doc's "Sample subsetting" section).
fn read_psam_sample_names(psam_path: &str) -> anyhow::Result<Vec<String>> {
    use std::io::BufRead;

    let file =
        std::fs::File::open(psam_path).with_context(|| format!("opening psam {psam_path}"))?;
    let mut lines = std::io::BufReader::new(file).lines();

    // First non-`##` line is the header; find the IID column index.
    let iid_col = loop {
        let line = lines
            .next()
            .with_context(|| format!("{psam_path}: reached EOF without a header line"))??;
        if line.starts_with("##") {
            continue;
        }
        anyhow::ensure!(
            line.starts_with('#'),
            "{psam_path}: expected a '#'-prefixed .psam header line, found '{line}'"
        );
        let col = line
            .split_whitespace()
            .position(|c| c.trim_start_matches('#') == "IID")
            .with_context(|| format!("{psam_path}: header is missing an 'IID' column"))?;
        break col;
    };

    let mut names = Vec::new();
    for line in lines {
        let line = line.with_context(|| format!("reading psam {psam_path}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let iid = line
            .split_whitespace()
            .nth(iid_col)
            .with_context(|| format!("{psam_path}: a sample row is missing its IID column"))?;
        names.push(iid.to_string());
    }
    anyhow::ensure!(!names.is_empty(), "{psam_path}: no sample rows found");
    Ok(names)
}

/// Decodes one PGEN window via genoray's record-stream pipeline. Holds the pgenlib
/// `Py<PyAny>` reader handle (constructed once, reused across `fill` calls — see the
/// module doc's "Sample subsetting" section on why mutating it between sequential calls
/// is safe) plus the `.pvar` path and the per-contig range map computed once at
/// construction.
pub struct PgenWindowFiller {
    reader: Py<PyAny>,
    pvar_path: String,
    /// Public→physical sample map: `phys[k]` is the `.psam` (physical) column of the
    /// k-th sample in the caller's lexicographically-sorted `sample_idx` order. See the
    /// module doc's "Sample subsetting" section. Its length is the public cohort size a
    /// job's `[s_lo, s_hi)` indexes into.
    phys: Vec<usize>,
    contig_ranges: HashMap<String, (usize, usize)>,
    overlap: OverlapMode,
    chunk_size: usize,
}

impl PgenWindowFiller {
    /// `pgen_path` must have a sibling `.pvar` (not `.pvar.zst`, see module doc) and a
    /// sibling `.psam`. `public_sample_names` is the caller's lexicographically-sorted
    /// `sample_idx` order (`_PgenBackend._sample_names`); `new` reads the `.psam` to learn
    /// the physical column order and builds the public→physical `phys` map (see the module
    /// doc's "Sample subsetting" section). The full on-disk cohort size passed to the
    /// pgenlib reader is derived from the `.psam` row count, not from `public_sample_names`
    /// (which may in principle be a subset).
    pub fn new(pgen_path: &str, public_sample_names: &[&str]) -> anyhow::Result<Self> {
        let pvar_path = std::path::Path::new(pgen_path)
            .with_extension("pvar")
            .to_str()
            .context("pgen_path is not valid UTF-8")?
            .to_string();
        // Multiallelic guard runs FIRST, before the `.psam` read / reader construction, so
        // a multiallelic `.pvar` still fails at construction even when no `.psam`/`.pgen`
        // is present (see the multiallelic-fixture test).
        let contig_ranges = contig_var_ranges(&pvar_path)?;

        // Build the public->physical sample map from the sibling `.psam`.
        let psam_path = std::path::Path::new(pgen_path)
            .with_extension("psam")
            .to_str()
            .context("pgen_path is not valid UTF-8")?
            .to_string();
        let psam_names = read_psam_sample_names(&psam_path)?;
        let n_samples_full = psam_names.len();

        let mut name_to_phys: HashMap<&str, usize> = HashMap::with_capacity(psam_names.len());
        for (i, name) in psam_names.iter().enumerate() {
            name_to_phys.insert(name.as_str(), i);
        }
        anyhow::ensure!(
            name_to_phys.len() == psam_names.len(),
            "{psam_path}: duplicate sample IDs in the .psam; sample names must be unique"
        );
        let phys: Vec<usize> = public_sample_names
            .iter()
            .map(|name| {
                name_to_phys.get(*name).copied().with_context(|| {
                    format!(
                        "PgenWindowFiller: requested sample {name:?} (from the sorted \
                         sample_idx order) is not present in {psam_path}"
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // pgenlib.PgenReader(bytes(pgen_path), n_samples_full) -- see the module doc's
        // "pgenlib reader construction" section. `allele_idx_offsets` is omitted
        // (biallelic-only v1).
        let reader: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let pgenlib = py.import("pgenlib")?;
            let cls = pgenlib.getattr("PgenReader")?;
            let path_bytes = PyBytes::new(py, pgen_path.as_bytes());
            let reader = cls.call1((path_bytes, n_samples_full))?;
            Ok(reader.unbind())
        })
        .with_context(|| format!("constructing pgenlib.PgenReader for {pgen_path}"))?;

        Ok(Self {
            reader,
            pvar_path,
            phys,
            contig_ranges,
            overlap: OverlapMode::Variant,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }

    /// Override the single-chunk size bound (tests only need a handful of variants).
    #[cfg(test)]
    fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the reader's active sample subset for the public slice `[s_lo, s_hi)` and
    /// return the `sample_perm` un-sorter mapping each public output column to the
    /// sorted-physical pgenlib column it must read. Always called at the top of `fill`
    /// (never memoized) — see the module doc's "Sample subsetting" section for the full
    /// public-sorted-name → physical `.psam` mapping and why there is no identity
    /// fast-path.
    fn apply_sample_subset(
        &self,
        py: Python<'_>,
        s_lo: usize,
        s_hi: usize,
    ) -> anyhow::Result<Vec<usize>> {
        anyhow::ensure!(
            s_hi <= self.phys.len() && s_lo <= s_hi,
            "PgenWindowFiller: invalid sample range [{s_lo}, {s_hi}) for a {}-sample \
             (public) cohort",
            self.phys.len()
        );

        // This job's physical indices, in public (sorted-name) order.
        let phys_subset: Vec<u32> = self.phys[s_lo..s_hi].iter().map(|&p| p as u32).collect();

        // pgenlib.change_sample_subset requires an ascending physical index array; sort a
        // copy and, for each public output column, record where its physical index landed
        // in that sorted array -- the un-sorter `PgenRecordSource` applies to reorder the
        // sorted-physical pgenlib columns back into public order. Physical indices are
        // unique (one per .psam name) so `sorted_phys` has no duplicates and each
        // `binary_search` is an exact hit.
        let mut sorted_phys = phys_subset.clone();
        sorted_phys.sort_unstable();
        let sample_perm: Vec<usize> = phys_subset
            .iter()
            .map(|p| {
                sorted_phys
                    .binary_search(p)
                    .expect("physical index must be present in its own sorted copy")
            })
            .collect();

        let reader = self.reader.bind(py);
        let arr = sorted_phys.into_pyarray(py);
        reader
            .call_method1("change_sample_subset", (arr,))
            .map_err(|e| anyhow::anyhow!("pgenlib change_sample_subset failed: {e}"))?;
        Ok(sample_perm)
    }
}

impl WindowFiller for PgenWindowFiller {
    fn fill(
        &self,
        job: &RecordJob,
        contig: &ContigRef,
        slot: &mut DecodedWindow,
    ) -> anyhow::Result<()> {
        let n_local_samples = job.s_hi - job.s_lo;

        // Set the pgenlib subset to this job's physical columns and get the un-sorter that
        // maps public (sorted-name) output columns back to sorted-physical pgenlib columns
        // (see the module doc's "Sample subsetting" section).
        let sample_perm = Python::attach(|py| self.apply_sample_subset(py, job.s_lo, job.s_hi))?;

        // Coarse per-contig range -- see the module doc's "Coarse var_start" section for
        // why this is NOT narrowed to the window's own regions. A contig absent from the
        // .pvar (no variants at all) falls back to an empty [0, 0) range, which decodes
        // to the same all-zero-CSR empty window as a region with no variants (see below).
        let (var_start, var_end) = self
            .contig_ranges
            .get(&contig.name)
            .copied()
            .unwrap_or((0, 0));

        let reader = Python::attach(|py| self.reader.clone_ref(py));
        let source = PgenRecordSource::new(
            reader,
            &self.pvar_path,
            var_start,
            var_end,
            n_local_samples,
            self.chunk_size,
            job.regions.clone(),
            self.overlap,
            sample_perm,
        )?;
        let mut asm = ChunkAssembler::new(
            Box::new(source),
            n_local_samples,
            PGEN_PLOIDY,
            /* fasta_path */ None,
            &contig.name,
            /* skip_out_of_scope */ true,
            CheckRef::Exclude,
            &[],
        )?;

        match asm.read_next_chunk(self.chunk_size, 0, None)? {
            Some(chunk) => {
                // Single-chunk invariant (see `vcf.rs`'s doc section of the same name).
                ensure!(
                    asm.read_next_chunk(self.chunk_size, 1, None)?.is_none(),
                    "PgenWindowFiller single-chunk invariant violated: window on {} \
                     regions {:?} has more variants than chunk_size ({}) — raise \
                     DEFAULT_CHUNK_SIZE or shrink the job's regions",
                    contig.name,
                    job.regions,
                    self.chunk_size,
                );
                fill_decoded_window(&chunk, n_local_samples, PGEN_PLOIDY, slot);
            }
            None => {
                let empty = DenseChunk {
                    chunk_id: 0,
                    pos: Vec::new(),
                    ilens: Vec::new(),
                    alt: Vec::new(),
                    alt_offsets: vec![0],
                    genos: BitGrid3::zeros(0, n_local_samples, PGEN_PLOIDY),
                    info_staged: Vec::new(),
                    format_staged: Vec::new(),
                };
                fill_decoded_window(&empty, n_local_samples, PGEN_PLOIDY, slot);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_stream::vcf::VcfWindowFiller;

    fn vcf_fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/two_var_two_sample.vcf.gz")
            .to_str()
            .unwrap()
            .to_string()
    }

    fn pgen_fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/two_var_two_sample.pgen")
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Fixture with an UNSORTED `.psam` sample order (physical order `S10, S2, S1`;
    /// lexicographically-sorted public order `S1, S10, S2` — note `"S10" < "S2"`), the
    /// regression fixture for the sample-ordering bug: `PgenWindowFiller` previously read
    /// physical column `s` for public index `s`, silently returning the wrong sample's
    /// genotypes for any cohort whose `.psam` is not already sorted. Each sample carries
    /// exactly one distinct SNP (S10→v0@10, S2→v1@30, S1→v2@50), so a physical-vs-sorted
    /// mix-up changes the genotype CSR. Generated (committed under tests/data/streaming/)
    /// via:
    /// `plink2 --vcf unsorted_samples.vcf.gz --make-pgen --allow-extra-chr \
    ///  --output-chr chrM --out unsorted_samples`.
    fn unsorted_vcf_fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/unsorted_samples.vcf.gz")
            .to_str()
            .unwrap()
            .to_string()
    }

    fn unsorted_pgen_fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/unsorted_samples.pgen")
            .to_str()
            .unwrap()
            .to_string()
    }

    /// `tests/data/streaming/multiallelic.pvar` is a hand-written, plain-text `.pvar`
    /// with a single multiallelic row (`ALT='G,T'`). No sibling `.pgen`/`.psam` exists
    /// (and none is needed): the multiallelic guard lives in `contig_var_ranges`, which
    /// `PgenWindowFiller::new` runs *before* ever touching the `.pgen` file (see
    /// `new`'s body), so construction must fail on the `.pvar` scan alone.
    fn multiallelic_pgen_fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/multiallelic.pgen")
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Fixture: `tests/data/streaming/two_var_two_sample.{pgen,pvar,psam}`, generated
    /// from the VCF fixture `two_var_two_sample.vcf.gz` via:
    /// `plink2 --vcf two_var_two_sample.vcf.gz --make-pgen --allow-extra-chr \
    ///  --output-chr chrM --out two_var_two_sample`
    /// (`--output-chr chrM` preserves the `chr1` contig spelling the VCF fixture uses;
    /// plink2's default strips the `chr` prefix for recognized human contigs.)
    #[test]
    fn pgen_filler_matches_vcf_filler_on_shared_variants() {
        let vcf_filler = VcfWindowFiller::new(&vcf_fixture_path(), &["s1", "s2"], 2, None).unwrap();
        let pgen_filler = PgenWindowFiller::new(&pgen_fixture_path(), &["s1", "s2"]).unwrap();

        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };

        let mut vcf_slot = DecodedWindow::default();
        vcf_filler.fill(&job, &contig, &mut vcf_slot).unwrap();
        let mut pgen_slot = DecodedWindow::default();
        pgen_filler.fill(&job, &contig, &mut pgen_slot).unwrap();

        assert_eq!(pgen_slot.v_starts, vcf_slot.v_starts);
        assert_eq!(pgen_slot.ilens, vcf_slot.ilens);
        assert_eq!(pgen_slot.alt_alleles, vcf_slot.alt_alleles);
        assert_eq!(pgen_slot.alt_offsets, vcf_slot.alt_offsets);
        assert_eq!(pgen_slot.geno_v_idxs, vcf_slot.geno_v_idxs);
        assert_eq!(pgen_slot.geno_offsets, vcf_slot.geno_offsets);
    }

    /// Cross-backend equivalence must hold for a sample sub-range too — proves
    /// `set_sample_subset`'s contiguous-range identity permutation is correct, not just
    /// the full-cohort no-op path.
    #[test]
    fn pgen_filler_matches_vcf_filler_on_sample_subrange() {
        let vcf_filler = VcfWindowFiller::new(&vcf_fixture_path(), &["s1", "s2"], 2, None).unwrap();
        let pgen_filler = PgenWindowFiller::new(&pgen_fixture_path(), &["s1", "s2"]).unwrap();

        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 1,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };

        let mut vcf_slot = DecodedWindow::default();
        vcf_filler.fill(&job, &contig, &mut vcf_slot).unwrap();
        let mut pgen_slot = DecodedWindow::default();
        pgen_filler.fill(&job, &contig, &mut pgen_slot).unwrap();

        assert_eq!(pgen_slot.v_starts, vcf_slot.v_starts);
        assert_eq!(pgen_slot.geno_v_idxs, vcf_slot.geno_v_idxs);
        assert_eq!(pgen_slot.geno_offsets, vcf_slot.geno_offsets);
    }

    /// REGRESSION (correctness blocker): for a cohort whose `.psam` order is NOT
    /// lexicographically sorted, `PgenWindowFiller` must map the public sorted-name
    /// `sample_idx` order onto the physical `.psam` columns — output column `k` must be
    /// `sorted_names[k]`'s genotypes, exactly what `VcfWindowFiller` (which resolves by
    /// name) yields. The `unsorted_samples` fixture's `.psam` is `S10, S2, S1` (sorted
    /// public `S1, S10, S2`, since `"S10" < "S2"`), each sample carrying a distinct SNP,
    /// so the pre-fix identity mapping returned the WRONG sample's genotype CSR and this
    /// test FAILS before the fix / PASSES after. Compares all six `DecodedWindow` arrays,
    /// esp. the genotype CSR (`geno_v_idxs`/`geno_offsets`) where the ordering bug shows.
    #[test]
    fn pgen_filler_matches_vcf_filler_unsorted_psam() {
        // Both fillers get the SAME sorted-name public order; VcfWindowFiller resolves by
        // name (correct oracle), PgenWindowFiller must reproduce it via the .psam map.
        // Sorted order is S1, S10, S2 (physical .psam order is S10, S2, S1).
        let public = ["S1", "S10", "S2"];
        let vcf_filler =
            VcfWindowFiller::new(&unsorted_vcf_fixture_path(), &public, 2, None).unwrap();
        let pgen_filler = PgenWindowFiller::new(&unsorted_pgen_fixture_path(), &public).unwrap();

        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 60)],
            s_lo: 0,
            s_hi: 3,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 60],
        };

        let mut vcf_slot = DecodedWindow::default();
        vcf_filler.fill(&job, &contig, &mut vcf_slot).unwrap();
        let mut pgen_slot = DecodedWindow::default();
        pgen_filler.fill(&job, &contig, &mut pgen_slot).unwrap();

        assert_eq!(pgen_slot.v_starts, vcf_slot.v_starts);
        assert_eq!(pgen_slot.ilens, vcf_slot.ilens);
        assert_eq!(pgen_slot.alt_alleles, vcf_slot.alt_alleles);
        assert_eq!(pgen_slot.alt_offsets, vcf_slot.alt_offsets);
        assert_eq!(
            pgen_slot.geno_v_idxs, vcf_slot.geno_v_idxs,
            "genotype CSR variant indices must match name-resolved VCF order"
        );
        assert_eq!(
            pgen_slot.geno_offsets, vcf_slot.geno_offsets,
            "genotype CSR per-hap offsets must match name-resolved VCF order"
        );
    }

    /// A sub-range of an unsorted `.psam` cohort must ALSO map correctly: public slice
    /// `[0, 2)` of sorted order `S1, S10, S2` is names `S1, S10` (physical `2, 0`) — an
    /// order-reversing physical subset that exercises both the `sorted_phys` subset AND
    /// the un-sorter (`sample_perm`), not just the full-cohort path above.
    #[test]
    fn pgen_filler_matches_vcf_filler_unsorted_psam_subrange() {
        let public = ["S1", "S10", "S2"];
        let vcf_filler =
            VcfWindowFiller::new(&unsorted_vcf_fixture_path(), &public, 2, None).unwrap();
        let pgen_filler = PgenWindowFiller::new(&unsorted_pgen_fixture_path(), &public).unwrap();

        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 60)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 60],
        };

        let mut vcf_slot = DecodedWindow::default();
        vcf_filler.fill(&job, &contig, &mut vcf_slot).unwrap();
        let mut pgen_slot = DecodedWindow::default();
        pgen_filler.fill(&job, &contig, &mut pgen_slot).unwrap();

        assert_eq!(pgen_slot.v_starts, vcf_slot.v_starts);
        assert_eq!(pgen_slot.geno_v_idxs, vcf_slot.geno_v_idxs);
        assert_eq!(pgen_slot.geno_offsets, vcf_slot.geno_offsets);
    }

    /// A region with no variants must decode to the same all-zero-CSR empty state as
    /// VCF's (`vcf_filler_empty_window_is_all_zero_csr`).
    #[test]
    fn pgen_filler_empty_window_is_all_zero_csr() {
        let filler = PgenWindowFiller::new(&pgen_fixture_path(), &["s1", "s2"]).unwrap();
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(500, 600)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 1000],
        };
        let mut slot = DecodedWindow::default();
        filler.fill(&job, &contig, &mut slot).unwrap();

        assert!(slot.v_starts.is_empty());
        assert!(slot.ilens.is_empty());
        assert!(slot.alt_alleles.is_empty());
        assert_eq!(slot.alt_offsets, vec![0]);
        assert!(slot.geno_v_idxs.is_empty());
        assert_eq!(slot.geno_offsets, vec![0, 0, 0, 0, 0]);
    }

    /// The single-chunk invariant: a `chunk_size` too small for the window's variant
    /// count must fail loudly, never silently truncate (mirrors VCF's
    /// `vcf_filler_errors_when_window_exceeds_chunk_size`).
    #[test]
    fn pgen_filler_errors_when_window_exceeds_chunk_size() {
        let filler = PgenWindowFiller::new(&pgen_fixture_path(), &["s1", "s2"])
            .unwrap()
            .with_chunk_size(1);
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };
        let mut slot = DecodedWindow::default();
        let err = filler.fill(&job, &contig, &mut slot).unwrap_err();
        assert!(
            format!("{err:?}").contains("single-chunk invariant"),
            "expected the single-chunk guard to fire, got: {err:?}"
        );
    }

    /// A multiallelic `.pvar` (ALT with a comma) must be rejected loudly at
    /// construction, never silently misdecoded (see the module doc's "pgenlib reader
    /// construction" section on why omitting `allele_idx_offsets` is unsafe for
    /// multiallelic input).
    #[test]
    fn pgen_filler_rejects_multiallelic_pvar_at_construction() {
        let err = match PgenWindowFiller::new(&multiallelic_pgen_fixture_path(), &["s0", "s1"]) {
            Ok(_) => panic!("expected PgenWindowFiller::new to reject a multiallelic .pvar"),
            Err(e) => e,
        };
        let msg = format!("{err:?}");
        assert!(
            msg.contains("multiallelic") && msg.contains("biallelic"),
            "expected a multiallelic/biallelic error message, got: {msg}"
        );
    }
}
