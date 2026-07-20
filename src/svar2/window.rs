//! GIL-free SVAR2 window range computation + super-batch reconstruction.
//! The producer thread in `Svar2StreamEngine` (PR-3 Task 2) runs this without the
//! GIL; the `svar2_fill_super_batch` pyfunction (`src/ffi/mod.rs`) is a thin
//! GIL-holding wrapper used to prove byte-identity vs the Python "sync" fill+drain
//! path (`_Svar2Backend._fill_super_batch` + `_drain`).
use std::ops::Range;

/// One window's live `find_ranges` output, kept as `Range<usize>` (no i64 flatten --
/// that flatten only exists to cross the Python/Rust FFI boundary in
/// `svar2_read_window`; a GIL-free caller has no such boundary to cross).
pub(crate) struct Svar2WindowRanges {
    pub n_reg: usize,
    pub n_s: usize,
    pub ploidy: usize,
    /// `[start, end)` into `vk_snp`'s packed positions/keys, per `(region, selected
    /// sample, ploid)` -- C-order, len `n_reg*n_s*ploidy`.
    pub vk_snp: Vec<Range<usize>>,
    /// `[start, end)` into `vk_indel`'s packed positions/keys, same shape as `vk_snp`.
    pub vk_indel: Vec<Range<usize>>,
    /// `[start, end)` into `dense/snp`'s on-disk positions/keys, per region.
    pub dense_snp: Vec<Range<usize>>,
    /// `[start, end)` into `dense/indel`'s on-disk positions/keys, per region.
    pub dense_indel: Vec<Range<usize>>,
    /// Selected slot -> original (physical) sample column, len `n_s`.
    pub sample_cols: Vec<usize>,
}

impl Svar2WindowRanges {
    /// Compute the window's ranges once (genoray `find_ranges`). `regions`/`samples`
    /// are physical (store column order), matching what `svar2_read_window` passes.
    pub(crate) fn compute(
        reader: &genoray_core::query::ContigReader,
        regions: &[(u32, u32)],
        samples: &[usize],
        ploidy: usize,
    ) -> Self {
        let rb = genoray_core::query::find_ranges(reader, regions, Some(samples));
        Self {
            n_reg: regions.len(),
            n_s: samples.len(),
            ploidy,
            vk_snp: rb.vk_snp_range,
            vk_indel: rb.vk_indel_range,
            dense_snp: rb.dense_snp_range,
            dense_indel: rb.dense_indel_range,
            sample_cols: rb.sample_cols,
        }
    }
}

/// Reconstruct C-order rows `[sb_lo, sb_hi)` of the window into `out_data`/
/// `out_offsets` (GIL-free) -- the one-call `find_ranges` -> per-row gather ->
/// `svar2_readbound_chain` primitive the streaming engine's producer thread reuses.
///
/// `region_bounds` is the window's per-region `(start, end)` i32 pairs, len `n_reg`
/// (same layout `_fill_super_batch`/`_gather_rows` build from `window["region_bounds"]`).
/// Jitter is out of scope here (shifts are always zero), matching `_gather_rows`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_super_batch_rs(
    reader: &genoray_core::query::ContigReader,
    ranges: &Svar2WindowRanges,
    region_bounds: &[(i32, i32)],
    ref_bytes: &[u8],
    pad_char: u8,
    sb_lo: usize,
    sb_hi: usize,
    parallel: bool,
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<i64>,
) {
    use numpy::ndarray::{Array2, ArrayView1};
    debug_assert_eq!(
        region_bounds.len(),
        ranges.n_reg,
        "region_bounds length must match the window's ranges.n_reg"
    );
    let n_s = ranges.n_s;
    let p = ranges.ploidy;
    let m = sb_hi - sb_lo;

    // Per-row expanded inputs (C-order (region, sample)), mirroring `_gather_rows`.
    let mut region_starts_v: Vec<u32> = Vec::with_capacity(m);
    let mut orig_samples_v: Vec<usize> = Vec::with_capacity(m);
    let mut vk_snp_v: Vec<Range<usize>> = Vec::with_capacity(m * p);
    let mut vk_indel_v: Vec<Range<usize>> = Vec::with_capacity(m * p);
    let mut dense_snp_v: Vec<Range<usize>> = Vec::with_capacity(m);
    let mut dense_indel_v: Vec<Range<usize>> = Vec::with_capacity(m);
    // `svar2_readbound_chain`'s `regions` is `(n_q, 3)` = `[contig_idx, start, end)`
    // (see `reconstruct::reconstruct_haplotypes_from_svar2`'s `regions[[query, 0/1/2]]`
    // indexing) -- NOT `(n_q, 2)`. `contig_idx` is always 0: `ref_bytes` is always the
    // single-contig slice the caller already sliced out (same convention
    // `reconstruct_haplotypes_from_svar2_readbound`/`svar2_reconstruct_super_batch` use).
    let mut region_bounds_a = Array2::<i32>::zeros((m, 3));
    let shifts_a = Array2::<i32>::zeros((m, p)); // jitter=0

    for (j, row) in (sb_lo..sb_hi).enumerate() {
        let ri = row / n_s;
        let si = row % n_s;
        let (rs, re) = region_bounds[ri];
        region_starts_v.push(rs as u32);
        region_bounds_a[[j, 1]] = rs;
        region_bounds_a[[j, 2]] = re;
        orig_samples_v.push(ranges.sample_cols[si]);
        let base = (ri * n_s + si) * p;
        for pp in 0..p {
            vk_snp_v.push(ranges.vk_snp[base + pp].clone());
            vk_indel_v.push(ranges.vk_indel[base + pp].clone());
        }
        dense_snp_v.push(ranges.dense_snp[ri].clone());
        dense_indel_v.push(ranges.dense_indel[ri].clone());
    }

    let ref_a: ArrayView1<u8> = ArrayView1::from(ref_bytes);
    let ref_offsets = [0i64, ref_bytes.len() as i64];
    let ref_offsets_a = ArrayView1::from(&ref_offsets[..]);

    crate::svar2::svar2_readbound_chain(
        reader,
        &region_starts_v,
        &orig_samples_v,
        &vk_snp_v,
        &vk_indel_v,
        &dense_snp_v,
        &dense_indel_v,
        region_bounds_a.view(),
        shifts_a.view(),
        ref_a,
        ref_offsets_a,
        pad_char,
        -1,
        parallel,
        false,
        out_data,
        out_offsets,
    );
}
