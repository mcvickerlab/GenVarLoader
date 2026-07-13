//! Single-haplotype reconstruction core (pure ndarray). PyO3 lives in `crate::ffi`.
//!
//! Mirrors `reconstruct_haplotype_from_sparse` in
//! `python/genvarloader/_dataset/_genotypes.py:277-465` statement-by-statement.
use ndarray::{s, ArrayView1, ArrayView2, ArrayViewMut1};
use rayon::prelude::*;

/// Single-haplotype inner kernel, generic over the variant source.
///
/// Mirror of numba `reconstruct_haplotype_from_sparse` (`_genotypes.py:277-465`), with
/// the per-variant reads factored out behind the `provide` closure so the same loop
/// body can be driven by either the SVAR 1.0 global variant table
/// (`reconstruct_haplotype_from_sparse`) or the SVAR2 two-channel (var_key ⋈ dense)
/// source (`reconstruct_haplotypes_from_svar2`). **Do not change the reconstruction
/// math here** — only the data source differs between call sites.
///
/// # Parameters
/// - `n_variants`  – number of variants this haplotype sees, i.e. `0..n_variants` are
///                   valid indices into `provide`
/// - `provide`     – `Fn(v) -> (v_pos, v_diff, allele, annot_id)` for `v in 0..n_variants`:
///                   `v_pos` genomic start, `v_diff` ilen (`alt_len - ref_len` for
///                   atomized variants), `allele` the full ALT allele bytes (borrowed or
///                   owned via `Cow`), `annot_id` the value written into `annot_v_idxs`
///                   when the variant is applied. The named lifetime `'a` ties the
///                   returned `Cow`'s borrow to the driver-call scope (`alt_flat` for
///                   SVAR1; `lut_bytes` for SVAR2) — using `Cow<'_, ...>` here instead
///                   would force a higher-ranked bound that fails to unify across the
///                   two call sites.
/// - `shift`       – total amount to shift by (i64)
/// - `ref_`        – reference contig bytes (u8)
/// - `ref_start`   – start position into the reference; may be negative (i64)
/// - `out`         – output buffer to fill (u8, length = desired haplotype length)
/// - `pad_char`    – byte used for padding where reference is unavailable
/// - `keep`        – optional per-haplotype-variant mask; `None` means use all
/// - `annot_v_idxs`  – optional annotation: `annot_id` per output position (i32; -1 = ref/pad)
/// - `annot_ref_pos` – optional annotation: reference position per output position (i32;
///                     -1 = leading pad, i32::MAX = trailing pad)
#[allow(clippy::too_many_arguments)]
fn reconstruct_haplotype_core<'a>(
    n_variants: usize,
    provide: impl Fn(usize) -> (i64, i64, std::borrow::Cow<'a, [u8]>, i32),
    shift: i64,
    ref_: ArrayView1<u8>,
    ref_start: i64,
    mut out: ArrayViewMut1<u8>,
    pad_char: u8,
    keep: Option<ArrayView1<bool>>,
    mut annot_v_idxs: Option<ArrayViewMut1<i32>>,
    mut annot_ref_pos: Option<ArrayViewMut1<i32>>,
) {
    let length = out.len() as i64;

    // Hoist contiguous-slice pointers once so the hot loops use direct byte ops
    // (fill/copy_from_slice) instead of ndarray's stride/do_slice dispatch path.
    let out_flat: &mut [u8] = out.as_slice_mut().unwrap();
    let ref_flat: &[u8] = ref_.as_slice().unwrap();
    let mut av_flat: Option<&mut [i32]> = annot_v_idxs.as_mut().and_then(|a| a.as_slice_mut());
    let mut ap_flat: Option<&mut [i32]> = annot_ref_pos.as_mut().and_then(|a| a.as_slice_mut());

    // where to get next reference subsequence
    let mut ref_idx: i64 = ref_start;
    // where to put next subsequence
    let mut out_idx: i64 = 0;
    // how much we've shifted
    let mut shifted: i64 = 0;

    // if ref_idx is negative, we need to pad the beginning of the haplotype
    if ref_idx < 0 {
        let pad_len_raw = -ref_idx;
        shifted = shift.min(pad_len_raw);
        let pad_len = pad_len_raw - shifted;
        let s = out_idx as usize;
        let e = (out_idx + pad_len) as usize;
        out_flat[s..e].fill(pad_char);
        if let Some(av) = av_flat.as_deref_mut() {
            av[s..e].fill(-1);
        }
        if let Some(ap) = ap_flat.as_deref_mut() {
            ap[s..e].fill(-1);
        }
        out_idx += pad_len;
        ref_idx = 0;
    }

    'variants: for v in 0..n_variants {
        if let Some(ref k) = keep {
            if !k[v] {
                continue;
            }
        }

        let (v_pos, v_diff, allele_cow, annot_id) = provide(v);
        let allele_full: &[u8] = allele_cow.as_ref();
        let v_len_full = allele_full.len() as i64;
        // +1 assumes atomized variants, exactly 1 nt shared between REF and ALT
        let v_ref_end: i64 = v_pos - 0i64.min(v_diff) + 1;

        // if variant is a DEL spanning start of query
        if v_pos < ref_start && v_diff < 0 && v_ref_end >= ref_start {
            ref_idx = v_ref_end;
            continue;
        }

        // overlapping variants
        // v_pos < ref_idx only if we see an ALT at a given position a second
        // time or more. We'll do what bcftools consensus does and only use the
        // first ALT variant we find.
        if v_pos < ref_idx {
            continue;
        }

        // handle shift
        // allele_start_idx tracks how much of the allele to skip (0 by default)
        let mut allele_start_idx: i64 = 0;
        if shifted < shift {
            let ref_shift_dist = v_pos - ref_idx;
            // not enough distance to finish the shift even with the variant
            if shifted + ref_shift_dist + v_len_full < shift {
                // skip the variant
                continue 'variants;
            }
            // enough distance between ref_idx and start of variant to finish shift
            else if shifted + ref_shift_dist >= shift {
                ref_idx += shift - shifted;
                shifted = shift;
                // can still use the variant and whatever ref is left between
                // ref_idx and the variant
            }
            // ref + all or some of variant is enough to finish shift
            else {
                // how much left to shift - amount of ref we can use
                allele_start_idx = shift - shifted - ref_shift_dist;
                shifted = shift;
                // enough dist with variant to complete shift
                if allele_start_idx == v_len_full {
                    // move ref to end of variant
                    ref_idx = v_ref_end;
                    // skip the variant
                    continue 'variants;
                }
                // consume ref up to beginning of variant
                // ref_idx will be moved to end of variant after using the variant
                ref_idx = v_pos;
                // adjust variant to start at allele_start_idx — done via offset below
            }
        }

        // Working allele slice (may start at allele_start_idx after shift consumption)
        let allele = &allele_full[allele_start_idx as usize..];
        let v_len = allele.len() as i64;

        // add reference sequence
        let ref_len = v_pos - ref_idx;
        if out_idx + ref_len >= length {
            // ref will get written by final clause
            // handles case where extraneous variants downstream of the haplotype were provided
            break;
        }
        {
            let os = out_idx as usize;
            let oe = (out_idx + ref_len) as usize;
            let rs = ref_idx as usize;
            let re = (ref_idx + ref_len) as usize;
            out_flat[os..oe].copy_from_slice(&ref_flat[rs..re]);
            if let Some(av) = av_flat.as_deref_mut() {
                av[os..oe].fill(-1);
            }
            if let Some(ap) = ap_flat.as_deref_mut() {
                // arange(ref_idx, ref_idx + ref_len)
                for (j, pos) in (os..oe).zip(rs..re) {
                    ap[j] = pos as i32;
                }
            }
        }
        out_idx += ref_len;

        // apply variant
        let writable_length = v_len.min(length - out_idx);
        {
            let os = out_idx as usize;
            let oe = (out_idx + writable_length) as usize;
            out_flat[os..oe].copy_from_slice(&allele[..writable_length as usize]);
            if let Some(av) = av_flat.as_deref_mut() {
                av[os..oe].fill(annot_id);
            }
            if let Some(ap) = ap_flat.as_deref_mut() {
                ap[os..oe].fill(v_pos as i32);
            }
        }
        out_idx += writable_length;

        // advance ref_idx to end of variant
        ref_idx = v_ref_end;

        if out_idx >= length {
            break;
        }
    }

    if shifted < shift {
        // need to shift the rest of the track
        ref_idx += shift - shifted;
        ref_idx = ref_idx.min(ref_flat.len() as i64);
        shifted = shift;
    }
    let _ = shifted; // used above, silence unused-assign warning

    // fill rest with reference sequence and right-pad with Ns
    let unfilled_length = length - out_idx;
    if unfilled_length > 0 {
        // fill with reference sequence; when ref_idx is past the contig end,
        // writable_ref <= 0 and the tail out[out_idx..length] is right-padded.
        let writable_ref = unfilled_length.min(ref_flat.len() as i64 - ref_idx);
        // Positive: copy ref bytes from ref_idx. Zero or negative: no-op.
        let out_end_idx = if writable_ref > 0 {
            let oe = out_idx + writable_ref;
            let re = ref_idx + writable_ref;
            {
                let os = out_idx as usize;
                let oe_u = oe as usize;
                let rs = ref_idx as usize;
                let re_u = re as usize;
                out_flat[os..oe_u].copy_from_slice(&ref_flat[rs..re_u]);
                if let Some(av) = av_flat.as_deref_mut() {
                    av[os..oe_u].fill(-1);
                }
                if let Some(ap) = ap_flat.as_deref_mut() {
                    for (j, pos) in (os..oe_u).zip(rs..re_u) {
                        ap[j] = pos as i32;
                    }
                }
            }
            oe
        } else {
            // writable_ref <= 0: ref exhausted (ref_idx at/after contig end).
            // No reference bytes remain to copy, so the entire unfilled tail
            // out[out_idx..length] must be padded. Clamp out_end_idx to out_idx
            // (NOT 0) so the right-pad below fills exactly out[out_idx..length]
            // and never overwrites already-written positions.
            out_idx
        };

        // right-pad
        if out_end_idx < length {
            let pe = length as usize;
            let ps = out_end_idx as usize;
            out_flat[ps..pe].fill(pad_char);
            if let Some(av) = av_flat.as_deref_mut() {
                av[ps..pe].fill(-1);
            }
            if let Some(ap) = ap_flat.as_deref_mut() {
                ap[ps..pe].fill(i32::MAX);
            }
        }
    }
}

/// Reconstruct a single haplotype from reference sequence and variants (SVAR 1.0
/// global variant table). Thin wrapper over [`reconstruct_haplotype_core`]: builds a
/// closure that indexes the global table and delegates. **Behavior is unchanged** from
/// before the closure-source refactor — this only changes where the per-variant data
/// comes from.
///
/// # Parameters
/// - `v_idxs`      – indices into the full variant table for this haplotype (i32)
/// - `v_starts`    – genomic start position of each variant (i32, indexed by variant)
/// - `ilens`       – insertion-length (ilen = alt_len − ref_len + 1) per variant (i32)
/// - `shift`       – total amount to shift by (i64)
/// - `alt_alleles` – packed ALT allele bytes for all variants (u8)
/// - `alt_offsets` – byte offsets into `alt_alleles`; length = total_variants + 1 (i64)
/// - `ref_`        – reference contig bytes (u8)
/// - `ref_start`   – start position into the reference; may be negative (i64)
/// - `out`         – output buffer to fill (u8, length = desired haplotype length)
/// - `pad_char`    – byte used for padding where reference is unavailable
/// - `keep`        – optional per-haplotype-variant mask; `None` means use all
/// - `annot_v_idxs`  – optional annotation: variant index per output position (i32; -1 = ref/pad)
/// - `annot_ref_pos` – optional annotation: reference position per output position (i32;
///                     -1 = leading pad, i32::MAX = trailing pad)
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotype_from_sparse(
    v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    shift: i64,
    alt_alleles: ArrayView1<u8>,
    alt_offsets: ArrayView1<i64>,
    ref_: ArrayView1<u8>,
    ref_start: i64,
    out: ArrayViewMut1<u8>,
    pad_char: u8,
    keep: Option<ArrayView1<bool>>,
    annot_v_idxs: Option<ArrayViewMut1<i32>>,
    annot_ref_pos: Option<ArrayViewMut1<i32>>,
) {
    let alt_flat: &[u8] = alt_alleles.as_slice().unwrap();
    let provide = |v: usize| {
        let variant = v_idxs[v] as usize;
        let ao_s = alt_offsets[variant] as usize;
        let ao_e = alt_offsets[variant + 1] as usize;
        (
            v_starts[variant] as i64,
            ilens[variant] as i64,
            std::borrow::Cow::Borrowed(&alt_flat[ao_s..ao_e]),
            variant as i32,
        )
    };
    reconstruct_haplotype_core(
        v_idxs.len(),
        provide,
        shift,
        ref_,
        ref_start,
        out,
        pad_char,
        keep,
        annot_v_idxs,
        annot_ref_pos,
    );
}

/// Batch driver: reconstruct haplotypes for all (query, hap) pairs.
///
/// Mirrors `reconstruct_haplotypes_from_sparse` (plural) in
/// `python/genvarloader/_dataset/_genotypes.py`.
///
/// # Parameters
/// - `out` – flat output buffer, length = out_offsets[-1] (u8); written in place
/// - `out_offsets` – shape (batch*ploidy + 1,) offsets into `out`
/// - `regions` – shape (batch, 3) as (contig_idx, start, end) i32
/// - `shifts` – shape (batch, ploidy) i32
/// - `geno_offset_idx` – shape (batch, ploidy) i64 indices into geno_o_starts/stops
/// - `geno_o_starts` – shape (n,) i64 — row(0) of normalized (2,n) geno_offsets
/// - `geno_o_stops` – shape (n,) i64 — row(1) of normalized (2,n) geno_offsets
/// - `geno_v_idxs` – flat sparse genotype variant indices i32
/// - `v_starts` – variant genomic start positions i32
/// - `ilens` – variant insertion lengths i32
/// - `alt_alleles` – packed ALT allele bytes u8
/// - `alt_offsets` – offsets into alt_alleles i64
/// - `ref_` – packed reference bytes u8
/// - `ref_offsets` – per-contig offsets into ref_ i64
/// - `pad_char` – padding byte u8
/// - `keep` – optional flat keep mask bool
/// - `keep_offsets` – optional 1D (batch*ploidy + 1) offsets into keep i64
/// - `annot_v_idxs` – optional annotation output i32 (same layout as out)
/// - `annot_ref_pos` – optional annotation output i32 (same layout as out)
/// - `parallel` – if true, use rayon to process work items concurrently
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_sparse(
    mut out: ArrayViewMut1<u8>,
    out_offsets: ArrayView1<i64>,
    regions: ArrayView2<i32>,
    shifts: ArrayView2<i32>,
    geno_offset_idx: ArrayView2<i64>,
    geno_o_starts: ArrayView1<i64>,
    geno_o_stops: ArrayView1<i64>,
    geno_v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    alt_alleles: ArrayView1<u8>,
    alt_offsets: ArrayView1<i64>,
    ref_: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    keep: Option<ArrayView1<bool>>,
    keep_offsets: Option<ArrayView1<i64>>,
    mut annot_v_idxs: Option<ArrayViewMut1<i32>>,
    mut annot_ref_pos: Option<ArrayViewMut1<i32>>,
    parallel: bool,
) {
    let batch_size = regions.nrows();
    let ploidy = shifts.ncols();
    let n_work = batch_size * ploidy;

    // Per-k inner work: given disjoint output slices, call the single-haplotype kernel.
    // All read-only ArrayViews are Send+Sync so the closure can borrow them freely.
    let do_work = |k: usize,
                   out_view: ArrayViewMut1<u8>,
                   av_view: Option<ArrayViewMut1<i32>>,
                   ap_view: Option<ArrayViewMut1<i32>>| {
        let query = k / ploidy;
        let hap = k % ploidy;

        // geno slice for this (query, hap)
        let o_idx = geno_offset_idx[[query, hap]] as usize;
        let o_s = geno_o_starts[o_idx] as usize;
        let o_e = geno_o_stops[o_idx] as usize;
        let qh_v_idxs = geno_v_idxs.slice(s![o_s..o_e]);

        // keep slice
        let qh_keep: Option<ArrayView1<bool>> =
            if let (Some(ref k_arr), Some(ref ko)) = (&keep, &keep_offsets) {
                let ks = ko[k] as usize;
                let ke = ko[k + 1] as usize;
                Some(k_arr.slice(s![ks..ke]))
            } else {
                None
            };

        // region info
        let c_idx = regions[[query, 0]] as usize;
        let c_s = ref_offsets[c_idx] as usize;
        let c_e = ref_offsets[c_idx + 1] as usize;
        let contig_ref = ref_.slice(s![c_s..c_e]);
        let ref_start = regions[[query, 1]] as i64;
        let shift = shifts[[query, hap]] as i64;

        reconstruct_haplotype_from_sparse(
            qh_v_idxs,
            v_starts,
            ilens,
            shift,
            alt_alleles,
            alt_offsets,
            contig_ref,
            ref_start,
            out_view,
            pad_char,
            qh_keep,
            av_view,
            ap_view,
        );
    };

    if parallel {
        // Build disjoint per-k mutable slices for all active buffers using the
        // proven split_at_mut chain idiom (mirrors get_reference in reference/mod.rs).
        // &mut [_] slices are Send, unlike raw *mut pointers — safe for rayon closures.
        let bounds: Vec<(usize, usize)> = (0..n_work)
            .map(|k| (out_offsets[k] as usize, out_offsets[k + 1] as usize))
            .collect();

        let out_slice = out.as_slice_mut().unwrap();
        let mut out_chunks: Vec<&mut [u8]> = Vec::with_capacity(n_work);
        {
            let mut rest = &mut out_slice[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
                // Contract: `out_offsets` is monotonically non-decreasing, so each
                // work item's range starts at or after the previous one's end. This
                // guarantees `s - cursor` does not underflow and the carved slices
                // are disjoint. The same `bounds` drives the annotation carves below.
                debug_assert!(
                    s >= cursor && e >= s,
                    "out_offsets must be monotonically non-decreasing (got s={s}, e={e}, cursor={cursor})"
                );
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                out_chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
        }

        // Carve annotation buffers only when they are Some.
        let av_chunks: Option<Vec<&mut [i32]>> = annot_v_idxs.as_mut().map(|av| {
            let av_slice = av.as_slice_mut().unwrap();
            let mut chunks: Vec<&mut [i32]> = Vec::with_capacity(n_work);
            let mut rest = &mut av_slice[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
            chunks
        });

        let ap_chunks: Option<Vec<&mut [i32]>> = annot_ref_pos.as_mut().map(|ap| {
            let ap_slice = ap.as_slice_mut().unwrap();
            let mut chunks: Vec<&mut [i32]> = Vec::with_capacity(n_work);
            let mut rest = &mut ap_slice[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
            chunks
        });

        // Zip all chunk vecs and dispatch in parallel.
        // Handle the four combinations of av/ap presence.
        match (av_chunks, ap_chunks) {
            (Some(avc), Some(apc)) => {
                out_chunks
                    .into_par_iter()
                    .zip(avc.into_par_iter())
                    .zip(apc.into_par_iter())
                    .enumerate()
                    .for_each(|(k, ((out_chunk, av_chunk), ap_chunk))| {
                        do_work(
                            k,
                            ArrayViewMut1::from(out_chunk),
                            Some(ArrayViewMut1::from(av_chunk)),
                            Some(ArrayViewMut1::from(ap_chunk)),
                        );
                    });
            }
            (Some(avc), None) => {
                out_chunks
                    .into_par_iter()
                    .zip(avc.into_par_iter())
                    .enumerate()
                    .for_each(|(k, (out_chunk, av_chunk))| {
                        do_work(
                            k,
                            ArrayViewMut1::from(out_chunk),
                            Some(ArrayViewMut1::from(av_chunk)),
                            None,
                        );
                    });
            }
            (None, Some(apc)) => {
                out_chunks
                    .into_par_iter()
                    .zip(apc.into_par_iter())
                    .enumerate()
                    .for_each(|(k, (out_chunk, ap_chunk))| {
                        do_work(
                            k,
                            ArrayViewMut1::from(out_chunk),
                            None,
                            Some(ArrayViewMut1::from(ap_chunk)),
                        );
                    });
            }
            (None, None) => {
                out_chunks
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(k, out_chunk)| {
                        do_work(k, ArrayViewMut1::from(out_chunk), None, None);
                    });
            }
        }
    } else {
        // Serial path: use raw pointers for disjoint sub-range access, exactly as before.
        // The serial loop prevents concurrent aliasing.
        let out_raw: *mut u8 = out.as_mut_ptr();
        let av_raw: Option<*mut i32> = annot_v_idxs.as_mut().map(|a| a.as_mut_ptr());
        let ap_raw: Option<*mut i32> = annot_ref_pos.as_mut().map(|a| a.as_mut_ptr());

        for k in 0..n_work {
            let out_s = out_offsets[k] as usize;
            let out_e = out_offsets[k + 1] as usize;
            debug_assert!(
                out_e >= out_s,
                "out_offsets must be monotonically non-decreasing (got out_s={out_s}, out_e={out_e})"
            );

            // SAFETY: `out_offsets` is required by the calling contract to be monotonically
            // non-decreasing, so consecutive (out_s, out_e) pairs are strictly non-overlapping
            // address ranges within the same allocation.  Because the loop is serial there are
            // no concurrent borrows, so constructing a `&mut [u8]` from each disjoint sub-range
            // is free of aliasing UB.
            let out_chunk =
                unsafe { std::slice::from_raw_parts_mut(out_raw.add(out_s), out_e - out_s) };
            let out_view = ArrayViewMut1::from(out_chunk);

            // SAFETY: same invariant as out_chunk — `out_offsets` non-decreasing guarantees
            // each [out_s..out_e] is a disjoint sub-range; serial loop prevents concurrent
            // aliasing.
            let av_view: Option<ArrayViewMut1<i32>> = av_raw.map(|p| {
                let chunk = unsafe { std::slice::from_raw_parts_mut(p.add(out_s), out_e - out_s) };
                ArrayViewMut1::from(chunk)
            });

            // SAFETY: same invariant as out_chunk — `out_offsets` non-decreasing guarantees
            // each [out_s..out_e] is a disjoint sub-range; serial loop prevents concurrent
            // aliasing.
            let ap_view: Option<ArrayViewMut1<i32>> = ap_raw.map(|p| {
                let chunk = unsafe { std::slice::from_raw_parts_mut(p.add(out_s), out_e - out_s) };
                ArrayViewMut1::from(chunk)
            });

            do_work(k, out_view, av_view, ap_view);
        }
    }
}

/// Batch driver: reconstruct haplotypes for all (query, hap) pairs from the SVAR2
/// two-channel (var_key ⋈ dense) source. Mirrors
/// [`reconstruct_haplotypes_from_sparse`]'s parallel/serial disjoint-slice carving
/// exactly — only the per-work-item variant source and decode differ. Additive: SVAR
/// 1.0's `reconstruct_haplotype[s]_from_sparse` are untouched by this function.
///
/// # Parameters
/// - `out` – flat output buffer, length = out_offsets[-1] (u8); written in place
/// - `out_offsets` – shape (batch*ploidy + 1,) offsets into `out`
/// - `regions` – shape (batch, 3) as (contig_idx, start, end) i32
/// - `shifts` – shape (batch, ploidy) i32
/// - `vk_pos` / `vk_key` – this hap's `var_key` channel: position + uniform key (i32)
/// - `vk_off` – shape (n_work + 1) CSR offsets into `vk_pos`/`vk_key`, indexed by flat
///   work index `k = query * ploidy + hap`
/// - `dense_pos` / `dense_key` – the shared `dense` channel: position + uniform key (i32)
/// - `dense_range` – shape (batch, 2) as `[ds, de)`, the window into `dense_pos`/`dense_key`
///   for each query row
/// - `dense_present` – packed presence bits over the dense window, LSB-first per byte (u8)
/// - `dense_present_off` – shape (n_work + 1) BIT offsets into `dense_present`, indexed by
///   flat work index `k`
/// - `lut_bytes` / `lut_off` – long-allele-bank bytes + CSR offsets, for `Lookup` keys
/// - `ref_` – packed reference bytes u8
/// - `ref_offsets` – per-contig offsets into ref_ i64
/// - `pad_char` – padding byte u8
/// - `parallel` – if true, use rayon to process work items concurrently
///
/// SVAR2 read-bound haplotypes are never annotated (annotated `.svar2` output is
/// `NotImplementedError`-guarded), so this kernel produces only un-annotated
/// sequence; the shared `reconstruct_haplotype_core` still carries the optional
/// annotation outputs for the SVAR1 path.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_svar2(
    mut out: ArrayViewMut1<u8>,
    out_offsets: ArrayView1<i64>,
    regions: ArrayView2<i32>,
    shifts: ArrayView2<i32>,
    vk_pos: ArrayView1<i32>,
    vk_key: ArrayView1<i32>,
    vk_off: ArrayView1<i64>,
    dense_pos: ArrayView1<i32>,
    dense_key: ArrayView1<i32>,
    dense_range: ArrayView2<i32>,
    dense_present: ArrayView1<u8>,
    dense_present_off: ArrayView1<i64>,
    lut_bytes: ArrayView1<u8>,
    lut_off: ArrayView1<i64>,
    ref_: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
) {
    let batch_size = regions.nrows();
    let ploidy = shifts.ncols();
    let n_work = batch_size * ploidy;

    // Hoist contiguous-slice pointers once, exactly as SVAR1's do_work captures
    // read-only views — these are Send+Sync so the rayon parallel path is unchanged.
    let vk_pos_s: &[i32] = vk_pos.as_slice().unwrap();
    let vk_key_s: &[i32] = vk_key.as_slice().unwrap();
    let dense_pos_s: &[i32] = dense_pos.as_slice().unwrap();
    let dense_key_s: &[i32] = dense_key.as_slice().unwrap();
    let dense_present_s: &[u8] = dense_present.as_slice().unwrap();
    let lut_bytes_s: &[u8] = lut_bytes.as_slice().unwrap();
    let lut_off_s: &[i64] = lut_off.as_slice().unwrap();

    // Per-k inner work: merge this hap's var_key ⋈ dense entries, then reconstruct via
    // the shared core with a decode closure. All read-only ArrayViews/slices are
    // Send+Sync so the closure can borrow them freely.
    let do_work = |k: usize, out_view: ArrayViewMut1<u8>| {
        let query = k / ploidy;
        let hap = k % ploidy;

        // region/ref
        let c_idx = regions[[query, 0]] as usize;
        let c_s = ref_offsets[c_idx] as usize;
        let c_e = ref_offsets[c_idx + 1] as usize;
        let contig_ref = ref_.slice(s![c_s..c_e]);
        let ref_start = regions[[query, 1]] as i64;
        let shift = shifts[[query, hap]] as i64;

        // var_key window for this hap
        let vk_lo = vk_off[k] as usize;
        let vk_hi = vk_off[k + 1] as usize;

        // dense window for this query
        let ds = dense_range[[query, 0]] as usize;
        let de = dense_range[[query, 1]] as usize;

        // presence bits for this hap start at bit `dense_present_off[k]`
        let base_bit = dense_present_off[k] as usize;
        let present_bit = |j: usize| -> bool {
            let bit = base_bit + j;
            (dense_present_s[bit / 8] >> (bit % 8)) & 1 == 1 // LSB-first within each byte
        };

        let merged = crate::svar2::merge_hap(
            vk_pos_s,
            vk_key_s,
            vk_lo,
            vk_hi,
            dense_pos_s,
            dense_key_s,
            ds,
            de,
            present_bit,
        );

        let contig_ref_s: &[u8] = contig_ref.as_slice().unwrap();
        let provide = |v: usize| {
            let (pos, key) = merged[v];
            let (v_diff, allele) = crate::svar2::decode_alt(key, lut_bytes_s, lut_off_s);
            // A pure DEL decodes to an empty ALT: genoray stores no anchor base in the key
            // (recovered from the reference downstream). But the reconstruction kernel writes
            // `allele_full` as the replacement and advances ref by |v_diff|+1, so it needs the
            // anchor base = ref[pos] present; an empty allele would delete the anchor too
            // (off-by-one: -(|v_diff|+1) instead of -|v_diff|). SNP/INS/Lookup alleles are
            // already non-empty and pass through unchanged.
            let allele = if allele.is_empty() {
                // The anchor base is `ref[pos]`. Valid gathered variants are within-contig
                // records (`pos < contig_ref_s.len()`); a `pos` past the contig end can only
                // come from a corrupt store. Assert it in debug/test builds rather than
                // silently reading OOB (release keeps the raw slice for speed).
                debug_assert!(
                    (pos as usize) < contig_ref_s.len(),
                    "pure-DEL anchor position {pos} is beyond contig ref length {} (corrupt store?)",
                    contig_ref_s.len()
                );
                std::borrow::Cow::Borrowed(&contig_ref_s[pos as usize..pos as usize + 1])
            } else {
                allele
            };
            (pos as i64, v_diff, allele, v as i32)
        };

        reconstruct_haplotype_core(
            merged.len(),
            provide,
            shift,
            contig_ref,
            ref_start,
            out_view,
            pad_char,
            None, // keep: SVAR2 has no per-haplotype keep mask
            None, // annot_v_idxs: SVAR2 read-bound haps are never annotated
            None, // annot_ref_pos: SVAR2 read-bound haps are never annotated
        );
    };

    if parallel {
        // Build disjoint per-k mutable slices for all active buffers using the
        // proven split_at_mut chain idiom (mirrors get_reference in reference/mod.rs).
        // &mut [_] slices are Send, unlike raw *mut pointers — safe for rayon closures.
        let bounds: Vec<(usize, usize)> = (0..n_work)
            .map(|k| (out_offsets[k] as usize, out_offsets[k + 1] as usize))
            .collect();

        let out_slice = out.as_slice_mut().unwrap();
        let mut out_chunks: Vec<&mut [u8]> = Vec::with_capacity(n_work);
        {
            let mut rest = &mut out_slice[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
                // Contract: `out_offsets` is monotonically non-decreasing, so each
                // work item's range starts at or after the previous one's end. This
                // guarantees `s - cursor` does not underflow and the carved slices
                // are disjoint. The same `bounds` drives the annotation carves below.
                debug_assert!(
                    s >= cursor && e >= s,
                    "out_offsets must be monotonically non-decreasing (got s={s}, e={e}, cursor={cursor})"
                );
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                out_chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
        }

        // SVAR2 read-bound haps are never annotated, so dispatch the un-annotated
        // work items straight across rayon.
        out_chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(k, out_chunk)| {
                do_work(k, ArrayViewMut1::from(out_chunk));
            });
    } else {
        // Serial path: use raw pointers for disjoint sub-range access, exactly as before.
        // The serial loop prevents concurrent aliasing.
        let out_raw: *mut u8 = out.as_mut_ptr();

        for k in 0..n_work {
            let out_s = out_offsets[k] as usize;
            let out_e = out_offsets[k + 1] as usize;
            debug_assert!(
                out_e >= out_s,
                "out_offsets must be monotonically non-decreasing (got out_s={out_s}, out_e={out_e})"
            );

            // SAFETY: `out_offsets` is required by the calling contract to be monotonically
            // non-decreasing, so consecutive (out_s, out_e) pairs are strictly non-overlapping
            // address ranges within the same allocation.  Because the loop is serial there are
            // no concurrent borrows, so constructing a `&mut [u8]` from each disjoint sub-range
            // is free of aliasing UB.
            let out_chunk =
                unsafe { std::slice::from_raw_parts_mut(out_raw.add(out_s), out_e - out_s) };
            do_work(k, ArrayViewMut1::from(out_chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1};

    /// Helper: run the kernel and return (out, annot_v_idxs, annot_ref_pos)
    fn run(
        v_idxs: &[i32],
        v_starts: &[i32],
        ilens: &[i32],
        shift: i64,
        alt_alleles: &[u8],
        alt_offsets: &[i64],
        ref_: &[u8],
        ref_start: i64,
        out_len: usize,
        pad_char: u8,
        keep: Option<&[bool]>,
        annotate: bool,
    ) -> (Vec<u8>, Vec<i32>, Vec<i32>) {
        let mut out = Array1::<u8>::from_elem(out_len, pad_char);
        let mut av = Array1::<i32>::from_elem(out_len, 0i32);
        let mut ap = Array1::<i32>::from_elem(out_len, 0i32);

        let keep_arr: Option<Array1<bool>> = keep.map(|k| arr1(k));

        if annotate {
            reconstruct_haplotype_from_sparse(
                arr1(v_idxs).view(),
                arr1(v_starts).view(),
                arr1(ilens).view(),
                shift,
                arr1(alt_alleles).view(),
                arr1(alt_offsets).view(),
                arr1(ref_).view(),
                ref_start,
                out.view_mut(),
                pad_char,
                keep_arr.as_ref().map(|k| k.view()),
                Some(av.view_mut()),
                Some(ap.view_mut()),
            );
        } else {
            reconstruct_haplotype_from_sparse(
                arr1(v_idxs).view(),
                arr1(v_starts).view(),
                arr1(ilens).view(),
                shift,
                arr1(alt_alleles).view(),
                arr1(alt_offsets).view(),
                arr1(ref_).view(),
                ref_start,
                out.view_mut(),
                pad_char,
                keep_arr.as_ref().map(|k| k.view()),
                None,
                None,
            );
        }
        (out.to_vec(), av.to_vec(), ap.to_vec())
    }

    // -------------------------------------------------------------------------
    // Case 1: no variants, shift=0, in-bounds
    // ref = [10,20,30,40,50], ref_start=1, out_len=3 → [20,30,40]
    // -------------------------------------------------------------------------
    #[test]
    fn no_variants_shift0_in_bounds() {
        let (out, _av, _ap) = run(
            &[],     // v_idxs
            &[],     // v_starts (indexed by variant)
            &[],     // ilens
            0,       // shift
            &[],     // alt_alleles
            &[0i64], // alt_offsets (1 sentinel for 0 variants)
            &[10, 20, 30, 40, 50],
            1, // ref_start
            3, // out_len
            0, // pad_char
            None,
            false,
        );
        assert_eq!(out, vec![20, 30, 40]);
    }

    // -------------------------------------------------------------------------
    // Case 2: negative ref_start → leading pad, annot_ref_pos == -1 over the pad
    // ref = [1,2,3,4,5], ref_start=-2, out_len=5, pad=9
    // → [9,9,1,2,3], annot_ref_pos over pad = [-1,-1,0,1,2]
    // -------------------------------------------------------------------------
    #[test]
    fn negative_ref_start_leading_pad() {
        let (out, av, ap) = run(
            &[],
            &[],
            &[],
            0,
            &[],
            &[0i64],
            &[1, 2, 3, 4, 5],
            -2, // ref_start
            5,
            9,
            None,
            true,
        );
        assert_eq!(out, vec![9, 9, 1, 2, 3]);
        assert_eq!(&av[..2], &[-1i32, -1]);
        assert_eq!(
            &ap[..2],
            &[-1i32, -1],
            "leading pad annot_ref_pos must be -1"
        );
        assert_eq!(&ap[2..], &[0i32, 1, 2]);
    }

    // -------------------------------------------------------------------------
    // Case 3: single SNP (ilen=0)
    // ref   = [A,C,G,T,A] = [65,67,71,84,65], ref_start=0, out_len=5
    // variant 0: pos=2, ilen=0, allele=[84] (T replaces G)
    // v_idxs=[0], v_starts=[2], ilens=[0], alt_alleles=[84], alt_offsets=[0,1]
    // expected out: [65,67,84,84,65]  (ref_end = 2 - min(0,0) + 1 = 3)
    // -------------------------------------------------------------------------
    #[test]
    fn single_snp() {
        // ref: A C G T A (positions 0..5)
        // variant at pos=2 (G→T), ilen=0 → v_ref_end = 2 - 0 + 1 = 3
        // out: A C [T] T A
        let (out, av, _ap) = run(
            &[0],                  // v_idxs: only variant 0
            &[2],                  // v_starts: variant 0 is at pos 2
            &[0],                  // ilens: SNP, no length change
            0,                     // shift
            &[84u8],               // alt_alleles: T
            &[0i64, 1],            // alt_offsets
            &[65, 67, 71, 84, 65], // A C G T A
            0,                     // ref_start
            5,
            0,
            None,
            true,
        );
        // ref[0..2]=AC, allele T, ref[3..5]=TA
        assert_eq!(out, vec![65, 67, 84, 84, 65]);
        // annot_v_idxs: [-1,-1, 0, -1,-1]
        assert_eq!(av, vec![-1, -1, 0, -1, -1]);
    }

    // -------------------------------------------------------------------------
    // Case 4: 2bp insertion (ilen=+2)
    // ref = [1,2,3,4,5], ref_start=0, out_len=5
    // variant at pos=2, ilen=+2, allele=[10,11,12] (3 bytes: REF anchor + 2 inserted)
    // v_ref_end = 2 - min(0,+2) + 1 = 3
    // Processing: ref[0..2]=[1,2], allele=[10,11,12] → 3 bytes, but out only has 1 slot left
    //   after 2 ref bytes → writes 3 bytes clipped to min(3, 5-2)=3: [10,11,12]
    //   out = [1,2,10,11,12]
    // -------------------------------------------------------------------------
    #[test]
    fn two_bp_insertion() {
        let (out, _av, _ap) = run(
            &[0],
            &[2], // variant 0 at pos 2
            &[2], // ilen=+2
            0,
            &[10u8, 11, 12],
            &[0i64, 3],
            &[1, 2, 3, 4, 5],
            0,
            5,
            0,
            None,
            false,
        );
        // ref[0..2]=[1,2], allele[0..3]=[10,11,12] (writable_length=min(3,3)=3)
        // v_ref_end=3, out_idx=5, break. Final clause: unfilled=0.
        assert_eq!(out, vec![1, 2, 10, 11, 12]);
    }

    // -------------------------------------------------------------------------
    // Case 5: deletion (ilen=-2)
    // ref = [1,2,3,4,5,6,7], ref_start=0, out_len=5
    // variant at pos=2, ilen=-2, allele=[30] (1 byte, anchor only)
    // v_ref_end = 2 - min(0,-2) + 1 = 2+2+1 = 5
    // Processing: ref[0..2]=[1,2], allele=[30] (1 byte), ref_idx→5
    //   remaining ref[5..7]=[6,7], out=[1,2,30,6,7]
    // -------------------------------------------------------------------------
    #[test]
    fn deletion() {
        let (out, _av, _ap) = run(
            &[0],
            &[2],  // variant 0 at pos 2
            &[-2], // ilen=-2
            0,
            &[30u8], // anchor allele byte
            &[0i64, 1],
            &[1, 2, 3, 4, 5, 6, 7],
            0,
            5,
            0,
            None,
            false,
        );
        // ref[0..2]=[1,2], allele=[30], ref_idx→5, then ref[5..7]=[6,7]
        assert_eq!(out, vec![1, 2, 30, 6, 7]);
    }

    // -------------------------------------------------------------------------
    // Case 6: DEL spanning ref_start
    // ref = [1,2,3,4,5,6,7], ref_start=3
    // variant: v_pos=1, ilen=-3, allele=[99]
    //   v_ref_end = 1 - min(0,-3) + 1 = 1+3+1 = 5
    //   condition: v_pos(1) < ref_start(3), v_diff(-3) < 0, v_ref_end(5) >= ref_start(3)
    //   → ref_idx = 5, continue
    // Then final clause fills ref[5..7]=[6,7] + right-pad
    // out_len=5: ref[5..7]→[6,7], right-pad [0,0,0]
    // -------------------------------------------------------------------------
    #[test]
    fn del_spanning_ref_start() {
        let (out, _av, ap) = run(
            &[0],
            &[1],  // v_pos=1
            &[-3], // ilen=-3
            0,
            &[99u8],
            &[0i64, 1],
            &[1, 2, 3, 4, 5, 6, 7],
            3, // ref_start=3
            5,
            0,
            None,
            true,
        );
        // ref_idx set to 5. Final: ref[5..7]=[6,7], pad [0,0]
        assert_eq!(out, vec![6, 7, 0, 0, 0]);
        // trailing pad annot_ref_pos must be i32::MAX
        assert_eq!(&ap[2..], &[i32::MAX, i32::MAX, i32::MAX]);
    }

    // -------------------------------------------------------------------------
    // Case: deletion drives ref_idx past the contig end (overshoot).
    // ref = [1,2,3,4] (len 4), ref_start=0, out_len=8.
    // variant at pos=2, ilen=-5, allele=[50] (anchor).
    //   v_ref_end = 2 - min(0,-5) + 1 = 8  → ref_idx advances to 8 (> len 4).
    // Processing: ref[0..2]=[1,2], allele=[50] → out_idx=3.
    // Final clause: unfilled=5, ref exhausted (writable_ref = min(5, 4-8) = -4 <= 0).
    // CORRECT: no ref left → pad the whole tail → [1,2,50,0,0,0,0,0].
    // (Pre-fix rust over-pads from index 0 → all zeros.)
    // -------------------------------------------------------------------------
    #[test]
    fn overshoot_ref_past_contig() {
        let (out, _av, _ap) = run(
            &[0],
            &[2],    // v_pos=2
            &[-5],   // ilen=-5 (deletion past contig end)
            0,       // shift
            &[50u8], // anchor allele
            &[0i64, 1],
            &[1, 2, 3, 4], // ref, len 4
            0,             // ref_start
            8,             // out_len
            0,             // pad_char
            None,
            false,
        );
        assert_eq!(out, vec![1, 2, 50, 0, 0, 0, 0, 0]);
    }

    // -------------------------------------------------------------------------
    // Case 7: overlapping ALTs — only first applied
    // ref = [1,2,3,4,5], ref_start=0, out_len=5
    // v_idxs=[0,1]: two variants both at pos=2, but second has v_pos < ref_idx after first
    // variant 0: pos=2, ilen=0, allele=[20]
    // variant 1: pos=2, ilen=0, allele=[30] — overlapping, must be skipped
    // expected: [1,2,20,4,5]
    // -------------------------------------------------------------------------
    #[test]
    fn overlapping_alts_first_applied() {
        let (out, _av, _ap) = run(
            &[0, 1], // v_idxs: variants 0 then 1
            &[2, 2], // both at pos=2
            &[0, 0], // both SNPs
            0,
            &[20u8, 30], // alleles: 20 and 30
            &[0i64, 1, 2],
            &[1, 2, 3, 4, 5],
            0,
            5,
            0,
            None,
            false,
        );
        // First: ref[0..2]=[1,2], allele=[20], ref_idx→3
        // Second: v_pos=2 < ref_idx=3 → skip
        // Final: ref[3..5]=[4,5]
        assert_eq!(out, vec![1, 2, 20, 4, 5]);
    }

    // -------------------------------------------------------------------------
    // Case 8: shift consumed partly by ref + partly by allele
    // ref = [1,2,3,4,5,6,7,8], ref_start=0, shift=4, out_len=4
    // variant 0: pos=3, ilen=0, allele=[99] (SNP at pos 3)
    //   shifted=0, ref_shift_dist=3-0=3, v_len=1
    //   shifted+ref_shift_dist+v_len = 0+3+1=4 == shift=4  → NOT < 4
    //   shifted+ref_shift_dist=3 < shift=4 → "else" branch
    //   allele_start_idx = 4 - 0 - 3 = 1
    //   allele_start_idx(1) == v_len(1) → ref_idx=v_ref_end=4, continue
    // After loop: shifted(0) < shift(4) → ref_idx += 4-0=4 → ref_idx=8, min(8,8)=8
    // Final: writable_ref = min(4, 8-8)=0, out=[pad,pad,pad,pad] → all 0
    // Wait: after the early-continue in shift branch, ref_idx=4 (not 0).
    // Let me re-trace: shifted=0, ref_idx=0, v_pos=3
    //   allele_start_idx = shift(4) - shifted(0) - ref_shift_dist(3) = 1
    //   allele_start_idx(1) == v_len(1) → ref_idx = v_ref_end = 4, continue
    // After loop: shifted(0) < shift(4) → ref_idx=4+(4-0)=8, min(8,8)=8
    // Final: unfilled=4, writable_ref=min(4, 8-8)=0 → all pad
    // Better test: shift=3, variant at pos=5, allele=[99,88] (2 bytes, ilen=+1)
    //   ref_shift_dist=5, shifted+ref_shift_dist=5 >= shift=3 → first elif
    //   ref_idx += 3-0=3 → ref_idx=3, shifted=3
    //   Then ref[3..5]=[4,5], allele=[99,88], ref[7..8]=[8]
    //   out_len=4: ref[3..5]=[4,5] (2 bytes), allele=[99,88] (2 bytes) → [4,5,99,88]
    // -------------------------------------------------------------------------
    #[test]
    fn shift_consumed_partly_ref_partly_allele() {
        // shift=2, ref=[1,2,3,4,5,6], ref_start=0, variant at pos=3, allele=[99,88] (ilen=+1)
        // ref_shift_dist = 3-0 = 3, shifted+ref_shift_dist+v_len = 0+3+2 = 5 >= shift=2
        // shifted+ref_shift_dist = 3 >= shift=2 → ref_idx += 2-0=2 → ref_idx=2
        // ref[2..3]=[3], allele=[99,88], ref[4..6]=[5,6]
        // out_len=5: [3, 99, 88, 5, 6]
        let (out, _av, _ap) = run(
            &[0],
            &[3], // v_pos=3
            &[1], // ilen=+1
            2,    // shift=2
            &[99u8, 88],
            &[0i64, 2],
            &[1, 2, 3, 4, 5, 6],
            0,
            5,
            0,
            None,
            false,
        );
        // ref_idx=2 after shift, ref[2..3]=[3], allele=[99,88], v_ref_end=4, ref[4..6]=[5,6]
        assert_eq!(out, vec![3, 99, 88, 5, 6]);
    }

    // -------------------------------------------------------------------------
    // Case 8b: shift partly consumed by allele itself (allele_start_idx < v_len)
    // shift=4, ref=[1,2,3,4,5,6,7,8], ref_start=0, out_len=4
    // variant at pos=3, ilen=+1, allele=[99,88] (2 bytes)
    //   ref_shift_dist=3, shifted+ref_shift_dist+v_len = 0+3+2=5 >= shift=4
    //   shifted+ref_shift_dist=3 < shift=4 → else branch
    //   allele_start_idx = 4-0-3 = 1
    //   allele_start_idx(1) != v_len(2) → ref_idx=v_pos=3, allele=allele[1:]=[88]
    //   ref_len = v_pos(3) - ref_idx(3) = 0 (no ref before variant)
    //   allele=[88] writable_length=min(1,4)=1
    //   ref_idx → v_ref_end=4
    //   Final: ref[4..8]=[5,6,7,8], out=[88,5,6,7]
    // -------------------------------------------------------------------------
    #[test]
    fn shift_partly_consumed_by_allele() {
        let (out, _av, _ap) = run(
            &[0],
            &[3],
            &[1], // ilen=+1, allele 2 bytes
            4,    // shift=4
            &[99u8, 88],
            &[0i64, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            0,
            4,
            0,
            None,
            false,
        );
        // allele starts at index 1: [88], then ref[4..8]=[5,6,7,8] → [88,5,6,7]
        assert_eq!(out, vec![88, 5, 6, 7]);
    }

    // -------------------------------------------------------------------------
    // Case 9: right-pad clause
    // ref = [1,2,3], ref_start=0, out_len=6, no variants
    // → ref fills [1,2,3], then pad [0,0,0]
    // trailing annot_ref_pos = i32::MAX
    // -------------------------------------------------------------------------
    #[test]
    fn right_pad_clause() {
        let (out, av, ap) = run(
            &[],
            &[],
            &[],
            0,
            &[],
            &[0i64],
            &[1, 2, 3],
            0,
            6,
            0,
            None,
            true,
        );
        assert_eq!(out, vec![1, 2, 3, 0, 0, 0]);
        // ref portion: annot_v_idxs=-1, annot_ref_pos=[0,1,2]
        assert_eq!(&av[..3], &[-1i32, -1, -1]);
        assert_eq!(&ap[..3], &[0i32, 1, 2]);
        // trailing pad: annot_v_idxs=-1, annot_ref_pos=i32::MAX
        assert_eq!(&av[3..], &[-1i32, -1, -1]);
        assert_eq!(
            &ap[3..],
            &[i32::MAX, i32::MAX, i32::MAX],
            "trailing pad annot_ref_pos must be i32::MAX"
        );
    }

    // -------------------------------------------------------------------------
    // Case 11: allele_start_idx == v_len → early-continue branch
    //
    // Exercises numba _genotypes.py:390-401 / Rust mod.rs:121-131:
    //   the "else" shift sub-branch where allele_start_idx == v_len, causing
    //   ref_idx to advance to v_ref_end and the variant to be skipped.
    //
    // Hand-derivation:
    //   ref = [1..8], ref_start=0, shift=4, out_len=4
    //   SNP at v_pos=3, ilen=0, allele=[88] (v_len=1)
    //   --- shift handling (shifted=0 < shift=4) ---
    //   ref_shift_dist = v_pos - ref_idx = 3 - 0 = 3
    //   check 1: shifted + ref_shift_dist + v_len = 0+3+1 = 4  → NOT < 4, skip
    //   check 2: shifted + ref_shift_dist = 3                  → NOT >= 4, skip
    //   else: allele_start_idx = shift - shifted - ref_shift_dist = 4-0-3 = 1
    //         shifted = 4  (numba:391 / Rust:124)
    //         allele_start_idx(1) == v_len(1)                  → TRUE
    //         ref_idx = v_ref_end = 3 - min(0,0) + 1 = 4
    //         continue  (numba:397-401 / Rust:126-130)
    //   --- after loop ---
    //   shifted(4) == shift(4) → no extra advance
    //   Final fill: ref_idx=4, unfilled=4, writable_ref=min(4,8-4)=4
    //   out = ref[4..8] = [5,6,7,8]
    // -------------------------------------------------------------------------
    #[test]
    fn allele_start_idx_eq_v_len_continue() {
        let (out, _av, _ap) = run(
            &[0],       // v_idxs: only variant 0
            &[3],       // v_starts: variant 0 at pos 3
            &[0],       // ilens: SNP, ilen=0
            4,          // shift=4
            &[88u8],    // alt_allele
            &[0i64, 1], // alt_offsets
            &[1, 2, 3, 4, 5, 6, 7, 8],
            0, // ref_start
            4, // out_len
            0, // pad_char
            None,
            false,
        );
        // allele_start_idx(1) == v_len(1): variant skipped, ref_idx→4
        // shifted=4 after continue, no further shift; final fills ref[4..8]=[5,6,7,8]
        assert_eq!(out, vec![5, 6, 7, 8]);
    }

    // -------------------------------------------------------------------------
    // Case 12: skip_variant_not_enough_distance
    //
    // Exercises numba _genotypes.py:377-380 / Rust mod.rs:108-112:
    //   the "not enough distance" branch where shifted + ref_shift_dist + v_len < shift,
    //   causing the variant to be skipped entirely without advancing ref_idx.
    //
    // Hand-derivation:
    //   ref = [1..15], ref_start=0, shift=10, out_len=3
    //   SNP at v_pos=3, ilen=0, allele=[77] (v_len=1)
    //   --- shift handling (shifted=0 < shift=10) ---
    //   ref_shift_dist = v_pos - ref_idx = 3 - 0 = 3
    //   check 1: shifted + ref_shift_dist + v_len = 0+3+1 = 4 < 10  → TRUE
    //            continue  (numba:379-380 / Rust:110-112)
    //   --- after loop ---
    //   shifted(0) < shift(10) → ref_idx += 10-0 = 10, min(10,15)=10, shifted=10
    //   Final fill: ref_idx=10, unfilled=3, writable_ref=min(3,15-10)=3
    //   out = ref[10..13] = [11,12,13]
    // -------------------------------------------------------------------------
    #[test]
    fn skip_variant_not_enough_distance() {
        let ref_: Vec<u8> = (1u8..=15).collect();
        let (out, _av, _ap) = run(
            &[0],       // v_idxs: only variant 0
            &[3],       // v_starts: variant 0 at pos 3
            &[0],       // ilens: SNP, ilen=0
            10,         // shift=10
            &[77u8],    // alt_allele (never used)
            &[0i64, 1], // alt_offsets
            &ref_,
            0, // ref_start
            3, // out_len
            0, // pad_char
            None,
            false,
        );
        // variant skipped (0+3+1=4 < 10); after loop ref_idx=10; final fills [11,12,13]
        assert_eq!(out, vec![11, 12, 13]);
    }

    // -------------------------------------------------------------------------
    // Case 13: keep_mask_excludes_variant
    //
    // Exercises numba _genotypes.py:351-352 / Rust mod.rs:72-75:
    //   keep=[false, true] so variant 0 is skipped and variant 1 is applied.
    //
    // Hand-derivation:
    //   ref = [1,2,3,4,5], ref_start=0, shift=0, out_len=5
    //   variant 0: pos=1, ilen=0, allele=[55]
    //   variant 1: pos=3, ilen=0, allele=[99]
    //   keep = [false, true]
    //   --- v=0: keep[0]=false → continue (skipped entirely) ---
    //   --- v=1: keep[1]=true → process ---
    //   ref_len = v_pos(3) - ref_idx(0) = 3 → write ref[0..3]=[1,2,3]
    //   allele=[99], writable_length=1 → write 99, out_idx=4
    //   ref_idx = v_ref_end = 3 - min(0,0) + 1 = 4
    //   Final fill: ref_idx=4, unfilled=1, writable_ref=min(1,5-4)=1
    //   out[4] = ref[4] = 5
    //   out = [1,2,3,99,5]
    //   variant 0 (at pos 1, allele 55) NOT applied; variant 1 IS applied at pos 3.
    // -------------------------------------------------------------------------
    #[test]
    fn keep_mask_excludes_variant() {
        let (out, av, _ap) = run(
            &[0, 1],       // v_idxs: variants 0 and 1
            &[1, 3],       // v_starts: variant 0 at pos 1, variant 1 at pos 3
            &[0, 0],       // ilens: both SNPs
            0,             // shift=0
            &[55u8, 99],   // alleles: 55 for v0, 99 for v1
            &[0i64, 1, 2], // alt_offsets
            &[1, 2, 3, 4, 5],
            0,                    // ref_start
            5,                    // out_len
            0,                    // pad_char
            Some(&[false, true]), // keep mask: skip v0, apply v1
            true,                 // annotate
        );
        // variant 0 (pos=1, allele=55) excluded by keep mask: ref[1] NOT replaced
        // variant 1 (pos=3, allele=99) applied: ref[3] replaced by 99
        assert_eq!(out, vec![1, 2, 3, 99, 5]);
        // annot_v_idxs: positions 0..3 are ref (-1), position 3 is variant 1, position 4 is ref (-1)
        assert_eq!(av, vec![-1, -1, -1, 1, -1]);
    }

    // -------------------------------------------------------------------------
    // Case 10: annotated vs non-annotated produce identical out bytes
    // ref = [1,2,3,4,5], ref_start=0, variant at pos=2 (SNP)
    // -------------------------------------------------------------------------
    #[test]
    fn annotated_vs_non_annotated_identical_out() {
        let params = (
            &[0i32][..],            // v_idxs
            &[2i32][..],            // v_starts
            &[0i32][..],            // ilens
            0i64,                   // shift
            &[77u8][..],            // alt_alleles
            &[0i64, 1][..],         // alt_offsets
            &[1u8, 2, 3, 4, 5][..], // ref_
            0i64,                   // ref_start
            5usize,                 // out_len
            0u8,                    // pad_char
        );
        let (out_annot, _, _) = run(
            params.0, params.1, params.2, params.3, params.4, params.5, params.6, params.7,
            params.8, params.9, None, true,
        );
        let (out_plain, _, _) = run(
            params.0, params.1, params.2, params.3, params.4, params.5, params.6, params.7,
            params.8, params.9, None, false,
        );
        assert_eq!(
            out_annot, out_plain,
            "annotated and non-annotated must produce identical out bytes"
        );
    }

    #[test]
    fn batch_correctness_two_queries() {
        // Correctness check for the batch driver: 2 queries × 1 haplotype, no variants.
        // The batch driver is intentionally serial-only: parity is this phase's only gate
        // (throughput is recorded, not gated); the rayon parallel path is deferred to the
        // throughput/fusion optimization pass.  The out/annotation buffers are written by
        // disjoint per-(query,hap) slices, so this loop is rayon-parallelizable later via
        // the same disjoint-chunk split used in src/reference/mod.rs get_reference.
        // Expected: each out chunk is just the corresponding ref slice.
        let reference = b"ACGTACGTACGT";
        let ref_ = arr1(reference.as_ref());
        let ref_offsets = arr1(&[0i64, 12]);
        let v_starts = arr1::<i32>(&[]);
        let ilens = arr1::<i32>(&[]);
        let alt_alleles = arr1::<u8>(&[]);
        let alt_offsets = arr1(&[0i64]);
        // Two regions: [0,4) and [4,8) on contig 0
        let regions = ndarray::arr2(&[[0i32, 0, 4], [0, 4, 8]]);
        let shifts = ndarray::arr2(&[[0i32], [0]]);
        let geno_offset_idx = ndarray::arr2(&[[0i64], [1]]);
        let geno_o_starts = arr1(&[0i64, 0]);
        let geno_o_stops = arr1(&[0i64, 0]);
        let geno_v_idxs = arr1::<i32>(&[]);
        let out_offsets = arr1(&[0i64, 4, 8]);
        let pad_char = b'N';

        let mut out = ndarray::Array1::<u8>::from_elem(8, pad_char);
        super::reconstruct_haplotypes_from_sparse(
            out.view_mut(),
            out_offsets.view(),
            regions.view(),
            shifts.view(),
            geno_offset_idx.view(),
            geno_o_starts.view(),
            geno_o_stops.view(),
            geno_v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt_alleles.view(),
            alt_offsets.view(),
            ref_.view(),
            ref_offsets.view(),
            pad_char,
            None,
            None,
            None,
            None,
            false,
        );

        assert_eq!(&out.as_slice().unwrap()[0..4], b"ACGT", "first region");
        assert_eq!(&out.as_slice().unwrap()[4..8], b"ACGT", "second region");
    }

    #[test]
    fn batch_correctness_with_snp() {
        // Correctness check for the batch driver with a SNP to exercise the
        // variant-application path (not just reference-copy).
        // Reference: "ACGTACGT" (8 bp, contig 0)
        // Two regions: [0,4) and [4,8).
        // One SNP at ref position 1 (C→T), present in haplotype 0 of query 0 only.
        // Expected region 0: "ATGT" (SNP applied), region 1: "ACGT" (no variant).
        let reference = b"ACGTACGT";
        let ref_ = arr1(reference.as_ref());
        let ref_offsets = arr1(&[0i64, 8]);

        // One SNP: position 1, iLen 0 (substitution), alt allele b'T'
        let v_starts = arr1::<i32>(&[1]);
        let ilens = arr1::<i32>(&[0]);
        let alt_alleles = arr1::<u8>(b"T");
        // alt_offsets: [start_of_allele_0, end_of_allele_0] = [0, 1]
        let alt_offsets = arr1(&[0i64, 1]);

        // Two queries, one haplotype each
        let regions = ndarray::arr2(&[[0i32, 0, 4], [0, 4, 8]]);
        let shifts = ndarray::arr2(&[[0i32], [0]]);

        // Query 0, hap 0: has the SNP at variant index 0
        // Query 1, hap 0: no variants
        // geno_offset_idx[query, hap] → index into geno_o_starts/stops
        let geno_offset_idx = ndarray::arr2(&[[0i64], [1]]);
        // For query 0 hap 0: variant block spans geno_v_idxs[0..1] → [0]
        // For query 1 hap 0: empty block (start == stop)
        let geno_o_starts = arr1(&[0i64, 1]);
        let geno_o_stops = arr1(&[1i64, 1]);
        let geno_v_idxs = arr1::<i32>(&[0]); // variant index 0 = the SNP

        let out_offsets = arr1(&[0i64, 4, 8]);
        let pad_char = b'N';

        let mut out = ndarray::Array1::<u8>::from_elem(8, pad_char);
        super::reconstruct_haplotypes_from_sparse(
            out.view_mut(),
            out_offsets.view(),
            regions.view(),
            shifts.view(),
            geno_offset_idx.view(),
            geno_o_starts.view(),
            geno_o_stops.view(),
            geno_v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt_alleles.view(),
            alt_offsets.view(),
            ref_.view(),
            ref_offsets.view(),
            pad_char,
            None,
            None,
            None,
            None,
            false,
        );

        assert_eq!(
            &out.as_slice().unwrap()[0..4],
            b"ATGT",
            "region 0 with SNP applied"
        );
        assert_eq!(
            &out.as_slice().unwrap()[4..8],
            b"ACGT",
            "region 1 reference-only"
        );
    }

    // -------------------------------------------------------------------------
    // SVAR2 driver: one region (R=1), one sample-slot (S=1), one ploid (P=1) →
    // n_work = 1, k = query = hap = 0. Hand-built two-channel input: a SNP in the
    // var_key channel and a pure DEL in the dense channel (with its presence bit set),
    // exercising both the merge and the decode paths in a single call.
    //
    // ref = "ACGTACGT" (8 bp, positions 0..8: A C G T A C G T)
    // var_key: SNP at pos=1 (ref 'C' -> 'T'), Inline key, decode_alt -> (v_diff=0, "T")
    // dense:   pure DEL at pos=4, ilen=-1, decode_alt -> (v_diff=-1, "" empty). The driver's
    //          `provide` closure substitutes the anchor base ref[pos]='A' for the empty ALT
    //          (genoray stores no anchor in the key; the kernel needs it, else the anchor is
    //          deleted too). So the DEL contributes allele="A" and deletes the following 'C'.
    // merge_hap orders by position: v=0 is the SNP (pos=1), v=1 is the DEL (pos=4).
    //
    // Hand derivation of reconstruct_haplotype_core(n_variants=2, ..., shift=0):
    //   start: ref_idx=0, out_idx=0 (ref_start=0, not negative: no leading pad)
    //   v=0 (SNP, pos=1, v_diff=0, allele="T", v_len_full=1):
    //     v_ref_end = 1 - min(0,0) + 1 = 2
    //     not a DEL-spanning-start case; pos(1) >= ref_idx(0); shift=0 so no shift branch
    //     ref_len = 1-0 = 1 -> out[0..1] = ref[0..1] = "A"; out_idx=1
    //     writable_length = min(1, 8-1)=1 -> out[1..2] = "T"; out_idx=2
    //     ref_idx = v_ref_end = 2
    //   v=1 (DEL, pos=4, v_diff=-1, allele="A" (anchor = ref[4])):
    //     v_ref_end = 4 - min(0,-1) + 1 = 4+1+1 = 6
    //     pos(4) >= ref_idx(2); shift=0 so no shift branch
    //     ref_len = 4-2 = 2 -> out[2..4] = ref[2..4] = "GT"; out_idx=4
    //     writable_length = min(1, 8-4) = 1 -> out[4..5] = "A"; out_idx=5
    //     ref_idx = v_ref_end = 6  (ref 'C' at pos 5 is skipped = deleted)
    //   tail: unfilled_length = 8-5 = 3; writable_ref = min(3, 8-6) = 2
    //     -> out[5..7] = ref[6..8] = "GT"; out_end_idx=7
    //     right-pad out[7..8] with pad_char 'N'
    //   out = "A" + "T" + "GT" + "A" + "GT" + "N" = "ATGTAGTN"
    //   (haplotype = ref with SNP C->T at pos1 and the 'C' at pos5 deleted)
    #[test]
    fn svar2_reconstruct_snp_and_del() {
        let reference = b"ACGTACGT";
        let ref_ = arr1(reference.as_ref());
        let ref_offsets = arr1(&[0i64, 8]);

        // One region on contig 0, one sample-slot, one ploid -> n_work = 1.
        let regions = ndarray::arr2(&[[0i32, 0, 8]]);
        let shifts = ndarray::arr2(&[[0i32]]);

        // var_key channel: one SNP at pos=1 (Inline key, 1-base ALT "T").
        let snp_key = svar2_codec::encode_alt_inline(b"T", 0) as i32;
        let vk_pos = arr1(&[1i32]);
        let vk_key = arr1(&[snp_key]);
        let vk_off = arr1(&[0i64, 1]);

        // dense channel: one pure DEL (1bp) at pos=4, present for this hap.
        let del_key = svar2_codec::encode_pure_del(-1) as i32;
        let dense_pos = arr1(&[4i32]);
        let dense_key = arr1(&[del_key]);
        let dense_range = ndarray::arr2(&[[0i32, 1]]);
        // Presence bits, LSB-first: bit 0 (this hap's only dense entry) is set.
        let dense_present = arr1(&[0b0000_0001u8]);
        let dense_present_off = arr1(&[0i64, 1]);

        // No long-allele-bank lookups exercised in this test.
        let lut_bytes = arr1::<u8>(&[]);
        let lut_off = arr1(&[0i64]);

        let out_offsets = arr1(&[0i64, 8]);
        let pad_char = b'N';
        let mut out = Array1::<u8>::from_elem(8, pad_char);

        super::reconstruct_haplotypes_from_svar2(
            out.view_mut(),
            out_offsets.view(),
            regions.view(),
            shifts.view(),
            vk_pos.view(),
            vk_key.view(),
            vk_off.view(),
            dense_pos.view(),
            dense_key.view(),
            dense_range.view(),
            dense_present.view(),
            dense_present_off.view(),
            lut_bytes.view(),
            lut_off.view(),
            ref_.view(),
            ref_offsets.view(),
            pad_char,
            false, // serial
        );

        assert_eq!(out.as_slice().unwrap(), b"ATGTAGTN");
    }

    // Regression guard for the pure-DEL anchor fix: a lone 1bp DEL must delete exactly
    // one base (the one AFTER the anchor), not two. ref="ACGT", DEL at pos=1 (ilen=-1):
    // keep anchor 'C' at pos1, delete 'G' at pos2 -> "ACT" + right-pad 'N' = "ACTN".
    #[test]
    fn svar2_pure_del_keeps_anchor() {
        let reference = b"ACGT";
        let ref_ = arr1(reference.as_ref());
        let ref_offsets = arr1(&[0i64, 4]);
        let regions = ndarray::arr2(&[[0i32, 0, 4]]);
        let shifts = ndarray::arr2(&[[0i32]]);

        // No var_key entries; a single dense pure DEL at pos=1.
        let vk_pos = arr1::<i32>(&[]);
        let vk_key = arr1::<i32>(&[]);
        let vk_off = arr1(&[0i64, 0]);
        let del_key = svar2_codec::encode_pure_del(-1) as i32;
        let dense_pos = arr1(&[1i32]);
        let dense_key = arr1(&[del_key]);
        let dense_range = ndarray::arr2(&[[0i32, 1]]);
        let dense_present = arr1(&[0b0000_0001u8]);
        let dense_present_off = arr1(&[0i64, 1]);
        let lut_bytes = arr1::<u8>(&[]);
        let lut_off = arr1(&[0i64]);
        let out_offsets = arr1(&[0i64, 4]);
        let pad_char = b'N';
        let mut out = Array1::<u8>::from_elem(4, pad_char);

        super::reconstruct_haplotypes_from_svar2(
            out.view_mut(),
            out_offsets.view(),
            regions.view(),
            shifts.view(),
            vk_pos.view(),
            vk_key.view(),
            vk_off.view(),
            dense_pos.view(),
            dense_key.view(),
            dense_range.view(),
            dense_present.view(),
            dense_present_off.view(),
            lut_bytes.view(),
            lut_off.view(),
            ref_.view(),
            ref_offsets.view(),
            pad_char,
            false,
        );

        assert_eq!(out.as_slice().unwrap(), b"ACTN");
    }
}
