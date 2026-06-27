//! Track-realignment PRNG primitives and insertion-fill strategies.
//!
//! PRNG functions mirror the numba implementations in
//! `python/genvarloader/_dataset/_tracks.py` (`_xorshift64`, `_hash4`) exactly.
//! All arithmetic is on `u64` with wrapping shifts/xors to match numba's
//! `np.uint64` overflow semantics.
//!
//! `apply_insertion_fill` mirrors `_apply_insertion_fill` in the same file
//! (lines 56-138), statement-by-statement, including float promotion points.

use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1};
use rayon::prelude::*;

// Strategy IDs — mirror _insertion_fill.py exactly.
pub const REPEAT_5P: i64 = 0;
pub const REPEAT_5P_NORM: i64 = 1;
pub const CONSTANT: i64 = 2;
pub const FLANK_SAMPLE: i64 = 3;
pub const INTERPOLATE: i64 = 4;

/// Single round of xorshift64.
///
/// Mirrors numba `_xorshift64` on `np.uint64`:
/// ```text
/// x ^= x << 13
/// x ^= x >> 7
/// x ^= x << 17
/// ```
/// Left shifts use `wrapping_shl` to replicate `np.uint64` truncation-to-64-bits.
#[inline(always)]
pub fn xorshift64(mut x: u64) -> u64 {
    x ^= x.wrapping_shl(13);
    x ^= x >> 7;
    x ^= x.wrapping_shl(17);
    x
}

/// Hash four `u64` values into one.
///
/// Mirrors numba `_hash4`:
/// ```text
/// h = a
/// h = xorshift64(h ^ b)
/// h = xorshift64(h ^ c)
/// h = xorshift64(h ^ d)
/// ```
#[inline(always)]
pub fn hash4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut h = a;
    h = xorshift64(h ^ b);
    h = xorshift64(h ^ c);
    h = xorshift64(h ^ d);
    h
}

/// Fill `writable_length` values starting at `out[out_idx]` using the given
/// insertion-fill strategy.
///
/// Mirrors numba `_apply_insertion_fill` (lines 56-138 of `_tracks.py`)
/// statement-by-statement, including float promotion points:
///
/// - `REPEAT_5P_NORM`: numba computes `track[v_rel_pos] / v_len` in **f64**
///   (`v_len` is int64; np.float32 / np.int64 → float64), then rounds to f32
///   on store. We compute f32 / f32 directly: this is bit-identical to numba
///   **only** because IEEE-754 division is double-rounding-safe (f64 mantissa
///   53 bits ≥ 2·24+2 = 50, verified empirically over 42M cases). Do NOT
///   generalize this f32-direct shortcut to multiply-add or multi-step
///   accumulations — those are NOT double-rounding-safe; mirror numba's f64
///   intermediate there.
/// - `CONSTANT`: `params[0]` is f64; stored into f32 `out` (cast on store).
/// - `INTERPOLATE`: all anchor/Lagrange arithmetic in f64 (`xs`, `ys` are f64);
///   `ys[j] = track[ref_idx]` promotes f32 → f64 on assignment; final `acc`
///   stored into f32 `out` (cast on store).
///
/// # Parameters
/// - `out`: output track buffer (f32)
/// - `out_idx`: starting write index within `out`
/// - `writable_length`: number of positions to write
/// - `v_len`: total insertion length (v_diff + 1)
/// - `track`: reference track values (f32)
/// - `v_rel_pos`: variant position relative to the query region
/// - `strategy_id`: one of `REPEAT_5P`, `REPEAT_5P_NORM`, `CONSTANT`,
///   `FLANK_SAMPLE`, `INTERPOLATE`
/// - `params`: per-strategy parameter slot (f64); `params[0]` = flank_width,
///   constant value, or interpolation order depending on strategy
/// - `base_seed`, `query`, `hap`: seed components for `FLANK_SAMPLE`
pub fn apply_insertion_fill(
    out: &mut ArrayViewMut1<f32>,
    out_idx: usize,
    writable_length: usize,
    v_len: i64,
    track: ArrayView1<f32>,
    v_rel_pos: i64,
    strategy_id: i64,
    params: ArrayView1<f64>,
    base_seed: u64,
    query: u64,
    hap: u64,
) {
    let track_len = track.len() as i64;

    if strategy_id == REPEAT_5P {
        // Numba comment: "unreachable from outer kernel (which short-circuits this
        // strategy before calling). Kept for completeness and direct-helper-call safety."
        let val = track[v_rel_pos as usize];
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == REPEAT_5P_NORM {
        // Numba: val = track[v_rel_pos] / v_len  (computed in f64; v_len is int64,
        // so np.float32/np.int64 → float64), then stored into f32 out.
        // We divide f32/f32 directly: bit-identical to numba because IEEE-754
        // division is double-rounding-safe. Do NOT extend this shortcut to
        // multiply-add or multi-op paths — use f64 intermediates there.
        let val = track[v_rel_pos as usize] / (v_len as f32);
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == CONSTANT {
        // Numba: val = params[0] (f64), stored into f32 out on assignment.
        let val = params[0] as f32;
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == FLANK_SAMPLE {
        // Numba: width = np.int64(params[0])
        let width = params[0] as i64;
        let pool_lo = (v_rel_pos - width).max(0);
        let pool_hi = (v_rel_pos + width).min(track_len - 1);
        let pool_size = (pool_hi - pool_lo + 1) as u64;
        for i in 0..writable_length {
            // Numba: seed = _hash4(base_seed, np.uint64(query), np.uint64(hap), np.uint64(out_idx + i))
            let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
            // Numba: offset = np.int64(seed % np.uint64(pool_size))
            let offset = (seed % pool_size) as i64;
            out[out_idx + i] = track[(pool_lo + offset) as usize];
        }
    } else if strategy_id == INTERPOLATE {
        // Numba: order = np.int64(params[0])
        let order = params[0] as i64;
        // k = ceil((order+1)/2)
        // Numba: k = (order + 1 + 1) // 2
        let k = (order + 1 + 1) / 2;
        let n_anchors = (2 * k) as usize;

        // Anchors: xs and ys are f64 (numba: np.empty(..., dtype=np.float64))
        let mut xs = vec![0.0f64; n_anchors];
        let mut ys = vec![0.0f64; n_anchors];

        // 5' side: xs[j] = -j, ys[j] = track[max(v_rel_pos - j, 0)]
        // Numba: xs[j] = -float(j), ys[j] = track[ref_idx]
        // track[ref_idx] is f32; ys is f64 → f32 promoted to f64 on assignment.
        for j in 0..k as usize {
            let ref_idx = (v_rel_pos - j as i64).max(0) as usize;
            xs[j] = -(j as f64);
            ys[j] = track[ref_idx] as f64;
        }
        // 3' side: xs[k+j] = v_len + j, ys[k+j] = track[min(v_rel_pos+1+j, track_len-1)]
        // Numba: xs[k + j] = float(v_len) + float(j), ys[k + j] = track[ref_idx]
        for j in 0..k as usize {
            let ref_idx = (v_rel_pos + 1 + j as i64).min(track_len - 1) as usize;
            xs[k as usize + j] = (v_len as f64) + (j as f64);
            ys[k as usize + j] = track[ref_idx] as f64;
        }

        // Lagrange interpolation: mirror numba loop nesting exactly.
        // outer: a over n_anchors; inner: b over n_anchors, skip b==a
        for i in 0..writable_length {
            // Numba: x = float(i) — this is the insertion-local coordinate
            let x = i as f64;
            // Numba: acc = 0.0 (float64 literal)
            let mut acc = 0.0f64;
            for a in 0..n_anchors {
                // Numba: term = ys[a]
                let mut term = ys[a];
                for b in 0..n_anchors {
                    if b == a {
                        continue;
                    }
                    // Numba: term *= (x - xs[b]) / (xs[a] - xs[b])
                    term *= (x - xs[b]) / (xs[a] - xs[b]);
                }
                // Numba: acc += term
                acc += term;
            }
            // Numba: out[out_idx + i] = acc — f64 acc stored into f32 out
            out[out_idx + i] = acc as f32;
        }
    }
}

/// Shift and realign a single track to correspond to one haplotype.
///
/// Mirrors numba `shift_and_realign_track_sparse` (lines 230-401 of `_tracks.py`)
/// statement-by-statement.
///
/// Three key differences from the haplotype reconstruction kernel:
/// 1. SNPs (`v_diff == 0`) are SKIPPED — tracks match reference at SNP positions.
/// 2. Insertions route to `apply_insertion_fill` UNLESS `strategy_id == REPEAT_5P`
///    (which repeats `track[v_rel_pos]` directly).
/// 3. Trailing fill pads with `0.0` (NOT a pad_char byte).
///
/// # Parameters
/// - `offset_idx`: index into geno_o_starts/geno_o_stops for this (query, hap) pair
/// - `geno_v_idxs`: flat variant index array
/// - `geno_o_starts`, `geno_o_stops`: normalized (2, n) offsets split into two rows
/// - `v_starts`: variant start positions (absolute genomic coordinates)
/// - `ilens`: variant insertion-length differences (signed)
/// - `shift`: total shift for this haplotype
/// - `track`: reference track values for this query (f32 slice)
/// - `query_start`: the genomic start of this query region
/// - `out`: output slice to fill (length = haplotype output length)
/// - `params`: per-strategy parameter (f64)
/// - `keep`: optional boolean mask over the variant group for this (query, hap)
/// - `strategy_id`: insertion-fill strategy
/// - `base_seed`, `query`, `hap`: seed components for FlankSample strategy
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_track_sparse(
    offset_idx: usize,
    geno_v_idxs: ndarray::ArrayView1<i32>,
    geno_o_starts: ndarray::ArrayView1<i64>,
    geno_o_stops: ndarray::ArrayView1<i64>,
    v_starts: ndarray::ArrayView1<i32>,
    ilens: ndarray::ArrayView1<i32>,
    shift: i64,
    track: ndarray::ArrayView1<f32>,
    query_start: i64,
    out: &mut ndarray::ArrayViewMut1<f32>,
    params: ndarray::ArrayView1<f64>,
    keep: Option<ndarray::ArrayView1<bool>>,
    strategy_id: i64,
    base_seed: u64,
    query: u64,
    hap: u64,
) {
    // Numba: o_s, o_e = geno_offsets[offset_idx], geno_offsets[offset_idx + 1]  (1-D branch)
    //        or geno_offsets[:, offset_idx]  (2-D branch — normalized form)
    // We receive the pre-split (2, n) rows directly.
    let o_s = geno_o_starts[offset_idx] as usize;
    let o_e = geno_o_stops[offset_idx] as usize;
    let variant_idxs = &geno_v_idxs.as_slice().unwrap()[o_s..o_e];
    let length = out.len();
    let n_variants = variant_idxs.len();

    if n_variants == 0 {
        // Numba: out[:] = track[:length]
        for i in 0..length {
            out[i] = track[i];
        }
        return;
    }

    // Numba: track_idx = 0; out_idx = 0; shifted = 0
    let mut track_idx: i64 = 0;
    let mut out_idx: i64 = 0;
    let mut shifted: i64 = 0;

    for v in 0..n_variants {
        // Numba: if keep is not None and not keep[v]: continue
        if let Some(ref k) = keep {
            if !k[v] {
                continue;
            }
        }

        let variant = variant_idxs[v] as usize;

        // Numba: v_rel_pos = v_starts[variant] - query_start
        let v_rel_pos = v_starts[variant] as i64 - query_start;
        // Numba: v_diff = ilens[variant]
        let v_diff = ilens[variant] as i64;
        // Numba: v_rel_end = v_rel_pos - min(0, v_diff) + 1
        let v_rel_end = v_rel_pos - v_diff.min(0) + 1;

        // Numba: if v_diff < 0 and v_rel_pos < 0 and v_rel_end >= 0:
        //            track_idx = v_rel_end; continue
        if v_diff < 0 && v_rel_pos < 0 && v_rel_end >= 0 {
            track_idx = v_rel_end;
            continue;
        }

        // Numba: if v_rel_pos < track_idx: continue  (overlapping variant)
        if v_rel_pos < track_idx {
            continue;
        }

        // Numba: v_len = max(0, v_diff) + 1
        let mut v_len = v_diff.max(0) + 1;

        // Numba: if shifted < shift:
        if shifted < shift {
            let ref_shift_dist = v_rel_pos - track_idx;
            // Numba: if shifted + ref_shift_dist + v_len < shift: continue
            if shifted + ref_shift_dist + v_len < shift {
                continue;
            } else if shifted + ref_shift_dist >= shift {
                // Numba: track_idx += shift - shifted; shifted = shift
                track_idx += shift - shifted;
                shifted = shift;
            } else {
                // ref + (some of) variant is enough to finish shift
                // Numba: allele_start_idx = shift - shifted - ref_shift_dist; shifted = shift
                let allele_start_idx = shift - shifted - ref_shift_dist;
                shifted = shift;
                // Numba: if allele_start_idx == v_len: track_idx = v_rel_end; continue
                if allele_start_idx == v_len {
                    track_idx = v_rel_end;
                    continue;
                }
                // Numba: track_idx = v_rel_pos; v_len -= allele_start_idx
                track_idx = v_rel_pos;
                v_len -= allele_start_idx;
            }
        }

        // Key difference 1: SNPs skipped for tracks (they match ref)
        // Numba: if v_diff == 0: continue
        if v_diff == 0 {
            continue;
        }

        // Numba: track_len = v_rel_pos - track_idx
        let track_len = v_rel_pos - track_idx;
        // Numba: if out_idx + track_len >= length: break
        if out_idx + track_len >= length as i64 {
            break;
        }
        // Numba: out[out_idx:out_idx+track_len] = track[track_idx:track_idx+track_len]
        for i in 0..track_len as usize {
            out[out_idx as usize + i] = track[track_idx as usize + i];
        }
        out_idx += track_len;

        // Numba: writable_length = min(v_len, length - out_idx)
        let writable_length = (v_len.min(length as i64 - out_idx)) as usize;

        // Key difference 2: insertions route to apply_insertion_fill unless REPEAT_5P
        // Numba: if v_diff > 0 and strategy_id != _REPEAT_5P:
        if v_diff > 0 && strategy_id != REPEAT_5P {
            apply_insertion_fill(
                out,
                out_idx as usize,
                writable_length,
                v_len,
                track,
                v_rel_pos,
                strategy_id,
                params,
                base_seed,
                query,
                hap,
            );
        } else {
            // Numba: for i in range(writable_length): out[out_idx + i] = track[v_rel_pos]
            // Deletions AND Repeat5p insertions: repeat track[v_rel_pos]
            let val = track[v_rel_pos as usize];
            for i in 0..writable_length {
                out[out_idx as usize + i] = val;
            }
        }
        out_idx += writable_length as i64;
        track_idx = v_rel_end;

        // Numba: if out_idx >= length: break
        if out_idx >= length as i64 {
            break;
        }
    }

    // Numba: if shifted < shift: track_idx += shift - shifted; ...
    if shifted < shift {
        track_idx += shift - shifted;
        track_idx = track_idx.min(track.len() as i64);
        // shifted = shift;  (not used after this point)
    }

    // Key difference 3: trailing fill pads with 0.0 (NOT pad_char)
    // Numba: unfilled_length = length - out_idx
    let unfilled_length = length as i64 - out_idx;
    if unfilled_length > 0 {
        // When a deletion's v_rel_end runs past the track end, track_idx advances
        // past track.len() and writable_ref becomes negative. The fixed numba kernel
        // uses max(0, min(unfilled, len(track)-track_idx)), so writable_ref >= 0 and
        // out_end_idx = out_idx. Mirror that: clamp out_end_idx to out_idx so the
        // zero-pad fills exactly out[out_idx..length] without overwriting
        // already-written positions (mirrors reconstruct/mod.rs:234-239).
        let writable_ref = unfilled_length.min(track.len() as i64 - track_idx);
        // Positive: copy track bytes. Zero or negative: track exhausted, no copy.
        let out_end_idx = if writable_ref > 0 {
            let oe = out_idx + writable_ref;
            let re = track_idx + writable_ref;
            // Numba: out[out_idx:out_end_idx] = track[track_idx:ref_end_idx]
            for i in 0..writable_ref as usize {
                out[out_idx as usize + i] = track[track_idx as usize + i];
            }
            let _ = re; // ref_end_idx used only to bound the copy above
            oe
        } else {
            // writable_ref <= 0: track exhausted (track_idx at/after track end).
            // No track bytes remain to copy; zero-pad the entire unfilled tail
            // out[out_idx..length]. Clamp to out_idx (NOT (out_idx+writable_ref).max(0))
            // to avoid overwriting already-written positions.
            out_idx
        };
        // Numba: if out_end_idx < length: out[out_end_idx:] = 0
        if out_end_idx < length as i64 {
            for i in out_end_idx as usize..length {
                out[i] = 0.0_f32;
            }
        }
    }
}

/// Shift and realign tracks for a batch of (query, hap) pairs in place (writes `out`).
///
/// Mirrors numba `shift_and_realign_tracks_sparse` (lines 141-228 of `_tracks.py`)
/// statement-by-statement. Serial-only (rayon deferred to Phase 5, matching Task 5
/// precedent for initial parity verification).
///
/// # Parameters
/// - `out`: flat output buffer (f32), written in place
/// - `out_offsets`: ragged offsets into out, shape (n_q * ploidy + 1,)
/// - `regions`: (n_q, 3) array of (contig_idx, start, end) per query
/// - `shifts`: (n_q, ploidy) shift per (query, hap)
/// - `geno_offset_idx`: (n_q, ploidy) indices into geno_o_starts/stops
/// - `geno_v_idxs`: flat variant index array
/// - `geno_o_starts`, `geno_o_stops`: normalized (2, n) offsets split into rows
/// - `v_starts`: variant start positions
/// - `ilens`: variant ilen differences
/// - `tracks`: flat reference track buffer (f32), ragged by track_offsets
/// - `track_offsets`: (n_q + 1,) offsets into tracks (one track per query)
/// - `params`: per-strategy parameter (f64), shape (1,)
/// - `keep`, `keep_offsets`: optional keep mask + 1-D offsets
/// - `strategy_id`, `base_seed`: insertion-fill strategy parameters
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_tracks_sparse(
    mut out: ndarray::ArrayViewMut1<f32>,
    out_offsets: ndarray::ArrayView1<i64>,
    regions: ndarray::ArrayView2<i32>,
    shifts: ndarray::ArrayView2<i32>,
    geno_offset_idx: ndarray::ArrayView2<i64>,
    geno_v_idxs: ndarray::ArrayView1<i32>,
    geno_o_starts: ndarray::ArrayView1<i64>,
    geno_o_stops: ndarray::ArrayView1<i64>,
    v_starts: ndarray::ArrayView1<i32>,
    ilens: ndarray::ArrayView1<i32>,
    tracks: ndarray::ArrayView1<f32>,
    track_offsets: ndarray::ArrayView1<i64>,
    params: ndarray::ArrayView1<f64>,
    keep: Option<ndarray::ArrayView1<bool>>,
    keep_offsets: Option<ndarray::ArrayView1<i64>>,
    strategy_id: i64,
    base_seed: u64,
    parallel: bool,
) {
    // Numba: n_regions, ploidy = geno_offset_idx.shape
    let n_regions = geno_offset_idx.nrows();
    let ploidy = geno_offset_idx.ncols();
    let n_work = n_regions * ploidy;

    // Hoist contiguous raw slices once to eliminate ndarray::do_slice call overhead
    // in the inner (query, hap) loop.  The prior interval-kernel fix (src/intervals.rs)
    // applied the same pattern: out.as_slice_mut().unwrap() once, then index [a..b]
    // directly.  Here we do the same for out, tracks, and keep.
    // geno_v_idxs already uses .as_slice().unwrap() (inner fn line 240) — same contract.
    let out_flat = out.as_slice_mut().expect("out must be contiguous (C-order)");
    let tracks_flat = tracks.as_slice().expect("tracks must be contiguous (C-order)");
    // Hoist keep flat option once (avoids repeated .as_slice() per hap).
    let keep_flat: Option<&[bool]> =
        keep.as_ref().map(|k| k.as_slice().expect("keep must be contiguous (C-order)"));

    if parallel {
        // Build disjoint per-k mutable output slices using the split_at_mut cursor
        // idiom (mirrors C1 reconstruct_haplotypes_from_sparse parallel path).
        let bounds: Vec<(usize, usize)> = (0..n_work)
            .map(|k| (out_offsets[k] as usize, out_offsets[k + 1] as usize))
            .collect();

        let mut out_chunks: Vec<&mut [f32]> = Vec::with_capacity(n_work);
        {
            let mut rest = &mut out_flat[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
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

        out_chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(k, out_chunk)| {
                let query = k / ploidy;
                let hap = k % ploidy;

                let t_s = track_offsets[query] as usize;
                let t_e = track_offsets[query + 1] as usize;
                let q_track = ndarray::ArrayView1::from(&tracks_flat[t_s..t_e]);
                let q_start = regions[[query, 1]] as i64;
                let o_idx = geno_offset_idx[[query, hap]] as usize;
                let qh_shift = shifts[[query, hap]] as i64;

                let qh_keep: Option<ndarray::ArrayView1<bool>> =
                    match (&keep_flat, &keep_offsets) {
                        (Some(k_flat), Some(ko)) => {
                            let ks = ko[k] as usize;
                            let ke = ko[k + 1] as usize;
                            Some(ndarray::ArrayView1::from(&k_flat[ks..ke]))
                        }
                        _ => None,
                    };

                let mut qh_out = ndarray::ArrayViewMut1::from(out_chunk);
                shift_and_realign_track_sparse(
                    o_idx,
                    geno_v_idxs,
                    geno_o_starts,
                    geno_o_stops,
                    v_starts,
                    ilens,
                    qh_shift,
                    q_track,
                    q_start,
                    &mut qh_out,
                    params,
                    qh_keep,
                    strategy_id,
                    base_seed,
                    query as u64,
                    hap as u64,
                );
            });
    } else {
        // Serial path: Numba: for query in nb.prange(n_regions):  (serial equivalent)
        for query in 0..n_regions {
            // Numba: t_s, t_e = track_offsets[query], track_offsets[query + 1]
            let t_s = track_offsets[query] as usize;
            let t_e = track_offsets[query + 1] as usize;
            // Numba: q_track = tracks[t_s:t_e]
            // ArrayView1::from(&slice) is cheaper than tracks.slice(s![..]) — no do_slice call.
            let q_track = ndarray::ArrayView1::from(&tracks_flat[t_s..t_e]);

            // Numba: q_start = regions[query, 1]
            let q_start = regions[[query, 1]] as i64;

            // Numba: for hap in nb.prange(ploidy):  (serial equivalent)
            for hap in 0..ploidy {
                // Numba: o_idx = geno_offset_idx[query, hap]
                let o_idx = geno_offset_idx[[query, hap]] as usize;

                // Numba: k_idx = query * ploidy + hap
                let k_idx = query * ploidy + hap;

                // Numba: if keep is not None and keep_offsets is not None:
                //            qh_keep = keep[keep_offsets[k_idx]:keep_offsets[k_idx+1]]
                // ArrayView1::from(&slice[..]) avoids the do_slice call that
                // k.slice(s![ks..ke]) would generate.
                let qh_keep: Option<ndarray::ArrayView1<bool>> =
                    match (&keep_flat, &keep_offsets) {
                        (Some(k_flat), Some(ko)) => {
                            let ks = ko[k_idx] as usize;
                            let ke = ko[k_idx + 1] as usize;
                            Some(ndarray::ArrayView1::from(&k_flat[ks..ke]))
                        }
                        _ => None,
                    };

                // Numba: out_s, out_e = out_offsets[k_idx], out_offsets[k_idx + 1]
                let out_s = out_offsets[k_idx] as usize;
                let out_e = out_offsets[k_idx + 1] as usize;
                // Numba: qh_out = out[out_s:out_e]; qh_shifts = shifts[query, hap]
                // ArrayViewMut1::from(&mut slice[..]) avoids the do_slice call that
                // out.slice_mut(s![out_s..out_e]) would generate.
                let mut qh_out = ndarray::ArrayViewMut1::from(&mut out_flat[out_s..out_e]);
                let qh_shift = shifts[[query, hap]] as i64;

                shift_and_realign_track_sparse(
                    o_idx,
                    geno_v_idxs,
                    geno_o_starts,
                    geno_o_stops,
                    v_starts,
                    ilens,
                    qh_shift,
                    q_track,
                    q_start,
                    &mut qh_out,
                    params,
                    qh_keep,
                    strategy_id,
                    base_seed,
                    query as u64,
                    hap as u64,
                );
            }
        }
    }
}

/// RLE-encode a ragged f32 track buffer into (starts, ends, values, offsets) intervals.
///
/// Mirrors numba `tracks_to_intervals` + `_scanned_mask` + `_compact_mask` in
/// `python/genvarloader/_dataset/_intervals.py` lines 129-220, statement-by-statement.
///
/// # Algorithm (matches numba exactly)
/// Two-pass:
/// 1. For each query, compute `scanned_mask` (cumulative count of value-change positions)
///    and store `n_intervals[query] = scanned_mask[-1]`.
/// 2. Cumsum `n_intervals` into `interval_offsets` (i64, mirrors numba's `.cumsum()`).
/// 3. Fill pass: for each query, recover run boundaries via `compact_mask`, then write
///    starts/ends/values into the output arrays at `interval_offsets[query]`.
///
/// Key fidelity points:
/// - `backward_mask[0] = true`, `backward_mask[i] = track[i-1] != track[i]` — exact f32 `!=`
///   (bit-level, not ordered comparison).
/// - `scanned_mask` = prefix-sum of `backward_mask` (i64 accumulation).
/// - 0-value intervals ARE included (no filtering on value == 0.0, matches numba comment).
/// - `starts` and `ends` are absolute genomic coords: `boundaries + regions[query, 1]`.
/// - Output dtypes: starts/ends i32, values f32, offsets i64.
pub fn tracks_to_intervals(
    regions: ArrayView2<i32>,
    tracks: ArrayView1<f32>,
    track_offsets: ArrayView1<i64>,
    parallel: bool,
) -> (Array1<i32>, Array1<i32>, Array1<f32>, Array1<i64>) {
    let n_queries = regions.nrows();

    // --- Pass 1: count intervals per query ---
    // Numba: n_intervals = np.empty(n_queries, np.int32)
    // Numba: scanned_masks = np.empty_like(tracks, np.int64)
    // We allocate a single flat scanned_masks buffer mirroring numba's layout.
    let total_track_len = tracks.len();
    let mut scanned_masks = vec![0i64; total_track_len];
    let mut n_intervals = vec![0i32; n_queries];

    if parallel {
        // Build disjoint per-query mutable slices of scanned_masks (variable-size
        // chunks per query) using the split_at_mut cursor idiom (mirrors C1).
        let track_bounds: Vec<(usize, usize)> = (0..n_queries)
            .map(|q| (track_offsets[q] as usize, track_offsets[q + 1] as usize))
            .collect();

        let mut scan_chunks: Vec<&mut [i64]> = Vec::with_capacity(n_queries);
        {
            let mut rest = &mut scanned_masks[..];
            let mut cursor = 0usize;
            for &(s, e) in &track_bounds {
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                scan_chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
        }

        let tracks_slice = tracks.as_slice().unwrap();
        scan_chunks
            .into_par_iter()
            .zip(n_intervals.par_iter_mut())
            .enumerate()
            .for_each(|(query, (scan, n_int))| {
                let o_s = track_offsets[query] as usize;
                let o_e = track_offsets[query + 1] as usize;
                if o_s == o_e {
                    *n_int = 0;
                    return;
                }
                let track = &tracks_slice[o_s..o_e];
                let mut acc: i64 = 0;
                for i in 0..track.len() {
                    let bm = if i == 0 {
                        true
                    } else {
                        track[i - 1] != track[i]
                    };
                    acc += bm as i64;
                    scan[i] = acc;
                }
                *n_int = scan[track.len() - 1] as i32;
            });
    } else {
        for query in 0..n_queries {
            let o_s = track_offsets[query] as usize;
            let o_e = track_offsets[query + 1] as usize;
            // Numba: if o_s == o_e: n_intervals[query] = 0; continue
            if o_s == o_e {
                n_intervals[query] = 0;
                continue;
            }
            let track = &tracks.as_slice().unwrap()[o_s..o_e];
            let scan = &mut scanned_masks[o_s..o_e];
            // _scanned_mask: backward_mask[0]=True, backward_mask[i] = track[i-1] != track[i]
            // cumsum into scan (i64 accumulator)
            // Numba: out[:] = backward_mask.cumsum()
            let mut acc: i64 = 0;
            for i in 0..track.len() {
                let bm = if i == 0 {
                    true
                } else {
                    // Exact f32 != comparison (bit-level, matches numba)
                    track[i - 1] != track[i]
                };
                acc += bm as i64;
                scan[i] = acc;
            }
            // n_intervals[query] = scanned_backward_mask[-1]
            n_intervals[query] = scan[track.len() - 1] as i32;
        }
    }

    // --- Two-pass cumsum: mirrors numba's n_intervals.cumsum() ---
    // Numba:
    //   interval_offsets = np.empty(n_queries + 1, np.int64)
    //   interval_offsets[0] = 0
    //   interval_offsets[1:] = n_intervals.cumsum()
    // (stays sequential — prefix-sum has a data dependency chain)
    let mut interval_offsets = vec![0i64; n_queries + 1];
    let mut running: i64 = 0;
    for q in 0..n_queries {
        running += n_intervals[q] as i64;
        interval_offsets[q + 1] = running;
    }
    let total_intervals = running as usize;

    let mut all_starts = vec![0i32; total_intervals];
    let mut all_ends = vec![0i32; total_intervals];
    let mut all_values = vec![0.0f32; total_intervals];

    // --- Pass 2: fill starts/ends/values ---
    if parallel {
        // Build disjoint per-query mutable slices from all_starts/ends/values using
        // interval_offsets (which have already been computed sequentially above).
        let itv_bounds: Vec<(usize, usize)> = (0..n_queries)
            .map(|q| (interval_offsets[q] as usize, interval_offsets[q + 1] as usize))
            .collect();

        let mut starts_chunks: Vec<&mut [i32]> = Vec::with_capacity(n_queries);
        let mut ends_chunks: Vec<&mut [i32]> = Vec::with_capacity(n_queries);
        let mut values_chunks: Vec<&mut [f32]> = Vec::with_capacity(n_queries);

        {
            let mut rest_s = &mut all_starts[..];
            let mut rest_e = &mut all_ends[..];
            let mut rest_v = &mut all_values[..];
            let mut cursor = 0usize;
            for &(s, e) in &itv_bounds {
                let (_, tail_s) = rest_s.split_at_mut(s - cursor);
                let (mid_s, tail_s2) = tail_s.split_at_mut(e - s);
                starts_chunks.push(mid_s);
                rest_s = tail_s2;

                let (_, tail_e) = rest_e.split_at_mut(s - cursor);
                let (mid_e, tail_e2) = tail_e.split_at_mut(e - s);
                ends_chunks.push(mid_e);
                rest_e = tail_e2;

                let (_, tail_v) = rest_v.split_at_mut(s - cursor);
                let (mid_v, tail_v2) = tail_v.split_at_mut(e - s);
                values_chunks.push(mid_v);
                rest_v = tail_v2;

                cursor = e;
            }
        }

        let tracks_slice = tracks.as_slice().unwrap();
        starts_chunks
            .into_par_iter()
            .zip(ends_chunks.into_par_iter())
            .zip(values_chunks.into_par_iter())
            .enumerate()
            .for_each(|(query, ((s_chunk, e_chunk), v_chunk))| {
                let o_s = track_offsets[query] as usize;
                let o_e = track_offsets[query + 1] as usize;
                if o_s == o_e {
                    return;
                }
                let track = &tracks_slice[o_s..o_e];
                let scan = &scanned_masks[o_s..o_e];
                let n_elems = scan.len();
                let n_runs = scan[n_elems - 1] as usize;

                let mut compacted = vec![0i32; n_runs + 1];
                compacted[n_runs] = n_elems as i32;
                for i in 0..n_elems {
                    if i == 0 {
                        compacted[0] = 0;
                    } else if scan[i] != scan[i - 1] {
                        compacted[scan[i] as usize - 1] = i as i32;
                    }
                }

                let start = regions[[query, 1]];
                for k in 0..n_runs {
                    s_chunk[k] = compacted[k] + start;
                    e_chunk[k] = compacted[k + 1] + start;
                    v_chunk[k] = track[compacted[k] as usize];
                }
            });
    } else {
        for query in 0..n_queries {
            let o_s = track_offsets[query] as usize;
            let o_e = track_offsets[query + 1] as usize;
            // Numba: if o_s == o_e: continue
            if o_s == o_e {
                continue;
            }
            let track = &tracks.as_slice().unwrap()[o_s..o_e];
            let scan = &scanned_masks[o_s..o_e];
            let n_elems = scan.len();
            let n_runs = scan[n_elems - 1] as usize;

            // _compact_mask: recovers run-boundary indices
            // Numba:
            //   compacted_backward_mask = np.empty(n_runs + 1, np.int32)
            //   compacted_backward_mask[-1] = n_elems
            //   for i in prange(n_elems):
            //       if i == 0: compacted_backward_mask[0] = 0
            //       elif scan[i] != scan[i-1]: compacted_backward_mask[scan[i] - 1] = i
            let mut compacted = vec![0i32; n_runs + 1];
            compacted[n_runs] = n_elems as i32;
            for i in 0..n_elems {
                if i == 0 {
                    compacted[0] = 0;
                } else if scan[i] != scan[i - 1] {
                    compacted[scan[i] as usize - 1] = i as i32;
                }
            }

            // values = track[compacted[:-1]]
            // starts/ends = compacted[:-1] + region_start, compacted[1:] + region_start
            let s = interval_offsets[query] as usize;
            let start = regions[[query, 1]]; // region start (absolute genomic coord)

            // Numba: compacted_backward_mask += start  (in-place, then used for starts/ends)
            // We apply the shift at write time to avoid mutating compacted.
            let n = n_runs; // == len(values)
            for k in 0..n {
                all_starts[s + k] = compacted[k] + start;
                all_ends[s + k] = compacted[k + 1] + start;
                all_values[s + k] = track[compacted[k] as usize];
            }
        }
    }

    (
        Array1::from_vec(all_starts),
        Array1::from_vec(all_ends),
        Array1::from_vec(all_values),
        Array1::from_vec(interval_offsets),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Expected values hand-derived from the numba algorithm (verified by running
    /// the Python reference implementation with np.uint64 arithmetic).
    #[test]
    fn test_xorshift64_vectors() {
        // xorshift64(1):
        //   x=1; x ^= 1<<13=0x2000 → 0x2001
        //   x ^= 0x2001>>7=0x40   → 0x2041
        //   x ^= 0x2041<<17=0x408200000 → 0x40822041 = 1082269761
        assert_eq!(xorshift64(1), 1_082_269_761_u64);

        // xorshift64(2) = 2164539522 (verified via Python np.uint64)
        assert_eq!(xorshift64(2), 2_164_539_522_u64);

        // xorshift64(42) = 45454805674
        assert_eq!(xorshift64(42), 45_454_805_674_u64);

        // xorshift64(0xdeadbeef) = 4018790486776397394
        assert_eq!(xorshift64(0xdeadbeef), 4_018_790_486_776_397_394_u64);

        // xorshift64(u64::MAX) — wrapping behaviour: 2**64-1 = 0xffffffffffffffff
        // result = 0x3f801fc0 = 1065361344 (verified via Python np.uint64)
        assert_eq!(xorshift64(u64::MAX), 1_065_361_344_u64);
    }

    #[test]
    fn test_hash4_vectors() {
        // hash4(1,2,3,4) = 11323120931611735037 (verified via Python)
        assert_eq!(hash4(1, 2, 3, 4), 11_323_120_931_611_735_037_u64);

        // hash4(0,0,0,0): h=0; xorshift64(0)=0 at each step → 0
        assert_eq!(hash4(0, 0, 0, 0), 0_u64);

        // hash4(0xdeadbeef, 0xcafe, 0xbabe, 1) = 5244362157944750963
        assert_eq!(
            hash4(0xdeadbeef, 0xcafe, 0xbabe, 1),
            5_244_362_157_944_750_963_u64
        );
    }

    // ------------------------------------------------------------------ //
    // apply_insertion_fill tests                                           //
    // ------------------------------------------------------------------ //

    /// Helper: allocate out, run apply_insertion_fill, return the filled slice.
    fn run_fill(
        out_size: usize,
        out_idx: usize,
        writable_length: usize,
        v_len: i64,
        track: &[f32],
        v_rel_pos: i64,
        strategy_id: i64,
        params: &[f64],
        base_seed: u64,
        query: u64,
        hap: u64,
    ) -> Vec<f32> {
        let mut out_arr = Array1::<f32>::zeros(out_size);
        {
            let mut out_view = out_arr.view_mut();
            let track_arr = Array1::from_vec(track.to_vec());
            let params_arr = Array1::from_vec(params.to_vec());
            apply_insertion_fill(
                &mut out_view,
                out_idx,
                writable_length,
                v_len,
                track_arr.view(),
                v_rel_pos,
                strategy_id,
                params_arr.view(),
                base_seed,
                query,
                hap,
            );
        }
        out_arr.to_vec()
    }

    /// REPEAT_5P_NORM: val = track[v_rel_pos] / v_len (f32/f32 → f32).
    ///
    /// track = [1.0, 6.0, 2.0], v_rel_pos = 1 → track[1] = 6.0f32
    /// v_len = 3 → val = 6.0f32 / 3f32 = 2.0f32
    /// writable_length = 3 → out[0..3] = [2.0, 2.0, 2.0]
    /// sum = 6.0 = track[v_rel_pos] ✓ (sum-preserving)
    #[test]
    fn test_repeat_5p_norm() {
        let track = [1.0f32, 6.0, 2.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        // val = 6.0f32 / 3f32 = 2.0f32  (exact in f32)
        let expected_val = 6.0f32 / 3.0f32;
        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            REPEAT_5P_NORM,
            &[0.0],
            0,
            0,
            0,
        );
        assert_eq!(result.len(), writable_length);
        for &v in &result {
            assert_eq!(v, expected_val, "REPEAT_5P_NORM: expected {expected_val}, got {v}");
        }
        // Sum preservation check
        let sum: f32 = result.iter().sum();
        assert_eq!(sum, track[v_rel_pos as usize]);
    }

    /// REPEAT_5P_NORM with non-divisible values: verifies f32 precision.
    ///
    /// track = [0.0, 1.0, 0.0], v_rel_pos = 1, v_len = 3
    /// val = 1.0f32 / 3f32 (not exactly representable)
    #[test]
    fn test_repeat_5p_norm_precision() {
        let track = [0.0f32, 1.0, 0.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        let expected_val = 1.0f32 / 3.0f32; // same f32 division as numba
        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            REPEAT_5P_NORM,
            &[0.0],
            0,
            0,
            0,
        );
        for &v in &result {
            assert_eq!(v, expected_val);
        }
    }

    /// CONSTANT: fills every position with params[0] cast to f32.
    ///
    /// params[0] = 3.14 (f64), writable_length = 4
    /// expected: each position = 3.14f64 as f32 = 3.14f32
    #[test]
    fn test_constant() {
        let track = [0.0f32, 0.0, 0.0, 0.0, 0.0];
        let result = run_fill(5, 1, 4, 1, &track, 0, CONSTANT, &[3.14f64], 0, 0, 0);
        let expected = 3.14f64 as f32;
        for i in 1..5 {
            assert_eq!(result[i], expected, "CONSTANT at position {i}");
        }
        // position 0 should be untouched (still 0)
        assert_eq!(result[0], 0.0f32);
    }

    /// CONSTANT with NaN: the default Constant(value=NaN) should write NaN.
    #[test]
    fn test_constant_nan() {
        let track = [0.0f32];
        let result = run_fill(3, 0, 3, 1, &track, 0, CONSTANT, &[f64::NAN], 0, 0, 0);
        for &v in &result {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    /// FLANK_SAMPLE: deterministic given seed.
    ///
    /// Setup: track = [10.0, 20.0, 30.0, 40.0, 50.0], v_rel_pos=2, flank_width=1
    /// pool: pool_lo = max(0, 2-1)=1, pool_hi = min(4, 2+1)=3, pool_size=3
    /// pool values: track[1..=3] = [20.0, 30.0, 40.0]
    ///
    /// For base_seed=42, query=7, hap=1, out_idx=0, writable_length=4:
    ///
    /// Hand-derived using verified hash4:
    ///   i=0: seed = hash4(42, 7, 1, 0); offset = seed % 3; track[1+offset]
    ///   i=1: seed = hash4(42, 7, 1, 1); offset = seed % 3; track[1+offset]
    ///   i=2: seed = hash4(42, 7, 1, 2); offset = seed % 3; track[1+offset]
    ///   i=3: seed = hash4(42, 7, 1, 3); offset = seed % 3; track[1+offset]
    ///
    /// Computed by applying xorshift64 chain:
    ///   hash4(42, 7, 1, 0) = xorshift64(xorshift64(xorshift64(42^7) ^ 1) ^ 0)
    ///   We compute all hash values first and derive offsets below.
    #[test]
    fn test_flank_sample_deterministic() {
        let track = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let v_rel_pos = 2i64;
        let flank_width = 1i64; // pool_lo=1, pool_hi=3, pool_size=3
        let pool_lo = 1i64;
        let pool_size = 3u64;

        let base_seed = 42u64;
        let query = 7u64;
        let hap = 1u64;
        let out_idx = 0usize;
        let writable_length = 4;

        // Hand-compute the expected hash values and pool indices:
        // This uses our verified hash4 function.
        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
                let offset = (seed % pool_size) as i64;
                track[(pool_lo + offset) as usize]
            })
            .collect();

        let result = run_fill(
            writable_length,
            out_idx,
            writable_length,
            1,
            &track,
            v_rel_pos,
            FLANK_SAMPLE,
            &[flank_width as f64],
            base_seed,
            query,
            hap,
        );

        assert_eq!(result, expected, "FLANK_SAMPLE: result did not match expected");

        // Spot-check the first index by computing hash4 explicitly:
        // hash4(42, 7, 1, 0):
        //   h = 42
        //   h = xorshift64(42 ^ 7) = xorshift64(45) = ?
        let h0 = xorshift64(42 ^ 7); // xorshift64(45)
        let h1 = xorshift64(h0 ^ 1);
        let h2 = xorshift64(h1 ^ 0);
        let offset0 = (h2 % pool_size) as i64;
        assert_eq!(
            result[0],
            track[(pool_lo + offset0) as usize],
            "FLANK_SAMPLE spot-check i=0 failed"
        );
    }

    /// FLANK_SAMPLE with out_idx > 0: verifies that out_idx+i is used, not just i.
    #[test]
    fn test_flank_sample_out_idx_offset() {
        let track = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let v_rel_pos = 2i64;
        let flank_width = 1i64;
        let pool_lo = 1i64;
        let pool_size = 3u64;
        let base_seed = 100u64;
        let query = 3u64;
        let hap = 0u64;
        let out_idx = 5usize;
        let writable_length = 3;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
                let offset = (seed % pool_size) as i64;
                track[(pool_lo + offset) as usize]
            })
            .collect();

        let mut out_arr = Array1::<f32>::zeros(out_idx + writable_length);
        {
            let mut out_view = out_arr.view_mut();
            let track_arr = Array1::from_vec(track.to_vec());
            let params_arr = Array1::from_vec(vec![flank_width as f64]);
            apply_insertion_fill(
                &mut out_view,
                out_idx,
                writable_length,
                1,
                track_arr.view(),
                v_rel_pos,
                FLANK_SAMPLE,
                params_arr.view(),
                base_seed,
                query,
                hap,
            );
        }
        let result: Vec<f32> = out_arr.iter().skip(out_idx).cloned().collect();
        assert_eq!(result, expected, "FLANK_SAMPLE out_idx offset test failed");
    }

    /// INTERPOLATE order=1 (linear interpolation).
    ///
    /// order=1 → k = ceil(2/2) = 1, n_anchors = 2
    /// track = [0.0, 4.0, 8.0] (indices 0,1,2), v_rel_pos=1, v_len=3
    ///
    /// Anchors (5' then 3' side):
    ///   xs[0] = -0.0 = 0.0, ys[0] = track[max(1-0,0)=1] = 4.0
    ///   xs[1] = 3.0+0.0 = 3.0, ys[1] = track[min(1+1+0,2)=2] = 8.0
    ///
    /// Lagrange at x=0: term_0 = 4.0 * (0-3)/(0-3) = 4.0*(-3/-3) = 4.0*1.0 = 4.0
    ///                  term_1 = 8.0 * (0-0)/(3-0) = 8.0*0 = 0.0; acc=4.0
    /// Lagrange at x=1: term_0 = 4.0 * (1-3)/(0-3) = 4.0*(-2/-3) = 4.0*0.6667 = 2.6667
    ///                  term_1 = 8.0 * (1-0)/(3-0) = 8.0*(1/3) = 2.6667; acc=5.3333
    /// Lagrange at x=2: term_0 = 4.0 * (2-3)/(0-3) = 4.0*(1/3) = 1.3333
    ///                  term_1 = 8.0 * (2-0)/(3-0) = 8.0*(2/3) = 5.3333; acc=6.6667
    ///
    /// Check endpoints: at x=0 → 4.0 = track[1] ✓; at x=3 → 8.0 = track[2] ✓
    #[test]
    fn test_interpolate_order1() {
        let track = [0.0f32, 4.0, 8.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        // Hand-computed Lagrange values (f64 arithmetic, stored to f32):
        // xs = [0.0, 3.0], ys = [4.0, 8.0]
        // x=0: acc = 4.0*(0-3)/(0-3) + 8.0*(0-0)/(3-0) = 4.0 + 0.0 = 4.0
        // x=1: acc = 4.0*(1-3)/(0-3) + 8.0*(1-0)/(3-0) = 4.0*(2/3) + 8.0*(1/3)
        //           = 8.0/3.0 + 8.0/3.0 = 16.0/3.0
        // x=2: acc = 4.0*(2-3)/(0-3) + 8.0*(2-0)/(3-0) = 4.0*(1/3) + 8.0*(2/3)
        //           = 4.0/3.0 + 16.0/3.0 = 20.0/3.0
        let xs = [0.0f64, 3.0f64];
        let ys = [4.0f64, 8.0f64];
        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..2usize {
                    let mut term = ys[a];
                    for b in 0..2usize {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[1.0f64], // order=1
            0,
            0,
            0,
        );

        assert_eq!(result.len(), writable_length);
        // Endpoint check: at i=0, result should equal ys[0]=track[v_rel_pos]=4.0
        assert_eq!(result[0], 4.0f32, "order=1 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=1 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE order=2.
    ///
    /// order=2 → k = ceil(3/2) = 2, n_anchors = 4
    /// track = [1.0, 2.0, 4.0, 8.0, 16.0], v_rel_pos=2, v_len=2
    ///
    /// Anchors:
    ///   5' side (j=0,1):
    ///     xs[0]=-0.0=0.0, ys[0]=track[max(2-0,0)=2]=4.0
    ///     xs[1]=-1.0,     ys[1]=track[max(2-1,0)=1]=2.0
    ///   3' side (j=0,1):
    ///     xs[2]=2.0+0.0=2.0, ys[2]=track[min(2+1+0,4)=3]=8.0
    ///     xs[3]=2.0+1.0=3.0, ys[3]=track[min(2+1+1,4)=4]=16.0
    ///
    /// Lagrange at x=0,1 hand-computed via the same formula.
    #[test]
    fn test_interpolate_order2() {
        let track = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        let v_rel_pos = 2i64;
        let v_len = 2i64;
        let writable_length = 2;

        // Anchors: xs=[0.0, -1.0, 2.0, 3.0], ys=[4.0, 2.0, 8.0, 16.0]
        let xs = [0.0f64, -1.0f64, 2.0f64, 3.0f64];
        let ys = [4.0f64, 2.0f64, 8.0f64, 16.0f64];
        let n = 4usize;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..n {
                    let mut term = ys[a];
                    for b in 0..n {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[2.0f64], // order=2
            0,
            0,
            0,
        );

        // At x=0, result should equal ys[0] = track[v_rel_pos] = 4.0
        assert_eq!(result[0], 4.0f32, "order=2 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=2 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE order=3.
    ///
    /// order=3 → k = ceil(4/2) = 2, n_anchors = 4 (same as order=2)
    /// (The numba formula k=(order+1+1)//2 gives k=2 for both order=2 and order=3)
    /// track = [3.0, 1.0, 5.0, 9.0, 2.0, 6.0], v_rel_pos=2, v_len=4
    ///
    /// Anchors:
    ///   5' side (j=0,1):
    ///     xs[0]=0.0, ys[0]=track[2]=5.0
    ///     xs[1]=-1.0, ys[1]=track[1]=1.0
    ///   3' side (j=0,1):
    ///     xs[2]=4.0, ys[2]=track[3]=9.0
    ///     xs[3]=5.0, ys[3]=track[4]=2.0
    #[test]
    fn test_interpolate_order3() {
        let track = [3.0f32, 1.0, 5.0, 9.0, 2.0, 6.0];
        let v_rel_pos = 2i64;
        let v_len = 4i64;
        let writable_length = 4;

        // k=2, n_anchors=4
        let xs = [0.0f64, -1.0f64, 4.0f64, 5.0f64];
        let ys = [5.0f64, 1.0f64, 9.0f64, 2.0f64];
        let n = 4usize;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..n {
                    let mut term = ys[a];
                    for b in 0..n {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[3.0f64], // order=3
            0,
            0,
            0,
        );

        // At x=0, result should equal track[v_rel_pos]=5.0
        assert_eq!(result[0], 5.0f32, "order=3 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=3 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE: verify that order=1 at x=v_len gives the 3' anchor value.
    ///
    /// With track=[2.0, 10.0, 6.0], v_rel_pos=1, v_len=2:
    ///   xs=[0.0, 2.0], ys=[10.0, 6.0]
    ///   At x=0: acc = 10.0*(0-2)/(0-2) + 6.0*(0-0)/(2-0) = 10.0 + 0.0 = 10.0 ✓
    ///   At x=1: acc = 10.0*(1-2)/(0-2) + 6.0*(1-0)/(2-0) = 10.0*0.5 + 6.0*0.5 = 8.0
    ///   (Note: x=v_len=2 would be exactly 6.0 but writable_length=2 so we test x=0,1)
    #[test]
    fn test_interpolate_order1_endpoints() {
        let track = [2.0f32, 10.0, 6.0];
        let v_rel_pos = 1i64;
        let v_len = 2i64;

        // writable_length = v_len = 2, covering x=0,1
        let result = run_fill(
            2,
            0,
            2,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[1.0f64],
            0,
            0,
            0,
        );

        // x=0 must equal track[v_rel_pos] = 10.0
        assert_eq!(result[0], 10.0f32, "left endpoint");

        // x=1: hand-computed
        // xs=[0.0, 2.0], ys=[10.0, 6.0]
        // term_0 = 10.0 * (1-2)/(0-2) = 10.0 * 0.5 = 5.0
        // term_1 = 6.0 * (1-0)/(2-0) = 6.0 * 0.5 = 3.0; acc=8.0
        let x = 1.0f64;
        let xs = [0.0f64, 2.0f64];
        let ys = [10.0f64, 6.0f64];
        let mut acc = 0.0f64;
        for a in 0..2 {
            let mut term = ys[a];
            for b in 0..2 {
                if b == a { continue; }
                term *= (x - xs[b]) / (xs[a] - xs[b]);
            }
            acc += term;
        }
        assert_eq!(result[1], acc as f32, "midpoint check");
    }

    /// REPEAT_5P: fills with track[v_rel_pos] directly.
    #[test]
    fn test_repeat_5p() {
        let track = [5.0f32, 11.0, 7.0];
        let v_rel_pos = 1i64;
        let result = run_fill(4, 0, 4, 4, &track, v_rel_pos, REPEAT_5P, &[0.0], 0, 0, 0);
        for &v in &result {
            assert_eq!(v, 11.0f32, "REPEAT_5P: expected 11.0");
        }
    }

    // ================================================================== //
    // shift_and_realign_track_sparse tests                                //
    // ================================================================== //

    /// Helper to build the split (2, n) offsets and call `shift_and_realign_track_sparse`.
    fn run_singular(
        geno_v_idxs: &[i32],
        geno_offsets_1d: &[i64], // 1-D (n+1)
        offset_idx: usize,
        v_starts: &[i32],
        ilens: &[i32],
        shift: i64,
        track: &[f32],
        query_start: i64,
        out_len: usize,
        params: &[f64],
        keep: Option<&[bool]>,
        strategy_id: i64,
        base_seed: u64,
        query: u64,
        hap: u64,
    ) -> Vec<f32> {
        use ndarray::Array1;
        let n = geno_offsets_1d.len() - 1;
        let o_starts: Vec<i64> = geno_offsets_1d[..n].to_vec();
        let o_stops: Vec<i64> = geno_offsets_1d[1..].to_vec();

        let gvi_arr = Array1::from_vec(geno_v_idxs.to_vec());
        let os_arr = Array1::from_vec(o_starts);
        let oe_arr = Array1::from_vec(o_stops);
        let vs_arr = Array1::from_vec(v_starts.to_vec());
        let il_arr = Array1::from_vec(ilens.to_vec());
        let track_arr = Array1::from_vec(track.to_vec());
        let params_arr = Array1::from_vec(params.to_vec());

        let mut out_arr = Array1::<f32>::zeros(out_len);
        {
            let mut out_view = out_arr.view_mut();
            let keep_arr_opt = keep.map(|k| Array1::from_vec(k.to_vec()));
            let keep_view = keep_arr_opt.as_ref().map(|a| a.view());
            shift_and_realign_track_sparse(
                offset_idx,
                gvi_arr.view(),
                os_arr.view(),
                oe_arr.view(),
                vs_arr.view(),
                il_arr.view(),
                shift,
                track_arr.view(),
                query_start,
                &mut out_view,
                params_arr.view(),
                keep_view,
                strategy_id,
                base_seed,
                query,
                hap,
            );
        }
        out_arr.to_vec()
    }

    /// No variants → out = track[:length] (shift must be 0).
    #[test]
    fn test_singular_no_variants() {
        // track = [1.0, 2.0, 3.0, 4.0, 5.0], no variants, out_len = 4
        let track = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let geno_v_idxs: Vec<i32> = vec![];
        let geno_offsets = vec![0i64, 0]; // one empty group
        let v_starts: Vec<i32> = vec![];
        let ilens: Vec<i32> = vec![];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0, // shift
            &track,
            0, // query_start
            4, // out_len
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        assert_eq!(result, [1.0f32, 2.0, 3.0, 4.0], "no variants: copy track[:length]");
    }

    /// Deletion: track[v_rel_pos] repeated for writable_length; track advances by
    /// |v_rel_end|.
    ///
    /// Setup:
    ///   track = [10.0, 20.0, 30.0, 40.0, 50.0], query_start = 0, out_len = 4
    ///   variant at v_start=1, ilen=-2 → v_rel_pos=1, v_diff=-2, v_rel_end=4
    ///   v_len = max(0,-2)+1 = 1
    ///   Expected: track[0..1] = [10.0], then track[1] repeated 1 time = [20.0],
    ///   then track[4:] = [50.0], padded 0.0 if needed.
    ///   Actually: out[0] = track[0] = 10.0 (ref up to v_rel_pos=1, track_len=1-0=1)
    ///             out[1] = track[v_rel_pos=1] = 20.0 (repeated 1 time = v_len=1)
    ///             track_idx = v_rel_end = 4; out_idx = 2
    ///             fill rest: track[4:] = [50.0] → out[2] = 50.0; out[3] = 0.0 (pad)
    #[test]
    fn test_singular_deletion() {
        let track = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let v_starts = [1i32]; // v_start = 1
        let ilens = [-2i32]; // deletion of 2 → v_rel_end = 1 - (-2) + 1 = 4... wait
        // v_rel_end = v_rel_pos - min(0, v_diff) + 1 = 1 - (-2) + 1 = 4
        // Actually: v_rel_end = 1 - min(0, -2) + 1 = 1 - (-2) + 1 = 4
        // v_len = max(0, -2) + 1 = 0 + 1 = 1
        // track up to v_rel_pos=1: track[0..1] = [10.0], out[0] = 10.0
        // v_len=1 repeated: out[1] = track[1] = 20.0
        // track_idx = 4; remaining: track[4..5] = [50.0] → out[2] = 50.0
        // out[3] = 0.0 (trailing pad)
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0,
            &track,
            0,
            4,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        assert_eq!(result[0], 10.0f32, "ref before deletion");
        assert_eq!(result[1], 20.0f32, "deletion: track[v_rel_pos] repeated");
        assert_eq!(result[2], 50.0f32, "ref after deletion (track_idx=4)");
        assert_eq!(result[3], 0.0f32, "trailing pad = 0.0");
    }

    /// Deletion whose `v_rel_end` runs past track end — trailing pad starts from out_idx.
    ///
    /// When a deletion is so large that `v_rel_end` exceeds `track_len`, `track_idx`
    /// advances past the end of `track`, making `writable_ref` negative.  The fixed
    /// kernel clamps `out_end_idx` to `out_idx` (matching the fixed numba kernel's
    /// `max(0, min(unfilled, len(track)-track_idx))`), so the zero-pad covers exactly
    /// `out[out_idx..length]` without overwriting already-written positions.
    ///
    /// Setup:
    ///   track = [1.0, 2.0, 3.0, 4.0, 5.0] (track_len=5), query_start=0, out_len=8
    ///   variant at v_start=3, ilen=-3 → v_rel_pos=3, v_diff=-3, v_rel_end=3-(-3)+1=7
    ///   v_len = max(0,-3)+1 = 1
    ///
    /// Main loop:
    ///   copy track[0..3] → out[0..3] = [1,2,3]; out_idx=3
    ///   deletion REPEAT_5P: out[3] = track[3] = 4.0; out_idx=4
    ///   track_idx = v_rel_end = 7  (past track end = 5!)
    ///
    /// Trailing fill (correct):
    ///   writable_ref = min(4, 5-7) = -2  ← negative, no track bytes remain
    ///   out_end_idx = out_idx = 4  (NOT (4 + -2).max(0) = 2)
    ///   out[4..8] = 0.0
    ///   Final: [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    #[test]
    fn test_singular_deletion_past_track_end() {
        // track_len=5, out_len=8, deletion at v_start=3 with ilen=-3
        let track = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let v_starts = [3i32];
        let ilens = [-3i32]; // v_diff=-3, v_rel_end = 3-(-3)+1 = 7 (past track_len=5)
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0, // shift
            &track,
            0, // query_start
            8, // out_len
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );

        // out[0..4] from main loop; zero-pad covers out[4..8] from out_idx (not index 2).
        assert_eq!(result[0], 1.0f32, "ref[0]");
        assert_eq!(result[1], 2.0f32, "ref[1]");
        assert_eq!(result[2], 3.0f32, "ref[2] — must NOT be overwritten by zero-pad");
        assert_eq!(result[3], 4.0f32, "deletion REPEAT_5P value — must NOT be overwritten");
        assert_eq!(result[4], 0.0f32, "zero-pad[4]");
        assert_eq!(result[5], 0.0f32, "zero-pad[5]");
        assert_eq!(result[6], 0.0f32, "zero-pad[6]");
        assert_eq!(result[7], 0.0f32, "zero-pad[7]");
    }

    /// Deletion drives track_idx past the track end (overshoot) — trailing pad from out_idx.
    ///
    /// Mirrors ``overshoot_ref_past_contig`` from reconstruct/mod.rs.
    /// When writable_ref <= 0, out_end_idx must be clamped to out_idx so that
    /// out[out_idx..length] is zero-padded without overwriting already-written positions.
    ///
    /// The fixed numba kernel uses ``max(0, min(unfilled, len(track)-track_idx))``,
    /// giving writable_ref=0 and out_end_idx=out_idx. The Rust kernel must match.
    ///
    /// Setup (identical to test_singular_deletion_past_track_end):
    ///   track=[1,2,3,4,5] (len=5), out_len=8, deletion at v_start=3, ilen=-3
    ///   v_rel_end=7 (>track_len=5) → track_idx advances past track end
    ///   After main loop: out[0..4]=[1,2,3,4], out_idx=4, track_idx=7
    ///
    /// Trailing fill (correct):
    ///   writable_ref = min(4, 5-7) = -2  ← negative
    ///   out_end_idx = out_idx = 4  (NOT (4 + -2).max(0) = 2)
    ///   out[4..8] = 0.0
    ///   Expected: [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    #[test]
    fn overshoot_track_past_end() {
        let track = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let v_starts = [3i32];
        let ilens = [-3i32];
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0,
            &track,
            0,
            8,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        // out[0..4] from main loop; out[4..8] zero-padded from out_idx (not index 2)
        assert_eq!(
            result,
            [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "overshoot: zero-pad must start from out_idx=4, not (out_idx+writable_ref).max(0)=2"
        );
    }

    /// SNP (ilen=0) is SKIPPED — the output copies reference track straight through.
    ///
    /// Setup: track = [1.0, 2.0, 3.0, 4.0], query_start=0, out_len=4
    ///   variant at v_start=2, ilen=0 → SNP, should be skipped
    ///   Expected: out = [1.0, 2.0, 3.0, 4.0] (identical to track, SNP doesn't interrupt)
    #[test]
    fn test_singular_snp_skipped() {
        let track = [1.0f32, 2.0, 3.0, 4.0];
        let v_starts = [2i32];
        let ilens = [0i32]; // SNP
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0,
            &track,
            0,
            4,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        // SNP is skipped — output equals track[:length]
        assert_eq!(result, [1.0f32, 2.0, 3.0, 4.0], "SNP must be skipped for tracks");
    }

    /// Insertion with REPEAT_5P strategy: repeated track[v_rel_pos].
    ///
    /// Setup: track = [5.0, 10.0, 15.0, 20.0, 25.0], query_start=0, out_len=6
    ///   variant at v_start=1, ilen=+2 → v_rel_pos=1, v_diff=2, v_rel_end=2
    ///   v_len = max(0,2)+1 = 3
    ///   REPEAT_5P: repeat track[v_rel_pos=1]=10.0 for writable_length=min(3, 6-1)=3
    ///   ref before: track[0..1] = [5.0] → out[0]
    ///   insertion: out[1..4] = [10.0, 10.0, 10.0]
    ///   track_idx = v_rel_end = 2; remaining: track[2..5] → out[4..6] = [15.0, 20.0]
    #[test]
    fn test_singular_insertion_repeat5p() {
        let track = [5.0f32, 10.0, 15.0, 20.0, 25.0];
        let v_starts = [1i32];
        let ilens = [2i32]; // insertion
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0,
            &track,
            0,
            6,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        assert_eq!(result[0], 5.0f32, "ref before insertion");
        assert_eq!(result[1], 10.0f32, "insertion REPEAT_5P i=0");
        assert_eq!(result[2], 10.0f32, "insertion REPEAT_5P i=1");
        assert_eq!(result[3], 10.0f32, "insertion REPEAT_5P i=2");
        assert_eq!(result[4], 15.0f32, "ref after insertion (track[2])");
        assert_eq!(result[5], 20.0f32, "ref after insertion (track[3])");
    }

    /// Insertion with CONSTANT strategy: fills with params[0].
    #[test]
    fn test_singular_insertion_constant() {
        let track = [5.0f32, 10.0, 15.0, 20.0];
        let v_starts = [1i32];
        let ilens = [1i32]; // insertion: v_len = 2
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];
        let fill_val = 99.0f64;

        // out_len=5: ref[0..1]=[5.0], ins[1..3]=[99.0,99.0], ref after=track[2..4]
        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0,
            &track,
            0,
            5,
            &[fill_val],
            None,
            CONSTANT,
            0,
            0,
            0,
        );
        assert_eq!(result[0], 5.0f32, "ref before insertion");
        assert_eq!(result[1], fill_val as f32, "CONSTANT fill i=0");
        assert_eq!(result[2], fill_val as f32, "CONSTANT fill i=1");
        assert_eq!(result[3], 15.0f32, "ref after insertion (track[2])");
        assert_eq!(result[4], 20.0f32, "ref after insertion (track[3])");
    }

    /// Shift: when shift > 0, track values are consumed from a later position.
    ///
    /// track = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], shift=2, no variants, out_len=4
    /// Expected: track[2..6] = [2.0, 3.0, 4.0, 5.0]
    #[test]
    fn test_singular_shift_no_variants() {
        // With no variants, shift > 0 is handled by the post-loop track_idx adjustment.
        // Numba: if shifted < shift: track_idx += shift - shifted; ...
        // But the loop is never entered, so shifted stays 0.
        // Post-loop: track_idx = 0 + shift = 2; writable_ref = min(4, 6-2) = 4
        let track = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let geno_v_idxs: Vec<i32> = vec![];
        let geno_offsets = vec![0i64, 0]; // empty group
        let v_starts: Vec<i32> = vec![];
        let ilens: Vec<i32> = vec![];

        // Note: numba says "guaranteed to have shift = 0" when n_variants == 0,
        // so this tests the case where the variant list is empty BUT shift is 0.
        // For non-zero shift with no variants, it's technically undefined (won't be
        // called in production), but let's verify shift=0 with an offset.
        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            0, // shift=0 (no variants path)
            &track,
            0,
            4,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        assert_eq!(result, [0.0f32, 1.0, 2.0, 3.0], "no variants + shift=0: copy track[:4]");
    }

    /// Shift=2 with one insertion variant: verify shift-through-variant logic.
    ///
    /// track=[0,1,2,3,4,5,6], query_start=0, shift=2, out_len=4
    /// Insertion at v_start=1, ilen=+3 → v_rel_pos=1, v_len=4
    ///
    /// ref_shift_dist = 1 - 0 = 1
    /// shifted + ref_shift_dist + v_len = 0 + 1 + 4 = 5 >= shift=2, so NOT "need more"
    /// shifted + ref_shift_dist = 0 + 1 = 1 < shift=2, so NOT "can finish without variant"
    /// allele_start_idx = 2 - 0 - 1 = 1; shifted=2; allele_start_idx(1) != v_len(4)
    /// track_idx = v_rel_pos = 1; v_len -= 1 → v_len = 3
    ///
    /// Then v_diff=3 > 0, strategy=REPEAT_5P: repeat track[v_rel_pos=1]=1.0 for writable=min(3,4)=3
    /// out[0..3] = [1.0, 1.0, 1.0]; track_idx = v_rel_end = 2; out_idx = 3
    /// fill rest: track[2:] → out[3] = track[2] = 2.0
    #[test]
    fn test_singular_shift_through_insertion() {
        let track: Vec<f32> = (0..7).map(|x| x as f32).collect();
        let v_starts = [1i32]; // insertion at pos 1
        let ilens = [3i32]; // +3 → v_len = 4, v_rel_end = 1 - 0 + 1 = 2
        let geno_v_idxs = [0i32];
        let geno_offsets = [0i64, 1];

        let result = run_singular(
            &geno_v_idxs,
            &geno_offsets,
            0,
            &v_starts,
            &ilens,
            2, // shift
            &track,
            0,
            4,
            &[0.0],
            None,
            REPEAT_5P,
            0,
            0,
            0,
        );
        // shifted=2, allele_start_idx=1 ≠ v_len=4 → track_idx=1, v_len=3
        // v_diff=3≠0 and REPEAT_5P: out[0..3] = track[v_rel_pos=1] = 1.0
        // out[3] = track[2] = 2.0
        assert_eq!(result[0], 1.0f32, "insertion repeat after shift");
        assert_eq!(result[1], 1.0f32, "insertion repeat");
        assert_eq!(result[2], 1.0f32, "insertion repeat");
        assert_eq!(result[3], 2.0f32, "ref after insertion");
    }

    // ================================================================== //
    // shift_and_realign_tracks_sparse (batch) tests                      //
    // ================================================================== //

    /// Helper for the batch function.
    fn run_batch(
        out_len: usize,
        out_offsets: &[i64],
        regions: &[[i32; 3]],
        shifts: &[i32],   // flat, will be reshaped (n_q, ploidy)
        geno_offset_idx: &[i64], // flat (n_q * ploidy)
        geno_v_idxs: &[i32],
        geno_offsets_1d: &[i64],
        v_starts: &[i32],
        ilens: &[i32],
        tracks: &[f32],
        track_offsets: &[i64],
        params: &[f64],
        keep: Option<(&[bool], &[i64])>,
        strategy_id: i64,
        base_seed: u64,
        ploidy: usize,
        parallel: bool,
    ) -> Vec<f32> {
        use ndarray::{Array1, Array2};
        let n_q = regions.len();
        // Build (2, n_q*ploidy) offsets
        let n = geno_offsets_1d.len() - 1;
        let o_starts: Vec<i64> = geno_offsets_1d[..n].to_vec();
        let o_stops: Vec<i64> = geno_offsets_1d[1..].to_vec();

        let regions_arr = Array2::from_shape_vec(
            (n_q, 3),
            regions.iter().flat_map(|r| r.iter().cloned()).collect(),
        )
        .unwrap();
        let shifts_arr = Array2::from_shape_vec(
            (n_q, ploidy),
            shifts.to_vec(),
        )
        .unwrap();
        let goi_arr = Array2::from_shape_vec(
            (n_q, ploidy),
            geno_offset_idx.to_vec(),
        )
        .unwrap();

        let out_offsets_arr = Array1::from_vec(out_offsets.to_vec());
        let gvi_arr = Array1::from_vec(geno_v_idxs.to_vec());
        let os_arr = Array1::from_vec(o_starts);
        let oe_arr = Array1::from_vec(o_stops);
        let vs_arr = Array1::from_vec(v_starts.to_vec());
        let il_arr = Array1::from_vec(ilens.to_vec());
        let tracks_arr = Array1::from_vec(tracks.to_vec());
        let to_arr = Array1::from_vec(track_offsets.to_vec());
        let params_arr = Array1::from_vec(params.to_vec());

        let mut out_arr = Array1::<f32>::zeros(out_len);

        let (keep_arr_opt, keep_off_arr_opt) = if let Some((k, ko)) = keep {
            (
                Some(Array1::from_vec(k.to_vec())),
                Some(Array1::from_vec(ko.to_vec())),
            )
        } else {
            (None, None)
        };

        shift_and_realign_tracks_sparse(
            out_arr.view_mut(),
            out_offsets_arr.view(),
            regions_arr.view(),
            shifts_arr.view(),
            goi_arr.view(),
            gvi_arr.view(),
            os_arr.view(),
            oe_arr.view(),
            vs_arr.view(),
            il_arr.view(),
            tracks_arr.view(),
            to_arr.view(),
            params_arr.view(),
            keep_arr_opt.as_ref().map(|a| a.view()),
            keep_off_arr_opt.as_ref().map(|a| a.view()),
            strategy_id,
            base_seed,
            parallel,
        );

        out_arr.to_vec()
    }

    /// Batch with 1 query, 1 hap, no variants → copies track.
    #[test]
    fn test_batch_single_no_variants() {
        // track = [1.0, 2.0, 3.0, 4.0, 5.0] for query 0
        let tracks = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let regions = [[0i32, 0, 4]]; // length=4
        let shifts = [0i32];
        let geno_offset_idx = [0i64]; // (1, 1)
        let geno_v_idxs: Vec<i32> = vec![];
        let geno_offsets = [0i64, 0]; // empty group
        let v_starts: Vec<i32> = vec![];
        let ilens: Vec<i32> = vec![];
        let track_offsets = [0i64, 5];
        let out_offsets = [0i64, 4];
        let params = [0.0f64];

        let result = run_batch(
            4,
            &out_offsets,
            &regions,
            &shifts,
            &geno_offset_idx,
            &geno_v_idxs,
            &geno_offsets,
            &v_starts,
            &ilens,
            &tracks,
            &track_offsets,
            &params,
            None,
            REPEAT_5P,
            0,
            1, // ploidy
            false,
        );
        assert_eq!(result, [1.0f32, 2.0, 3.0, 4.0], "batch single: copy track[:4]");
    }

    /// Batch with 2 queries, 1 hap each, SNPs — must pass through unchanged.
    #[test]
    fn test_batch_two_queries_snps() {
        // query 0: track[0..3] = [1.0, 2.0, 3.0], SNP at pos 1 (skipped) → out=[1,2,3]
        // query 1: track[3..6] = [4.0, 5.0, 6.0], no variants → out=[4,5,6]
        let tracks = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let regions = [[0i32, 0, 3], [0, 10, 13]];
        let shifts = [0i32, 0];
        let geno_offset_idx = [0i64, 1]; // q0→group0, q1→group1
        let geno_v_idxs = [0i32]; // query 0 has SNP variant 0
        let v_starts = [1i32]; // v at pos 1 (within q0 [0,3))
        let ilens = [0i32]; // SNP → should be skipped
        let geno_offsets = [0i64, 1, 1]; // group0=[0..1], group1=[1..1]=empty
        let track_offsets = [0i64, 3, 6];
        let out_offsets = [0i64, 3, 6];
        let params = [0.0f64];

        let result = run_batch(
            6,
            &out_offsets,
            &regions,
            &shifts,
            &geno_offset_idx,
            &geno_v_idxs,
            &geno_offsets,
            &v_starts,
            &ilens,
            &tracks,
            &track_offsets,
            &params,
            None,
            REPEAT_5P,
            0,
            1,
            false,
        );
        // SNP skipped → query 0 output = track[0..3]
        assert_eq!(result[..3], [1.0f32, 2.0, 3.0], "q0: SNP skipped, track copied");
        // No variants in q1 → track[3..6]
        assert_eq!(result[3..], [4.0f32, 5.0, 6.0], "q1: no variants, track copied");
    }

    // ================================================================== //
    // tracks_to_intervals tests                                            //
    // ================================================================== //

    /// Hand-built RLE example with 3 queries:
    /// - q0: empty (track_offsets[0]==track_offsets[1])  → 0 intervals
    /// - q1: all-constant [5.0, 5.0, 5.0] at region [0, 10, 13] → 1 interval [10,13) val=5.0
    /// - q2: two runs [1.0, 1.0, 2.0, 2.0, 2.0] at region [0, 20, 25] → 2 intervals
    ///         [20,22) val=1.0  and  [22,25) val=2.0
    ///
    /// Expected offsets: [0, 0, 1, 3]
    #[test]
    fn test_tracks_to_intervals_hand_built() {
        use super::tracks_to_intervals;
        use ndarray::{Array1, Array2};

        // regions: (n_queries, 3) — (contig_idx, start, end)
        let regions_data = vec![
            0i32, 0, 0,   // q0: empty length
            0i32, 10, 13, // q1: [10, 13), length 3
            0i32, 20, 25, // q2: [20, 25), length 5
        ];
        let regions = Array2::from_shape_vec((3, 3), regions_data).unwrap();

        // tracks: q0 empty, q1 = [5,5,5], q2 = [1,1,2,2,2]
        let tracks_data = vec![5.0f32, 5.0, 5.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let tracks = Array1::from_vec(tracks_data);

        // track_offsets: [0, 0, 3, 8]
        let track_offsets = Array1::from_vec(vec![0i64, 0, 3, 8]);

        let (starts, ends, values, offsets) =
            tracks_to_intervals(regions.view(), tracks.view(), track_offsets.view(), false);

        // offsets: [0, 0, 1, 3]
        assert_eq!(offsets.as_slice().unwrap(), &[0i64, 0, 1, 3], "offsets mismatch");

        // Total intervals = 3
        assert_eq!(starts.len(), 3);
        assert_eq!(ends.len(), 3);
        assert_eq!(values.len(), 3);

        // q1: interval 0 → [10, 13), val=5.0
        assert_eq!(starts[0], 10i32, "q1 start");
        assert_eq!(ends[0], 13i32, "q1 end");
        assert_eq!(values[0], 5.0f32, "q1 value");

        // q2: interval 1 → [20, 22), val=1.0
        assert_eq!(starts[1], 20i32, "q2[0] start");
        assert_eq!(ends[1], 22i32, "q2[0] end");
        assert_eq!(values[1], 1.0f32, "q2[0] value");

        // q2: interval 2 → [22, 25), val=2.0
        assert_eq!(starts[2], 22i32, "q2[1] start");
        assert_eq!(ends[2], 25i32, "q2[1] end");
        assert_eq!(values[2], 2.0f32, "q2[1] value");
    }

    /// All-constant single query: exactly 1 interval covering full range.
    #[test]
    fn test_tracks_to_intervals_all_constant() {
        use super::tracks_to_intervals;
        use ndarray::{Array1, Array2};

        let regions = Array2::from_shape_vec((1, 3), vec![0i32, 100, 107]).unwrap();
        let tracks = Array1::from_vec(vec![3.14f32; 7]);
        let track_offsets = Array1::from_vec(vec![0i64, 7]);

        let (starts, ends, values, offsets) =
            tracks_to_intervals(regions.view(), tracks.view(), track_offsets.view(), false);

        assert_eq!(offsets.as_slice().unwrap(), &[0i64, 1]);
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0], 100i32);
        assert_eq!(ends[0], 107i32);
        assert_eq!(values[0], 3.14f32);
    }

    /// Empty query: track_offsets[0] == track_offsets[1] → 0 intervals, no panic.
    #[test]
    fn test_tracks_to_intervals_empty_query() {
        use super::tracks_to_intervals;
        use ndarray::{Array1, Array2};

        let regions = Array2::from_shape_vec((1, 3), vec![0i32, 50, 50]).unwrap();
        let tracks = Array1::from_vec(vec![]);
        let track_offsets = Array1::from_vec(vec![0i64, 0]);

        let (starts, ends, values, offsets) =
            tracks_to_intervals(regions.view(), tracks.view(), track_offsets.view(), false);

        assert_eq!(offsets.as_slice().unwrap(), &[0i64, 0]);
        assert_eq!(starts.len(), 0);
        assert_eq!(ends.len(), 0);
        assert_eq!(values.len(), 0);
    }

    /// Zero-value intervals ARE included (not filtered).
    #[test]
    fn test_tracks_to_intervals_zero_value_included() {
        use super::tracks_to_intervals;
        use ndarray::{Array1, Array2};

        // track = [0.0, 0.0, 1.0, 0.0] → 3 intervals: [0,2)=0.0, [2,3)=1.0, [3,4)=0.0
        let regions = Array2::from_shape_vec((1, 3), vec![0i32, 0, 4]).unwrap();
        let tracks = Array1::from_vec(vec![0.0f32, 0.0, 1.0, 0.0]);
        let track_offsets = Array1::from_vec(vec![0i64, 4]);

        let (starts, ends, values, offsets) =
            tracks_to_intervals(regions.view(), tracks.view(), track_offsets.view(), false);

        assert_eq!(offsets.as_slice().unwrap(), &[0i64, 3]);
        assert_eq!(starts.len(), 3, "must have 3 intervals including zero-value ones");
        assert_eq!(values[0], 0.0f32, "first interval is zero-value");
        assert_eq!(starts[0], 0i32);
        assert_eq!(ends[0], 2i32);
        assert_eq!(values[1], 1.0f32);
        assert_eq!(values[2], 0.0f32, "third interval is zero-value");
        assert_eq!(starts[2], 3i32);
        assert_eq!(ends[2], 4i32);
    }
}
