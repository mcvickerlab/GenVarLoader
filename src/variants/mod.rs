//! Flat variant gather/fill cores (pure ndarray). PyO3 lives in `crate::ffi`.
pub mod windows;
use ndarray::{Array1, ArrayView1};

/// Flat variant buffers for a `[row_lo, row_hi)` batch row slice (Wave B PR-B1):
/// `alt_data`/`alt_seq_offsets` are the ragged ALT bytes (`gather_alleles`-shaped, one
/// entry per kept variant), `start`/`ilen` are one scalar per kept variant (same order),
/// and `row_offsets` delimits kept variants per `(row, ploid)` output row (length
/// `n_rows*ploidy + 1` — phased ploidy, no unphased-union fold yet). No dataset-global id:
/// variants output is self-contained (issue #313). Lives here (not `record_stream::engine`)
/// so `crate::ffi::stream_core::EngineBackend` can name it without a record_stream<->ffi
/// module cycle.
pub struct VariantsBatch {
    pub alt_data: Array1<u8>,
    pub alt_seq_offsets: Array1<i64>,
    pub start: Array1<i32>,
    pub ilen: Array1<i32>,
    pub row_offsets: Array1<i64>,
    /// Ride-along per-variant INFO columns (Wave B PR-B3a), one entry per requested
    /// `var_fields` INFO column, each with exactly one value per kept variant — i.e.
    /// the same length as `start`/`ilen`, gathered by the same kept `v_idxs`.
    pub info_out: Vec<(String, crate::record_stream::transpose::InfoVals)>,
    /// Ragged per-variant REF allele bytes (Wave B PR-B3a `var_fields`/`ref="allele"`
    /// input), gathered by the SAME kept `v_idxs` as `alt_data`/`start`/`ilen` — i.e.
    /// element-for-element aligned with them. `Some` only when REF was requested
    /// (`want_ref`); `None` (both fields) is the default `var_fields` path and must
    /// stay byte-unchanged (no allocation, no behavior change).
    pub ref_data: Option<Array1<u8>>,
    /// Ragged-array offsets for `ref_data` (length `start.len() + 1`). `Some` iff
    /// `ref_data` is `Some`.
    pub ref_seq_offsets: Option<Array1<i64>>,
}

/// Generic per-row gather core. `T: Copy` — no num-traits needed.
fn gather_rows_impl<T: Copy>(
    geno_offset_idx: ArrayView1<i64>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    data: ArrayView1<T>,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = geno_offset_idx.len();
    let mut out_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        out_offsets[i + 1] = out_offsets[i] + (o_stops[goi] - o_starts[goi]);
    }
    let total = out_offsets[n_rows] as usize;
    let mut v: Vec<T> = Vec::with_capacity(total);
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        let s = o_starts[goi] as usize;
        let e = o_stops[goi] as usize;
        for k in s..e {
            v.push(data[k]);
        }
    }
    (Array1::from_vec(v), out_offsets)
}

/// Per-row i32 gather (variant indices). Mirrors numba `_gather_v_idxs` / `_ss`.
pub fn gather_rows_i32(
    geno_offset_idx: ArrayView1<i64>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    data: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i64>) {
    gather_rows_impl(geno_offset_idx, o_starts, o_stops, data)
}

/// Per-row f32 gather (dosage values). Preserves float32 dtype exactly.
pub fn gather_rows_f32(
    geno_offset_idx: ArrayView1<i64>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    data: ArrayView1<f32>,
) -> (Array1<f32>, Array1<i64>) {
    gather_rows_impl(geno_offset_idx, o_starts, o_stops, data)
}

/// Gather variable-length allele bytestrings. Mirrors numba `_gather_alleles`.
pub fn gather_alleles(
    v_idxs: ArrayView1<i32>,
    allele_bytes: ArrayView1<u8>,
    allele_offsets: ArrayView1<i64>,
) -> (Array1<u8>, Array1<i64>) {
    let n = v_idxs.len();
    let mut seq_offsets = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let v = v_idxs[i] as usize;
        seq_offsets[i + 1] = seq_offsets[i] + (allele_offsets[v + 1] - allele_offsets[v]);
    }
    let total = seq_offsets[n] as usize;
    let mut data = Array1::<u8>::zeros(total);
    let mut dst = 0usize;
    for i in 0..n {
        let v = v_idxs[i] as usize;
        let s = allele_offsets[v] as usize;
        let e = allele_offsets[v + 1] as usize;
        for k in s..e {
            data[dst] = allele_bytes[k];
            dst += 1;
        }
    }
    (data, seq_offsets)
}

/// Window CSR -> flat variant buffers (issue #276 Wave B PR-B1). Shared by every
/// streaming backend's variants-output generate path (`RecordBackend::generate_variants`
/// for VCF/PGEN; the SVAR1 path reuses this too — see that backend's task). Takes the
/// already-clipped, already-flat kept window-local variant indices `v_idxs` (built by the
/// caller's per-row CSR walk + region-overlap clip — see `src/genotypes/mod.rs:68-74` for
/// the identical `v_end = v_start - v_ilen.min(0) + 1` overlap-keep predicate this mirrors)
/// plus the window's static table, and assembles the ragged ALT bytes (via
/// [`gather_alleles`]) and the scalar `start`/`ilen` arrays. Matches the written path's
/// `_assemble_variant_buffers_rust` (`_flat_variants.py`) byte-for-byte given
/// byte-identical inputs — no dataset-global id is produced here (variants output is
/// self-contained, per #313).
///
/// `ref_src` (Wave B PR-B3a) is an optional per-variant REF byte table
/// `(bytes, offsets)`, gathered by the SAME `v_idxs` via [`gather_alleles`] — the same
/// gather ALT uses, so REF stays element-for-element aligned with `start`/`ilen`/ALT.
/// `None` (the default `var_fields` path) yields `(None, None)` with no extra
/// allocation or work — byte-unchanged behavior.
pub fn assemble_variants_window(
    v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    alt_alleles: ArrayView1<u8>,
    alt_offsets: ArrayView1<i64>,
    ref_src: Option<(ArrayView1<u8>, ArrayView1<i64>)>,
) -> (
    Array1<u8>,
    Array1<i64>,
    Array1<i32>,
    Array1<i32>,
    Option<Array1<u8>>,
    Option<Array1<i64>>,
) {
    let (alt_data, alt_seq_offsets) = gather_alleles(v_idxs, alt_alleles, alt_offsets);
    let start: Array1<i32> = v_idxs.iter().map(|&vi| v_starts[vi as usize]).collect();
    let ilen: Array1<i32> = v_idxs.iter().map(|&vi| ilens[vi as usize]).collect();
    let (ref_data, ref_seq_offsets) = match ref_src {
        Some((bytes, offs)) => {
            let (d, o) = gather_alleles(v_idxs, bytes, offs);
            (Some(d), Some(o))
        }
        None => (None, None),
    };
    (alt_data, alt_seq_offsets, start, ilen, ref_data, ref_seq_offsets)
}

/// Reverse-complement the alleles of mask-selected `(b*p)` rows, in place.
///
/// `byte_data`   contiguous allele bytes (mutated in place)
/// `seq_offsets` per-allele byte boundaries (len n_alleles + 1)
/// `var_offsets` per-(b*p)-row allele boundaries (len n_rows + 1)
/// `to_rc_row`   per-(b*p)-row bool mask (len n_rows)
///
/// Single fused pass: for each masked `(b*p)` row, reverse-complements each of
/// its alleles directly via `reverse::rc_row`. `var_offsets` partition the
/// alleles by row (contiguous, disjoint), so this RCs exactly the alleles the
/// old per-allele-mask delegation did, in the same order — byte-identical —
/// without the intermediate `Vec<bool>` alloc or the second full-allele scan.
pub fn rc_alleles_inplace(
    byte_data: &mut [u8],
    seq_offsets: ndarray::ArrayView1<i64>,
    var_offsets: ndarray::ArrayView1<i64>,
    to_rc_row: ndarray::ArrayView1<bool>,
) {
    for g in 0..to_rc_row.len() {
        if !to_rc_row[g] {
            continue;
        }
        let a0 = var_offsets[g] as usize;
        let a1 = var_offsets[g + 1] as usize;
        for a in a0..a1 {
            let s = seq_offsets[a] as usize;
            let e = seq_offsets[a + 1] as usize;
            crate::reverse::rc_row(&mut byte_data[s..e]);
        }
    }
}

/// Generic compact-keep core. Drops values where `keep[j]` is false and
/// rebuilds row offsets. No `num_traits` dependency — uses `Vec<T>`.
fn compact_keep_impl<T: Copy>(
    values: ArrayView1<T>,
    row_offsets: ArrayView1<i64>,
    keep: ArrayView1<bool>,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = row_offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    let mut n_keep: i64 = 0;
    for i in 0..n_rows {
        for j in row_offsets[i] as usize..row_offsets[i + 1] as usize {
            if keep[j] {
                n_keep += 1;
            }
        }
        new_offsets[i + 1] = n_keep;
    }
    let mut new_v: Vec<T> = Vec::with_capacity(n_keep as usize);
    for j in 0..values.len() {
        if keep[j] {
            new_v.push(values[j]);
        }
    }
    (Array1::from_vec(new_v), new_offsets)
}

/// Compact i32 values (variant indices). Mirrors numba `_compact_keep`.
pub fn compact_keep_i32(
    values: ArrayView1<i32>,
    row_offsets: ArrayView1<i64>,
    keep: ArrayView1<bool>,
) -> (Array1<i32>, Array1<i64>) {
    compact_keep_impl(values, row_offsets, keep)
}

/// Compact f32 values (dosage). Preserves float32 bit-pattern exactly.
pub fn compact_keep_f32(
    values: ArrayView1<f32>,
    row_offsets: ArrayView1<i64>,
    keep: ArrayView1<bool>,
) -> (Array1<f32>, Array1<i64>) {
    compact_keep_impl(values, row_offsets, keep)
}

/// Generic fill-empty-scalar core. Each empty row gets one `fill` element;
/// non-empty rows copy through unchanged. No `num_traits` needed — `from_elem`.
fn fill_empty_scalar_impl<T: Copy>(
    data: ArrayView1<T>,
    offsets: ArrayView1<i64>,
    fill: T,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let ln = offsets[i + 1] - offsets[i];
        new_offsets[i + 1] = new_offsets[i] + if ln > 0 { ln } else { 1 };
    }
    let total = new_offsets[n_rows] as usize;
    // Pre-fill with `fill` so empty-row slots are already correct; copy non-empty.
    let mut new_data = Array1::<T>::from_elem(total, fill);
    for i in 0..n_rows {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        let mut d = new_offsets[i] as usize;
        if e != s {
            for k in s..e {
                new_data[d] = data[k];
                d += 1;
            }
        }
    }
    (new_data, new_offsets)
}

/// Fill-empty-scalar for i32 data (variant start / ilen). Mirrors numba `_fill_empty_scalar`.
pub fn fill_empty_scalar_i32(
    data: ArrayView1<i32>,
    offsets: ArrayView1<i64>,
    fill: i32,
) -> (Array1<i32>, Array1<i64>) {
    fill_empty_scalar_impl(data, offsets, fill)
}

/// Fill-empty-scalar for f32 data (dosage). Mirrors numba `_fill_empty_scalar`.
pub fn fill_empty_scalar_f32(
    data: ArrayView1<f32>,
    offsets: ArrayView1<i64>,
    fill: f32,
) -> (Array1<f32>, Array1<i64>) {
    fill_empty_scalar_impl(data, offsets, fill)
}

/// Generic fill-empty-fixed core. Each empty row gets `inner` copies of `fill`;
/// non-empty rows copy their `n_var * inner` elements through.
fn fill_empty_fixed_impl<T: Copy>(
    data: ArrayView1<T>,
    offsets: ArrayView1<i64>,
    inner: i64,
    fill: T,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let nv = offsets[i + 1] - offsets[i];
        new_offsets[i + 1] = new_offsets[i] + if nv > 0 { nv } else { 1 };
    }
    let total_vars = new_offsets[n_rows] as usize;
    let inner_u = inner as usize;
    let mut new_data = Array1::<T>::from_elem(total_vars * inner_u, fill);
    let mut dptr = 0usize;
    for i in 0..n_rows {
        let vs = offsets[i] as usize;
        let ve = offsets[i + 1] as usize;
        if ve == vs {
            dptr += inner_u; // already filled by from_elem
        } else {
            for k in vs * inner_u..ve * inner_u {
                new_data[dptr] = data[k];
                dptr += 1;
            }
        }
    }
    (new_data, new_offsets)
}

/// Fill-empty-fixed for i32 data (flank_tokens). Mirrors numba `_fill_empty_fixed`.
pub fn fill_empty_fixed_i32(
    data: ArrayView1<i32>,
    offsets: ArrayView1<i64>,
    inner: i64,
    fill: i32,
) -> (Array1<i32>, Array1<i64>) {
    fill_empty_fixed_impl(data, offsets, inner, fill)
}

/// Fill-empty-fixed for f32 data. Mirrors numba `_fill_empty_fixed`.
pub fn fill_empty_fixed_f32(
    data: ArrayView1<f32>,
    offsets: ArrayView1<i64>,
    inner: i64,
    fill: f32,
) -> (Array1<f32>, Array1<i64>) {
    fill_empty_fixed_impl(data, offsets, inner, fill)
}

/// Generic two-level dummy-fill for allele/token bytestrings. Mirrors numba `_fill_empty_seq`.
/// Empty variant-rows receive one dummy allele/token sequence of `dummy` elements.
/// Returns `(new_data, new_var_offsets, new_seq_offsets)`.
fn fill_empty_seq_impl<T: Copy>(
    data: ArrayView1<T>,
    var_offsets: ArrayView1<i64>,
    seq_offsets: ArrayView1<i64>,
    dummy: ArrayView1<T>,
) -> (Array1<T>, Array1<i64>, Array1<i64>) {
    let n_rows = var_offsets.len() - 1;
    let l = dummy.len() as i64;
    let mut new_var = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let nv = var_offsets[i + 1] - var_offsets[i];
        new_var[i + 1] = new_var[i] + if nv > 0 { nv } else { 1 };
    }
    let total_vars = new_var[n_rows] as usize;
    let mut new_seq = Array1::<i64>::zeros(total_vars + 1);
    let mut vptr = 0usize;
    for i in 0..n_rows {
        let vs = var_offsets[i] as usize;
        let ve = var_offsets[i + 1] as usize;
        if ve == vs {
            new_seq[vptr + 1] = new_seq[vptr] + l;
            vptr += 1;
        } else {
            for v in vs..ve {
                let vlen = seq_offsets[v + 1] - seq_offsets[v];
                new_seq[vptr + 1] = new_seq[vptr] + vlen;
                vptr += 1;
            }
        }
    }
    let total = new_seq[total_vars] as usize;
    let mut new_data: Vec<T> = Vec::with_capacity(total);
    for i in 0..n_rows {
        let vs = var_offsets[i] as usize;
        let ve = var_offsets[i + 1] as usize;
        if ve == vs {
            for k in 0..dummy.len() {
                new_data.push(dummy[k]);
            }
        } else {
            for v in vs..ve {
                let bs = seq_offsets[v] as usize;
                let be = seq_offsets[v + 1] as usize;
                for k in bs..be {
                    new_data.push(data[k]);
                }
            }
        }
    }
    (Array1::from_vec(new_data), new_var, new_seq)
}

/// Two-level dummy-fill for allele bytestrings (uint8). Mirrors numba `_fill_empty_seq`.
pub fn fill_empty_seq_u8(
    data: ArrayView1<u8>,
    var_offsets: ArrayView1<i64>,
    seq_offsets: ArrayView1<i64>,
    dummy: ArrayView1<u8>,
) -> (Array1<u8>, Array1<i64>, Array1<i64>) {
    fill_empty_seq_impl(data, var_offsets, seq_offsets, dummy)
}

/// Two-level dummy-fill for token windows (int32). Mirrors numba `_fill_empty_seq`.
pub fn fill_empty_seq_i32(
    data: ArrayView1<i32>,
    var_offsets: ArrayView1<i64>,
    seq_offsets: ArrayView1<i64>,
    dummy: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i64>, Array1<i64>) {
    fill_empty_seq_impl(data, var_offsets, seq_offsets, dummy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_gather_rows_basic() {
        // 2 rows selecting offset groups 1 then 0.
        let goi = arr1(&[1i64, 0]);
        let o_starts = arr1(&[0i64, 2]);
        let o_stops = arr1(&[2i64, 5]);
        let data = arr1(&[10i32, 11, 12, 13, 14]);
        let (v, off) = gather_rows_i32(goi.view(), o_starts.view(), o_stops.view(), data.view());
        assert_eq!(v.to_vec(), vec![12, 13, 14, 10, 11]);
        assert_eq!(off.to_vec(), vec![0, 3, 5]);
    }

    #[test]
    fn test_gather_rows_f32() {
        // Exact binary float32 values must be preserved — no rounding.
        let goi = arr1(&[0i64]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[2i64]);
        let data = arr1(&[0.25f32, 0.75f32]);
        let (v, off) = gather_rows_f32(goi.view(), o_starts.view(), o_stops.view(), data.view());
        assert_eq!(v.to_vec(), vec![0.25f32, 0.75f32]);
        assert_eq!(off.to_vec(), vec![0i64, 2]);
    }

    #[test]
    fn test_gather_alleles_basic() {
        // alleles: v0="AC"(65,67), v1="G"(71). gather [1,0,1].
        let v_idxs = arr1(&[1i32, 0, 1]);
        let bytes = arr1(&[65u8, 67, 71]);
        let offs = arr1(&[0i64, 2, 3]);
        let (data, seq) = gather_alleles(v_idxs.view(), bytes.view(), offs.view());
        assert_eq!(data.to_vec(), vec![71, 65, 67, 71]);
        assert_eq!(seq.to_vec(), vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_compact_keep_i32() {
        // 2 rows: [10, 11 | 12]; keep [T, F, T] → [10 | 12], offsets [0, 1, 2].
        let vals = arr1(&[10i32, 11, 12]);
        let off = arr1(&[0i64, 2, 3]);
        let keep = arr1(&[true, false, true]);
        let (v, o) = compact_keep_i32(vals.view(), off.view(), keep.view());
        assert_eq!(v.to_vec(), vec![10, 12]);
        assert_eq!(o.to_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_compact_keep_f32() {
        // 1 row: [0.25, 0.75, 0.5]; keep [T, F, T] → [0.25, 0.5], offsets [0, 2].
        let vals = arr1(&[0.25f32, 0.75f32, 0.5f32]);
        let off = arr1(&[0i64, 3]);
        let keep = arr1(&[true, false, true]);
        let (v, o) = compact_keep_f32(vals.view(), off.view(), keep.view());
        assert_eq!(v.to_vec(), vec![0.25f32, 0.5f32]);
        assert_eq!(o.to_vec(), vec![0i64, 2]);
    }

    #[test]
    fn test_fill_empty_scalar_i32() {
        // 3 rows: offsets [0,2,2,3] — middle row is empty.
        // Non-empty rows: [10,11] and [20]. Empty row gets one fill (99).
        let data = arr1(&[10i32, 11, 20]);
        let offsets = arr1(&[0i64, 2, 2, 3]);
        let (v, o) = fill_empty_scalar_i32(data.view(), offsets.view(), 99);
        assert_eq!(v.to_vec(), vec![10, 11, 99, 20]);
        assert_eq!(o.to_vec(), vec![0i64, 2, 3, 4]);
    }

    #[test]
    fn test_fill_empty_scalar_f32() {
        // 2 rows: offsets [0,1,1] — second row is empty. fill = -1.0.
        let data = arr1(&[0.5f32]);
        let offsets = arr1(&[0i64, 1, 1]);
        let (v, o) = fill_empty_scalar_f32(data.view(), offsets.view(), -1.0f32);
        assert_eq!(v.to_vec(), vec![0.5f32, -1.0f32]);
        assert_eq!(o.to_vec(), vec![0i64, 1, 2]);
    }

    #[test]
    fn test_fill_empty_fixed_i32() {
        // 3 rows: offsets [0,2,2,3], inner=2 — middle row empty → 2 copies of fill.
        // data = [10,11, 12,13, 20,21] (2 per variant for rows 0 and 2).
        let data = arr1(&[10i32, 11, 12, 13, 20, 21]);
        let offsets = arr1(&[0i64, 2, 2, 3]);
        let (v, o) = fill_empty_fixed_i32(data.view(), offsets.view(), 2, 7);
        // Row 0: 2 vars * 2 inner = 4 elems [10,11,12,13]
        // Row 1: empty → 1 dummy var * 2 inner = 2 elems [7,7]
        // Row 2: 1 var * 2 inner = 2 elems [20,21]
        assert_eq!(v.to_vec(), vec![10, 11, 12, 13, 7, 7, 20, 21]);
        assert_eq!(o.to_vec(), vec![0i64, 2, 3, 4]);
    }

    #[test]
    fn test_fill_empty_fixed_f32() {
        // 2 rows: offsets [0,1,1], inner=3 — second row empty.
        let data = arr1(&[1.0f32, 2.0, 3.0]);
        let offsets = arr1(&[0i64, 1, 1]);
        let (v, o) = fill_empty_fixed_f32(data.view(), offsets.view(), 3, 0.0f32);
        assert_eq!(v.to_vec(), vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(o.to_vec(), vec![0i64, 1, 2]);
    }

    #[test]
    fn test_fill_empty_seq_u8() {
        // 3 rows: var_offsets [0,1,1,2] — middle row (row 1) is empty.
        // Row 0: 1 variant with bytes [65,67] ("AC").
        // Row 1: empty → gets dummy [78] ("N"), length 1.
        // Row 2: 1 variant with bytes [71] ("G").
        // seq_offsets: [0,2,3] (lengths: 2,1).
        let data = arr1(&[65u8, 67, 71]);
        let var_offsets = arr1(&[0i64, 1, 1, 2]);
        let seq_offsets = arr1(&[0i64, 2, 3]);
        let dummy = arr1(&[78u8]); // "N"
        let (nd, nvar, nseq) =
            fill_empty_seq_u8(data.view(), var_offsets.view(), seq_offsets.view(), dummy.view());
        // new_var: row 0 has 1 var, row 1 empty→1 dummy, row 2 has 1 var → [0,1,2,3]
        assert_eq!(nvar.to_vec(), vec![0i64, 1, 2, 3]);
        // new_seq: var0 len=2, dummy len=1, var2 len=1 → [0,2,3,4]
        assert_eq!(nseq.to_vec(), vec![0i64, 2, 3, 4]);
        // new_data: [65,67] (row0), [78] (dummy), [71] (row2)
        assert_eq!(nd.to_vec(), vec![65u8, 67, 78, 71]);
    }

    #[test]
    fn test_fill_empty_seq_i32() {
        // 2 rows: var_offsets [0,0,2] — first row (row 0) is empty.
        // Row 0: empty → gets dummy token [999i32], length 1.
        // Row 1: 2 variants: tokens [10,20] and [30,40,50].
        // seq_offsets: [0,2,5].
        let data = arr1(&[10i32, 20, 30, 40, 50]);
        let var_offsets = arr1(&[0i64, 0, 2]);
        let seq_offsets = arr1(&[0i64, 2, 5]);
        let dummy = arr1(&[999i32]);
        let (nd, nvar, nseq) =
            fill_empty_seq_i32(data.view(), var_offsets.view(), seq_offsets.view(), dummy.view());
        // new_var: row 0 empty→1, row 1 has 2 → [0,1,3]
        assert_eq!(nvar.to_vec(), vec![0i64, 1, 3]);
        // new_seq: dummy len=1, var0 len=2, var1 len=3 → [0,1,3,6]
        assert_eq!(nseq.to_vec(), vec![0i64, 1, 3, 6]);
        // new_data: [999] (dummy), [10,20] (var0), [30,40,50] (var1)
        assert_eq!(nd.to_vec(), vec![999i32, 10, 20, 30, 40, 50]);
    }

    #[test]
    fn rc_alleles_rcs_only_masked_rows() {
        // 2 rows. row0 (masked) has 2 alleles: "AC","G". row1 (unmasked): "TT".
        // seq_offsets delimit alleles: [0,2,3,5]; var_offsets delimit rows: [0,2,3].
        let mut data = b"ACGTT".to_vec();
        let seq_offsets = ndarray::array![0i64, 2, 3, 5];
        let var_offsets = ndarray::array![0i64, 2, 3];
        let to_rc_row = ndarray::array![true, false];
        rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
        // row0: "AC"->"GT", "G"->"C"; row1 "TT" untouched.
        assert_eq!(&data, b"GTCTT");
    }

    #[test]
    fn rc_alleles_all_false_is_noop() {
        let mut data = b"ACG".to_vec();
        let seq_offsets = ndarray::array![0i64, 1, 3];
        let var_offsets = ndarray::array![0i64, 2];
        let to_rc_row = ndarray::array![false];
        rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
        assert_eq!(&data, b"ACG");
    }

    #[test]
    fn rc_alleles_handles_empty_allele_and_n() {
        // 1 masked row, 2 alleles: "" (empty) and "ACN".
        let mut data = b"ACN".to_vec();
        let seq_offsets = ndarray::array![0i64, 0, 3];
        let var_offsets = ndarray::array![0i64, 2];
        let to_rc_row = ndarray::array![true];
        rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
        // "" stays ""; "ACN" -> revcomp -> "NGT".
        assert_eq!(&data, b"NGT");
    }

    /// PR-B3a: with a per-variant REF table, `assemble_variants_window` gathers REF
    /// bytes exactly the way it gathers ALT bytes.
    #[test]
    fn assemble_variants_window_gathers_ref_when_table_supplied() {
        let v_idxs = Array1::from_vec(vec![2i32, 0]);
        let v_starts = Array1::from_vec(vec![10i32, 20, 30]);
        let ilens = Array1::from_vec(vec![0i32, 0, 0]);
        let alt = Array1::from_vec(b"AACCGG".to_vec());
        let alt_off = Array1::from_vec(vec![0i64, 2, 4, 6]);
        let refe = Array1::from_vec(b"TTTGGGCCC".to_vec());
        let ref_off = Array1::from_vec(vec![0i64, 3, 6, 9]);
        let (_a, _ao, _s, _i, rd, ro) = assemble_variants_window(
            v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt.view(),
            alt_off.view(),
            Some((refe.view(), ref_off.view())),
        );
        let rd = rd.expect("ref requested");
        let ro = ro.expect("ref requested");
        // v_idx 2 -> "CCC", v_idx 0 -> "TTT"
        assert_eq!(rd.to_vec(), b"CCCTTT".to_vec());
        assert_eq!(ro.to_vec(), vec![0i64, 3, 6]);
    }

    /// No REF table supplied ⇒ both REF outputs are None (default `var_fields`).
    #[test]
    fn assemble_variants_window_ref_is_none_without_table() {
        let v_idxs = Array1::from_vec(vec![0i32]);
        let v_starts = Array1::from_vec(vec![10i32]);
        let ilens = Array1::from_vec(vec![0i32]);
        let alt = Array1::from_vec(b"A".to_vec());
        let alt_off = Array1::from_vec(vec![0i64, 1]);
        let (_a, _ao, _s, _i, rd, ro) = assemble_variants_window(
            v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt.view(),
            alt_off.view(),
            None,
        );
        assert!(rd.is_none());
        assert!(ro.is_none());
    }
}
