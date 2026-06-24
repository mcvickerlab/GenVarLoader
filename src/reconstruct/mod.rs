//! Single-haplotype reconstruction core (pure ndarray). PyO3 lives in `crate::ffi`.
//!
//! Mirrors `reconstruct_haplotype_from_sparse` in
//! `python/genvarloader/_dataset/_genotypes.py:277-465` statement-by-statement.
use ndarray::{s, ArrayView1, ArrayViewMut1};

/// Reconstruct a single haplotype from reference sequence and variants.
///
/// Single-haplotype inner kernel. Mirror of numba
/// `reconstruct_haplotype_from_sparse` (`_genotypes.py:277-465`).
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
    mut out: ArrayViewMut1<u8>,
    pad_char: u8,
    keep: Option<ArrayView1<bool>>,
    mut annot_v_idxs: Option<ArrayViewMut1<i32>>,
    mut annot_ref_pos: Option<ArrayViewMut1<i32>>,
) {
    let length = out.len() as i64;
    let n_variants = v_idxs.len();

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
        out.slice_mut(s![s..e]).fill(pad_char);
        if let Some(ref mut av) = annot_v_idxs {
            av.slice_mut(s![s..e]).fill(-1);
        }
        if let Some(ref mut ap) = annot_ref_pos {
            ap.slice_mut(s![s..e]).fill(-1);
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

        let variant = v_idxs[v] as usize;
        let v_pos = v_starts[variant] as i64;
        let v_diff = ilens[variant] as i64;
        let ao_s = alt_offsets[variant] as usize;
        let ao_e = alt_offsets[variant + 1] as usize;
        // full allele slice; may be sub-sliced below for shift consumption
        let allele_full = alt_alleles.slice(s![ao_s..ao_e]);
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
        let allele = allele_full.slice(s![allele_start_idx as usize..]);
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
            out.slice_mut(s![os..oe]).assign(&ref_.slice(s![rs..re]));
            if let Some(ref mut av) = annot_v_idxs {
                av.slice_mut(s![os..oe]).fill(-1);
            }
            if let Some(ref mut ap) = annot_ref_pos {
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
            out.slice_mut(s![os..oe])
                .assign(&allele.slice(s![..writable_length as usize]));
            if let Some(ref mut av) = annot_v_idxs {
                av.slice_mut(s![os..oe]).fill(variant as i32);
            }
            if let Some(ref mut ap) = annot_ref_pos {
                ap.slice_mut(s![os..oe]).fill(v_pos as i32);
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
        ref_idx = ref_idx.min(ref_.len() as i64);
        shifted = shift;
    }
    let _ = shifted; // used above, silence unused-assign warning

    // fill rest with reference sequence and right-pad with Ns
    let unfilled_length = length - out_idx;
    if unfilled_length > 0 {
        // fill with reference sequence
        let writable_ref = unfilled_length.min(ref_.len() as i64 - ref_idx);
        let out_end_idx = out_idx + writable_ref;
        let ref_end_idx = ref_idx + writable_ref;
        {
            let os = out_idx as usize;
            let oe = out_end_idx as usize;
            let rs = ref_idx as usize;
            let re = ref_end_idx as usize;
            out.slice_mut(s![os..oe]).assign(&ref_.slice(s![rs..re]));
            if let Some(ref mut av) = annot_v_idxs {
                av.slice_mut(s![os..oe]).fill(-1);
            }
            if let Some(ref mut ap) = annot_ref_pos {
                for (j, pos) in (os..oe).zip(rs..re) {
                    ap[j] = pos as i32;
                }
            }
        }

        // right-pad
        if out_end_idx < length {
            let pe = length as usize;
            let ps = out_end_idx as usize;
            out.slice_mut(s![ps..pe]).fill(pad_char);
            if let Some(ref mut av) = annot_v_idxs {
                av.slice_mut(s![ps..pe]).fill(-1);
            }
            if let Some(ref mut ap) = annot_ref_pos {
                ap.slice_mut(s![ps..pe]).fill(i32::MAX);
            }
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
            1,  // ref_start
            3,  // out_len
            0,  // pad_char
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
        assert_eq!(&ap[..2], &[-1i32, -1], "leading pad annot_ref_pos must be -1");
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
            &[0],        // v_idxs: only variant 0
            &[2],        // v_starts: variant 0 is at pos 2
            &[0],        // ilens: SNP, no length change
            0,           // shift
            &[84u8],     // alt_alleles: T
            &[0i64, 1],  // alt_offsets
            &[65, 67, 71, 84, 65], // A C G T A
            0,           // ref_start
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
            &[2],        // variant 0 at pos 2
            &[2],        // ilen=+2
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
            &[2],        // variant 0 at pos 2
            &[-2],       // ilen=-2
            0,
            &[30u8],     // anchor allele byte
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
            &[1],        // v_pos=1
            &[-3],       // ilen=-3
            0,
            &[99u8],
            &[0i64, 1],
            &[1, 2, 3, 4, 5, 6, 7],
            3,           // ref_start=3
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
            &[0, 1],     // v_idxs: variants 0 then 1
            &[2, 2],     // both at pos=2
            &[0, 0],     // both SNPs
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
            &[3],        // v_pos=3
            &[1],        // ilen=+1
            2,           // shift=2
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
            &[1],        // ilen=+1, allele 2 bytes
            4,           // shift=4
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
    // Case 10: annotated vs non-annotated produce identical out bytes
    // ref = [1,2,3,4,5], ref_start=0, variant at pos=2 (SNP)
    // -------------------------------------------------------------------------
    #[test]
    fn annotated_vs_non_annotated_identical_out() {
        let params = (
            &[0i32][..],   // v_idxs
            &[2i32][..],   // v_starts
            &[0i32][..],   // ilens
            0i64,          // shift
            &[77u8][..],   // alt_alleles
            &[0i64, 1][..],// alt_offsets
            &[1u8,2,3,4,5][..], // ref_
            0i64,          // ref_start
            5usize,        // out_len
            0u8,           // pad_char
        );
        let (out_annot, _, _) = run(
            params.0, params.1, params.2, params.3,
            params.4, params.5, params.6, params.7,
            params.8, params.9, None, true,
        );
        let (out_plain, _, _) = run(
            params.0, params.1, params.2, params.3,
            params.4, params.5, params.6, params.7,
            params.8, params.9, None, false,
        );
        assert_eq!(out_annot, out_plain, "annotated and non-annotated must produce identical out bytes");
    }
}
