//! Genotype assembly/selection cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Per-(query, hap) reference-length diffs. Mirrors the numba
/// `get_diffs_sparse` exactly. `o_starts`/`o_stops` are the two rows of the
/// normalized (2, n) offset array: `o_s = o_starts[o_idx]`, `o_e = o_stops[o_idx]`.
/// Length sums stay far within i32 for real variants; accumulate in i64 and
/// truncate on store to mirror numpy's `int32`-slot assignment.
#[allow(clippy::too_many_arguments)]
pub fn get_diffs_sparse(
    geno_offset_idx: ArrayView2<i64>,
    geno_v_idxs: ArrayView1<i32>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    ilens: ArrayView1<i32>,
    keep: Option<ArrayView1<bool>>,
    keep_offsets: Option<ArrayView1<i64>>,
    q_starts: Option<ArrayView1<i32>>,
    q_ends: Option<ArrayView1<i32>>,
    v_starts: Option<ArrayView1<i32>>,
) -> Array2<i32> {
    let (n_queries, ploidy) = geno_offset_idx.dim();
    let mut diffs = Array2::<i32>::zeros((n_queries, ploidy));
    let has_query = q_starts.is_some() && q_ends.is_some() && v_starts.is_some();
    let has_keep = keep.is_some() && keep_offsets.is_some();

    for query in 0..n_queries {
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let o_s = o_starts[o_idx] as usize;
            let o_e = o_stops[o_idx] as usize;
            let n_variants = o_e - o_s;

            if n_variants == 0 {
                diffs[[query, hap]] = 0;
            } else if has_query {
                let qs = q_starts.unwrap();
                let qe = q_ends.unwrap();
                let vs = v_starts.unwrap();
                let q_start = qs[query] as i64;
                let q_end = qe[query] as i64;
                let mut ref_idx = q_start;
                let mut acc: i64 = 0;
                for v in o_s..o_e {
                    if has_keep {
                        let kp = keep.unwrap();
                        let ko = keep_offsets.unwrap();
                        let k_s = ko[query * ploidy + hap] as usize;
                        if !kp[k_s + (v - o_s)] {
                            continue;
                        }
                    }
                    let v_idx = geno_v_idxs[v] as usize;
                    let v_start = vs[v_idx] as i64;
                    let mut v_ilen = ilens[v_idx] as i64;
                    let v_end = v_start - v_ilen.min(0) + 1;
                    if v_end <= q_start {
                        continue;
                    }
                    if v_start >= q_end {
                        break;
                    }
                    if v_start >= q_start && v_start < ref_idx {
                        continue;
                    }
                    ref_idx = ref_idx.max(v_end);
                    if v_ilen < 0 {
                        v_ilen += (q_start - v_start - 1).max(0);
                    }
                    v_ilen += (v_end - q_end).max(0);
                    acc += v_ilen;
                }
                diffs[[query, hap]] = acc as i32;
            } else if has_keep {
                let kp = keep.unwrap();
                let ko = keep_offsets.unwrap();
                let k_s = ko[query * ploidy + hap] as usize;
                let mut sum: i64 = 0;
                for (j, v) in (o_s..o_e).enumerate() {
                    if kp[k_s + j] {
                        sum += ilens[geno_v_idxs[v] as usize] as i64;
                    }
                }
                diffs[[query, hap]] = sum as i32;
            } else {
                let mut sum: i64 = 0;
                for v in o_s..o_e {
                    sum += ilens[geno_v_idxs[v] as usize] as i64;
                }
                diffs[[query, hap]] = sum as i32;
            }
        }
    }
    diffs
}

/// Keep-mask for variants fully contained in each query interval. Mirrors the
/// numba `choose_exonic_variants` + inner `_choose_exonic_variants`. Returns
/// `(keep, keep_offsets)` where keep_offsets is the per-group prefix sum of
/// group sizes (len n_groups + 1).
#[allow(clippy::too_many_arguments)]
pub fn choose_exonic_variants(
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
    geno_offset_idx: ArrayView2<i64>,
    geno_v_idxs: ArrayView1<i32>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
) -> (Array1<bool>, Array1<i64>) {
    let (n_regions, ploidy) = geno_offset_idx.dim();

    // keep_offsets = prefix sum of per-group lengths (numba uses lengths.cumsum()).
    let mut keep_offsets = Array1::<i64>::zeros(n_regions * ploidy + 1);
    let mut acc: i64 = 0;
    for query in 0..n_regions {
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let len = (o_stops[o_idx] - o_starts[o_idx]).max(0);
            acc += len;
            keep_offsets[query * ploidy + hap + 1] = acc;
        }
    }

    let n_variants = keep_offsets[n_regions * ploidy] as usize;
    let mut keep = Array1::<bool>::default(n_variants);

    for query in 0..n_regions {
        let ref_start = starts[query] as i64;
        let ref_end = ends[query] as i64;
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let o_s = o_starts[o_idx] as usize;
            let o_e = o_stops[o_idx] as usize;
            let k_s = keep_offsets[query * ploidy + hap] as usize;
            for (j, v) in (o_s..o_e).enumerate() {
                let v_idx = geno_v_idxs[v] as usize;
                let v_pos = v_starts[v_idx] as i64;
                let v_ref_end = v_pos - (ilens[v_idx] as i64).min(0) + 1;
                keep[k_s + j] = v_pos >= ref_start && v_ref_end <= ref_end;
            }
        }
    }
    (keep, keep_offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_plain_sum() {
        // 1 query, ploidy 1, two variants with ilens [-2, 3] → sum 1.
        let goi = arr2(&[[0i64]]);
        let v_idxs = arr1(&[0i32, 1]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[2i64]);
        let ilens = arr1(&[-2i32, 3]);
        let d = get_diffs_sparse(
            goi.view(), v_idxs.view(), o_starts.view(), o_stops.view(),
            ilens.view(), None, None, None, None, None,
        );
        assert_eq!(d[[0, 0]], 1);
    }

    #[test]
    fn test_empty_group_is_zero() {
        let goi = arr2(&[[0i64]]);
        let v_idxs: ndarray::Array1<i32> = ndarray::Array1::from(vec![]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[0i64]); // empty slice
        let ilens: ndarray::Array1<i32> = ndarray::Array1::from(vec![]);
        let d = get_diffs_sparse(
            goi.view(), v_idxs.view(), o_starts.view(), o_stops.view(),
            ilens.view(), None, None, None, None, None,
        );
        assert_eq!(d[[0, 0]], 0);
    }

    #[test]
    fn test_exonic_contained_only() {
        // region [10, 20). variants at pos 12 (ilen 0 -> end 13, kept) and
        // pos 19 (ilen 0 -> end 20, kept), pos 19 with ilen -2 -> end 22 (dropped).
        let goi = arr2(&[[0i64]]);
        let v_idxs = arr1(&[0i32, 1, 2]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[3i64]);
        let v_starts = arr1(&[12i32, 19, 19]);
        let ilens = arr1(&[0i32, 0, -2]);
        let (keep, koff) = choose_exonic_variants(
            arr1(&[10i32]).view(), arr1(&[20i32]).view(), goi.view(),
            v_idxs.view(), o_starts.view(), o_stops.view(),
            v_starts.view(), ilens.view(),
        );
        assert_eq!(keep.to_vec(), vec![true, true, false]);
        assert_eq!(koff.to_vec(), vec![0, 3]);
    }
}
