//! Reference sequence assembly cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1};
use rayon::prelude::*;

/// Copy `arr[start:stop]` into `out`, padding with `pad_val` where the slice
/// runs past `[0, arr.len())`. Mirrors numba `padded_slice`
/// (`_dataset/_utils.py`). `out.len()` MUST equal `stop - start` for the
/// in-bounds case (the caller guarantees this via out_offsets).
pub fn padded_slice(
    arr: ArrayView1<u8>,
    start: i64,
    stop: i64,
    pad_val: u8,
    mut out: ArrayViewMut1<u8>,
) {
    if start >= stop {
        return;
    }
    if stop < 0 {
        out.fill(pad_val);
        return;
    }
    let len = arr.len() as i64;
    let pad_left = (-start).max(0);
    let pad_right = (stop - len).max(0);
    if pad_left == 0 && pad_right == 0 {
        // out[:] = arr[start:stop]
        out.assign(&arr.slice(ndarray::s![start as usize..stop as usize]));
        return;
    }
    let out_len = out.len() as i64;
    if pad_left > 0 && pad_right > 0 {
        let out_stop = out_len - pad_right;
        out.slice_mut(ndarray::s![..pad_left as usize]).fill(pad_val);
        out.slice_mut(ndarray::s![pad_left as usize..out_stop as usize])
            .assign(&arr);
        out.slice_mut(ndarray::s![out_stop as usize..]).fill(pad_val);
    } else if pad_left > 0 {
        // out[:pad_left] = pad; out[pad_left:] = arr[:stop]
        out.slice_mut(ndarray::s![..pad_left as usize]).fill(pad_val);
        out.slice_mut(ndarray::s![pad_left as usize..])
            .assign(&arr.slice(ndarray::s![..stop as usize]));
    } else {
        // pad_right > 0: out[:out_stop] = arr[start:]; out[out_stop:] = pad
        let out_stop = out_len - pad_right;
        out.slice_mut(ndarray::s![..out_stop as usize])
            .assign(&arr.slice(ndarray::s![start as usize..]));
        out.slice_mut(ndarray::s![out_stop as usize..]).fill(pad_val);
    }
}

/// Fetch padded reference rows for each region into one flat buffer.
/// `regions[i] = (contig_idx, start, end)`. Mirrors numba
/// `_get_reference_par/_ser` + `_get_reference_row`. Scheduling (rayon vs
/// serial) does not affect output — out-slices are disjoint.
pub fn get_reference(
    regions: ArrayView2<i32>,
    out_offsets: ArrayView1<i64>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
    to_rc: Option<ArrayView1<bool>>,
) -> Array1<u8> {
    let total = out_offsets[out_offsets.len() - 1] as usize;
    let mut out = Array1::<u8>::zeros(total);
    let n = regions.nrows();

    // Build disjoint mutable row slices so we can fill each region independently.
    let row = |i: usize, dst: &mut [u8]| {
        let c_idx = regions[[i, 0]] as usize;
        let start = regions[[i, 1]] as i64;
        let end = regions[[i, 2]] as i64;
        let c_s = ref_offsets[c_idx] as usize;
        let c_e = ref_offsets[c_idx + 1] as usize;
        let contig = reference.slice(ndarray::s![c_s..c_e]);
        let mut dst_view = ndarray::ArrayViewMut1::from(dst);
        padded_slice(contig, start, end, pad_char, dst_view.view_mut());
    };

    // Partition `out` into per-region chunks by out_offsets, then fill.
    let bounds: Vec<(usize, usize)> = (0..n)
        .map(|i| (out_offsets[i] as usize, out_offsets[i + 1] as usize))
        .collect();
    let out_slice = out.as_slice_mut().unwrap();
    if parallel {
        // split_at_mut chain over sorted disjoint bounds
        let mut chunks: Vec<&mut [u8]> = Vec::with_capacity(n);
        let mut rest = out_slice;
        let mut cursor = 0usize;
        for &(s, e) in &bounds {
            let (_, tail) = rest.split_at_mut(s - cursor);
            let (mid, tail2) = tail.split_at_mut(e - s);
            chunks.push(mid);
            rest = tail2;
            cursor = e;
        }
        chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(i, dst)| row(i, dst));
    } else {
        for (i, &(s, e)) in bounds.iter().enumerate() {
            row(i, &mut out_slice[s..e]);
        }
    }
    if let Some(to_rc) = to_rc {
        debug_assert_eq!(
            to_rc.len(),
            out_offsets.len() - 1,
            "to_rc mask length must equal number of output rows (offsets.len() - 1)"
        );
        crate::reverse::rc_flat_rows_inplace(
            out.as_slice_mut().unwrap(),
            out_offsets,
            to_rc,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1};

    fn run(arr: &[u8], start: i64, stop: i64, pad: u8) -> Vec<u8> {
        let a = arr1(arr);
        let mut out = Array1::<u8>::zeros((stop - start).max(0) as usize);
        padded_slice(a.view(), start, stop, pad, out.view_mut());
        out.to_vec()
    }

    #[test]
    fn in_bounds() {
        assert_eq!(run(&[1, 2, 3, 4, 5], 1, 4, 0), vec![2, 3, 4]);
    }
    #[test]
    fn pad_left_only() {
        assert_eq!(run(&[1, 2, 3], -2, 2, 9), vec![9, 9, 1, 2]);
    }
    #[test]
    fn pad_right_only() {
        assert_eq!(run(&[1, 2, 3], 1, 5, 9), vec![2, 3, 9, 9]);
    }
    #[test]
    fn pad_both() {
        assert_eq!(run(&[1, 2], -1, 3, 9), vec![9, 1, 2, 9]);
    }
    #[test]
    fn empty_when_start_ge_stop() {
        assert_eq!(run(&[1, 2, 3], 2, 2, 9), Vec::<u8>::new());
    }
    #[test]
    fn all_pad_when_stop_negative() {
        let a = arr1(&[1u8, 2, 3]);
        let mut out = Array1::<u8>::zeros(3);
        padded_slice(a.view(), -5, -1, 7, out.view_mut());
        // stop < 0 → numba returns early after filling pad_val on the whole out
        assert_eq!(out.to_vec(), vec![7, 7, 7]);
    }

    // Helper: run get_reference with a flat reference + single contig
    fn run_get_reference(
        reference: &[u8],
        regions: &[[i32; 3]],
        pad: u8,
        parallel: bool,
    ) -> Vec<u8> {
        let n_contigs = 1usize;
        let ref_arr = Array1::from_vec(reference.to_vec());
        let ref_offsets = Array1::from_vec(vec![0i64, reference.len() as i64]);
        let lengths: Vec<usize> = regions.iter().map(|r| (r[2] - r[1]).max(0) as usize).collect();
        let out_offsets: Vec<i64> = std::iter::once(0i64)
            .chain(lengths.iter().scan(0i64, |acc, &l| {
                *acc += l as i64;
                Some(*acc)
            }))
            .collect();
        let out_offsets_arr = Array1::from_vec(out_offsets);
        let n = regions.len();
        let flat: Vec<i32> = regions.iter().flat_map(|r| r.iter().copied()).collect();
        let regions_arr = ndarray::Array2::from_shape_vec((n, 3), flat).unwrap();
        get_reference(
            regions_arr.view(),
            out_offsets_arr.view(),
            ref_arr.view(),
            ref_offsets.view(),
            pad,
            parallel,
            None,
        )
        .to_vec()
    }

    #[test]
    fn get_reference_fully_in_bounds() {
        // region [1,4) on contig [10,20,30,40,50] → [20,30,40]
        let result = run_get_reference(&[10, 20, 30, 40, 50], &[[0, 1, 4]], 0, false);
        assert_eq!(result, vec![20, 30, 40]);
    }

    #[test]
    fn get_reference_straddling_left_edge() {
        // region [-2,2) on contig [1,2,3] → pad pad 1 2
        let result = run_get_reference(&[1, 2, 3], &[[0, -2, 2]], 9, false);
        assert_eq!(result, vec![9, 9, 1, 2]);
    }

    #[test]
    fn get_reference_straddling_right_edge() {
        // region [1,5) on contig [1,2,3] → 2 3 pad pad
        let result = run_get_reference(&[1, 2, 3], &[[0, 1, 5]], 9, false);
        assert_eq!(result, vec![2, 3, 9, 9]);
    }

    #[test]
    fn get_reference_two_contigs() {
        // reference = [10,20] | [30,40,50]; ref_offsets = [0,2,5]
        // region 0: contig 0, [0,2) → [10,20]
        // region 1: contig 1, [1,3) → [40,50]
        let reference = Array1::from_vec(vec![10u8, 20, 30, 40, 50]);
        let ref_offsets = Array1::from_vec(vec![0i64, 2, 5]);
        let regions = arr2(&[[0i32, 0, 2], [1, 1, 3]]);
        let out_offsets = Array1::from_vec(vec![0i64, 2, 4]);
        let result = get_reference(
            regions.view(),
            out_offsets.view(),
            reference.view(),
            ref_offsets.view(),
            0,
            false,
            None,
        );
        assert_eq!(result.to_vec(), vec![10, 20, 40, 50]);
    }

    #[test]
    fn get_reference_parallel_matches_serial() {
        let reference: Vec<u8> = (0..30).collect();
        let regions_data = vec![[0i32, -1, 4], [0, 5, 10], [0, 25, 32]];
        let serial = run_get_reference(&reference, &regions_data, 255, false);
        let parallel = run_get_reference(&reference, &regions_data, 255, true);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn get_reference_applies_rc_when_masked() {
        // contig "ACGTAA"; region [0,3) -> forward "ACG" -> revcomp "CGT" (non-palindrome)
        let reference = ndarray::array![b'A', b'C', b'G', b'T', b'A', b'A'];
        let ref_offsets = ndarray::array![0i64, 6];
        let regions = ndarray::array![[0i32, 0, 3]];
        let out_offsets = ndarray::array![0i64, 3];
        let to_rc = ndarray::array![true];
        let out = get_reference(
            regions.view(),
            out_offsets.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
            false,
            Some(to_rc.view()),
        );
        assert_eq!(out.to_vec(), b"CGT".to_vec());
    }
}
