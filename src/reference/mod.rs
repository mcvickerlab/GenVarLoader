//! Reference sequence assembly cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{ArrayView1, ArrayViewMut1};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1};

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
}
