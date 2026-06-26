//! Variant-windows / variants flat-buffer assembly cores (pure ndarray).
//! PyO3 lives in `crate::ffi`. Mirrors the Python helpers in
//! `_dataset/_flat_flanks.py` (`tokenize_alleles`, `_slice_flanks`,
//! `_assemble_alt_windows`, `compute_*`) — byte-identical by construction.
use ndarray::{Array1, Array2, ArrayView1};

/// Apply a 256-entry byte->token lookup table. `out[i] = lut[bytes[i]]`.
/// Mirrors numpy `lut[bytes]`. `Tok` is the token dtype (u8 or i32).
pub fn tokenize<Tok: Copy>(bytes: ArrayView1<u8>, lut: ArrayView1<Tok>) -> Array1<Tok> {
    let n = bytes.len();
    let mut out: Vec<Tok> = Vec::with_capacity(n);
    for i in 0..n {
        out.push(lut[bytes[i] as usize]);
    }
    Array1::from_vec(out)
}

/// Derive per-variant (f5, f3) fixed-`flank_len` flanks from a contiguous
/// per-variant window read `[start-L, end+L)`. `f5` = first `L` bytes of each
/// row, `f3` = last `L`. Both returned flat `(n*L,)`, variant-major. Mirrors
/// `_slice_flanks` (`f5 = data[rw_off[:-1,None]+cols]`,
/// `f3 = data[rw_off[1:,None]-L+cols]`).
pub fn slice_flanks(
    data: ArrayView1<u8>,
    rw_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<u8>) {
    let n = rw_off.len() - 1;
    let mut f5: Vec<u8> = Vec::with_capacity(n * flank_len);
    let mut f3: Vec<u8> = Vec::with_capacity(n * flank_len);
    for i in 0..n {
        let s = rw_off[i] as usize;
        let e = rw_off[i + 1] as usize;
        for k in 0..flank_len {
            f5.push(data[s + k]);
        }
        for k in 0..flank_len {
            f3.push(data[e - flank_len + k]);
        }
    }
    (Array1::from_vec(f5), Array1::from_vec(f3))
}

/// Concatenate `flank5 . alt . flank3` per variant into a flat byte buffer.
/// `f5`/`f3` are `(n*flank_len,)` variant-major. Mirrors numba
/// `_assemble_alt_windows`. Returns `(out_bytes, out_offsets)`.
pub fn assemble_alt_window(
    f5: ArrayView1<u8>,
    f3: ArrayView1<u8>,
    alt_data: ArrayView1<u8>,
    alt_seq_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<i64>) {
    let n = alt_seq_off.len() - 1;
    let mut out_off = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let alt_len = alt_seq_off[i + 1] - alt_seq_off[i];
        out_off[i + 1] = out_off[i] + 2 * flank_len as i64 + alt_len;
    }
    let total = out_off[n] as usize;
    let mut out: Vec<u8> = Vec::with_capacity(total);
    for i in 0..n {
        for k in 0..flank_len {
            out.push(f5[i * flank_len + k]);
        }
        for k in alt_seq_off[i] as usize..alt_seq_off[i + 1] as usize {
            out.push(alt_data[k]);
        }
        for k in 0..flank_len {
            out.push(f3[i * flank_len + k]);
        }
    }
    (Array1::from_vec(out), out_off)
}

/// Fetch the per-variant reference window `[start-L, end+L)` into one flat
/// buffer, with `ends = starts - min(ilen, 0) + 1`. Returns `(data, rw_off)`
/// where `rw_off` are per-variant byte boundaries (len `n+1`). Reuses
/// `reference::get_reference`'s padded core (absolute-coordinate OOB padding).
/// Mirrors `reference.fetch(v_contigs, starts-L, ends+L)`.
pub fn fetch_windows(
    v_contigs: ArrayView1<i32>,
    starts_v: ArrayView1<i32>,
    ilens_v: ArrayView1<i32>,
    flank_len: i64,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> (Array1<u8>, Array1<i64>) {
    let n = starts_v.len();
    let mut regions = Array2::<i32>::zeros((n, 3));
    let mut rw_off = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let start = starts_v[i] as i64;
        let ilen = ilens_v[i] as i64;
        let end = start - ilen.min(0) + 1;
        let rstart = start - flank_len;
        let rend = end + flank_len;
        regions[[i, 0]] = v_contigs[i];
        regions[[i, 1]] = rstart as i32;
        regions[[i, 2]] = rend as i32;
        rw_off[i + 1] = rw_off[i] + (rend - rstart);
    }
    let data = crate::reference::get_reference(
        regions.view(),
        rw_off.view(),
        reference,
        ref_offsets,
        pad_char,
        false, // serial: disjoint output already; this is per-variant fanout
    );
    (data, rw_off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_tokenize_u8() {
        // lut maps byte 65('A')->0, 67('C')->1, everything else->9 (unknown).
        let mut lut = vec![9u8; 256];
        lut[65] = 0;
        lut[67] = 1;
        let lut = Array1::from_vec(lut);
        let bytes = arr1(&[65u8, 67, 78]); // A, C, N(unknown)
        let out = tokenize(bytes.view(), lut.view());
        assert_eq!(out.to_vec(), vec![0u8, 1, 9]);
    }

    #[test]
    fn test_tokenize_i32() {
        // i32 tokens (alphabet larger than 255 forces i32 in Python).
        let mut lut = vec![999i32; 256];
        lut[71] = 300; // 'G' -> 300
        let lut = Array1::from_vec(lut);
        let bytes = arr1(&[71u8, 84]); // G, T(unknown)
        let out = tokenize(bytes.view(), lut.view());
        assert_eq!(out.to_vec(), vec![300i32, 999]);
    }

    #[test]
    fn test_slice_flanks() {
        // 2 variants, L=2. var0 window=[1,2,3,4,5] (len 5), var1=[6,7,8,9] (len 4).
        // rw_off = [0, 5, 9].
        let data = arr1(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9]);
        let rw_off = arr1(&[0i64, 5, 9]);
        let (f5, f3) = slice_flanks(data.view(), rw_off.view(), 2);
        // f5: first 2 of each = [1,2 | 6,7]; f3: last 2 of each = [4,5 | 8,9]
        assert_eq!(f5.to_vec(), vec![1u8, 2, 6, 7]);
        assert_eq!(f3.to_vec(), vec![4u8, 5, 8, 9]);
    }

    #[test]
    fn test_assemble_alt_window() {
        // L=1. f5=[10|20], f3=[11|21]. alt: var0="A"(65), var1="CG"(67,71).
        let f5 = arr1(&[10u8, 20]);
        let f3 = arr1(&[11u8, 21]);
        let alt_data = arr1(&[65u8, 67, 71]);
        let alt_seq_off = arr1(&[0i64, 1, 3]);
        let (out, off) = assemble_alt_window(
            f5.view(),
            f3.view(),
            alt_data.view(),
            alt_seq_off.view(),
            1,
        );
        // var0: 10, 65, 11  (2*1 + 1 = 3 bytes)
        // var1: 20, 67,71, 21  (2*1 + 2 = 4 bytes)
        assert_eq!(out.to_vec(), vec![10u8, 65, 11, 20, 67, 71, 21]);
        assert_eq!(off.to_vec(), vec![0i64, 3, 7]);
    }

    #[test]
    fn test_fetch_windows() {
        use ndarray::Array1 as A1;
        // Single contig reference: bytes 0..20.
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        // 1 variant, contig 0, start=5, ilen=0 (SNP) → end = 5 - 0 + 1 = 6.
        // L=2 → read [start-L, end+L) = [3, 8) → bytes [3,4,5,6,7].
        let v_contigs = arr1(&[0i32]);
        let starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let (data, rw_off) = fetch_windows(
            v_contigs.view(),
            starts.view(),
            ilens.view(),
            2,
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        assert_eq!(data.to_vec(), vec![3u8, 4, 5, 6, 7]);
        assert_eq!(rw_off.to_vec(), vec![0i64, 5]);
    }

    #[test]
    fn test_fetch_windows_deletion_widens() {
        use ndarray::Array1 as A1;
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        // ilen=-2 (2bp deletion) → end = start - (-2) + 1 = start + 3.
        // start=5, L=1 → read [4, 9) → bytes [4,5,6,7,8] (len 5).
        let v_contigs = arr1(&[0i32]);
        let starts = arr1(&[5i32]);
        let ilens = arr1(&[-2i32]);
        let (data, rw_off) = fetch_windows(
            v_contigs.view(),
            starts.view(),
            ilens.view(),
            1,
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        assert_eq!(data.to_vec(), vec![4u8, 5, 6, 7, 8]);
        assert_eq!(rw_off.to_vec(), vec![0i64, 5]);
    }
}
