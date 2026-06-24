//! Flat variant gather/fill cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{Array1, ArrayView1};

/// Per-row variant-index gather. Mirrors numba `_gather_v_idxs` (and `_ss` via
/// the (2, n) normalized offsets). `o_s = o_starts[goi]`, `o_e = o_stops[goi]`.
pub fn gather_rows(
    geno_offset_idx: ArrayView1<i64>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    geno_v_idxs: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i64>) {
    let n_rows = geno_offset_idx.len();
    let mut out_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        out_offsets[i + 1] = out_offsets[i] + (o_stops[goi] - o_starts[goi]);
    }
    let total = out_offsets[n_rows] as usize;
    let mut v_idxs = Array1::<i32>::zeros(total);
    let mut dst = 0usize;
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        let s = o_starts[goi] as usize;
        let e = o_stops[goi] as usize;
        for k in s..e {
            v_idxs[dst] = geno_v_idxs[k];
            dst += 1;
        }
    }
    (v_idxs, out_offsets)
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
        let (v, off) = gather_rows(goi.view(), o_starts.view(), o_stops.view(), data.view());
        assert_eq!(v.to_vec(), vec![12, 13, 14, 10, 11]);
        assert_eq!(off.to_vec(), vec![0, 3, 5]);
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
}
