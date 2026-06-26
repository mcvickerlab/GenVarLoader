//! In-place reverse / reverse-complement of masked rows in a flat (data, offsets)
//! buffer. Used by the read-path kernels to emit negative-strand output already
//! reverse-complemented, replacing the Python RC post-pass on the rust backend.

use ndarray::ArrayView1;

/// ACGT<->TGCA complement, identity for every other byte. Mirrors
/// `bytes.maketrans(b"ACGT", b"TGCA")` (python/genvarloader/_ragged.py).
pub const COMP: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t[b'A' as usize] = b'T';
    t[b'T' as usize] = b'A';
    t[b'C' as usize] = b'G';
    t[b'G' as usize] = b'C';
    t
};

/// Reverse element order within each masked row (no complement). Generic over
/// element width so it serves f32 tracks and i32/i64 annotation arrays.
pub fn reverse_flat_rows_inplace<T: Copy>(
    data: &mut [T],
    offsets: ArrayView1<i64>,
    to_rc: ArrayView1<bool>,
) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        data[s..e].reverse();
    }
}

/// Reverse AND complement bytes within each masked row via `COMP`.
pub fn rc_flat_rows_inplace(
    data: &mut [u8],
    offsets: ArrayView1<i64>,
    to_rc: ArrayView1<bool>,
) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        let row = &mut data[s..e];
        row.reverse();
        for b in row.iter_mut() {
            *b = COMP[*b as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn comp_lut_matches_maketrans() {
        // identity except ACGT<->TGCA uppercase
        assert_eq!(COMP[b'A' as usize], b'T');
        assert_eq!(COMP[b'T' as usize], b'A');
        assert_eq!(COMP[b'C' as usize], b'G');
        assert_eq!(COMP[b'G' as usize], b'C');
        assert_eq!(COMP[b'N' as usize], b'N');
        assert_eq!(COMP[b'a' as usize], b'a'); // lowercase pass-through
        assert_eq!(COMP[b'c' as usize], b'c');
        assert_eq!(COMP[b'R' as usize], b'R'); // IUPAC pass-through
        assert_eq!(COMP[0u8 as usize], 0u8);
    }

    #[test]
    fn rc_reverses_and_complements_masked_rows_only() {
        // two rows: "ACGT" (rc -> "ACGT") and "AACG" (not rc)
        let mut data = b"ACGTAACG".to_vec();
        let offsets = array![0i64, 4, 8];
        let to_rc = array![true, false];
        rc_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(&data[0..4], b"ACGT"); // revcomp of ACGT is ACGT
        assert_eq!(&data[4..8], b"AACG"); // untouched
    }

    #[test]
    fn rc_handles_odd_length_and_n() {
        let mut data = b"ACN".to_vec(); // revcomp -> "NGT"
        let offsets = array![0i64, 3];
        let to_rc = array![true];
        rc_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(&data, b"NGT");
    }

    #[test]
    fn reverse_only_no_complement_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 9.0];
        let offsets = array![0i64, 3, 4];
        let to_rc = array![true, false];
        reverse_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(data, vec![3.0, 2.0, 1.0, 9.0]);
    }

    #[test]
    fn reverse_only_i32_for_annot_arrays() {
        let mut data = vec![10i32, 11, 12];
        let offsets = array![0i64, 3];
        let to_rc = array![true];
        reverse_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(data, vec![12, 11, 10]);
    }

    #[test]
    fn empty_row_and_all_false_are_noops() {
        let mut data = b"AC".to_vec();
        let offsets = array![0i64, 0, 2]; // first row empty
        rc_flat_rows_inplace(&mut data, offsets.view(), array![true, false].view());
        assert_eq!(&data, b"AC");
    }
}
