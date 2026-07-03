//! SVAR2 two-source variant provider: merge a hap's var_key ⋈ dense channels and
//! decode keys via svar2-codec, feeding the reconstruction kernel with no
//! intermediate variant table. Additive to the SVAR 1.0 global-table path.

use std::borrow::Cow;

use svar2_codec::{decode_key, DecodedKey};

/// Decode one uniform key into `(v_diff, allele)`, resolving long-INS via the LUT
/// arrays. Mirrors genoray's `decode_keyref`.
pub fn decode_alt<'a>(key: u32, lut_bytes: &'a [u8], lut_off: &[i64]) -> (i64, Cow<'a, [u8]>) {
    match decode_key(key) {
        DecodedKey::Inline { alt } => (alt.len() as i64 - 1, Cow::Owned(alt)),
        DecodedKey::PureDel { ilen } => (ilen as i64, Cow::Borrowed(&[][..])),
        DecodedKey::Lookup { row } => {
            let s = lut_off[row as usize] as usize;
            let e = lut_off[row as usize + 1] as usize;
            let alt = &lut_bytes[s..e];
            (alt.len() as i64 - 1, Cow::Borrowed(alt))
        }
    }
}

/// Merge one hap's `var_key` slice with its carried `dense` set-bits into a single
/// position-sorted `(pos, key)` list (stable: var_key before dense on ties, matching
/// genoray's merge). `dense` is region `r`'s `[ds, de)` window; `present` are this hap's
/// LSB-first presence bits over that window.
#[allow(clippy::too_many_arguments)]
pub fn merge_hap(
    vk_pos: &[i32],
    vk_key: &[i32],
    vk_lo: usize,
    vk_hi: usize,
    dense_pos: &[i32],
    dense_key: &[i32],
    ds: usize,
    de: usize,
    present_bit: impl Fn(usize) -> bool, // present_bit(k) for k in 0..(de-ds)
) -> Vec<(u32, u32)> {
    let mut a: Vec<(u32, u32)> = (vk_lo..vk_hi)
        .map(|i| (vk_pos[i] as u32, vk_key[i] as u32))
        .collect();
    for (k, j) in (ds..de).enumerate() {
        if present_bit(k) {
            a.push((dense_pos[j] as u32, dense_key[j] as u32));
        }
    }
    a.sort_by_key(|&(p, _)| p); // stable; var_key pushed first keeps it ahead on ties
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_inline_del_lookup() {
        // Pure DEL of length 2 → v_diff -2, empty allele.
        let del = svar2_codec::encode_pure_del(-2);
        let (d, a) = decode_alt(del, &[], &[0]);
        assert_eq!(d, -2);
        assert!(a.is_empty());

        // Lookup row 0 → "ACGT" from the LUT (v_diff = len-1 = 3).
        let lut = b"ACGT".to_vec();
        let off = vec![0i64, 4];
        let lk = svar2_codec::encode_lookup(0);
        let (d, a) = decode_alt(lk, &lut, &off);
        assert_eq!(d, 3);
        assert_eq!(a.as_ref(), b"ACGT");
    }

    #[test]
    fn test_merge_hap_position_sorted_var_key_before_dense_on_tie() {
        // var_key entries for this hap (positions 10 and 20; 20 ties with a dense entry).
        let vk_pos = [10i32, 20];
        let vk_key = [100i32, 200];

        // dense channel spans multiple haps/regions; this hap's window is [ds, de).
        // dense positions: 15, 20 (ties with vk_pos[1]=20), 30.
        let dense_pos = [15i32, 20, 30];
        let dense_key = [150i32, 250, 300];
        let ds = 0usize;
        let de = 3usize;

        // Present bits (LSB-first) over the window: all three dense entries present.
        let present = [true, true, true];
        let present_bit = |k: usize| present[k];

        let merged = merge_hap(
            &vk_pos,
            &vk_key,
            0,
            vk_pos.len(),
            &dense_pos,
            &dense_key,
            ds,
            de,
            present_bit,
        );

        // Expect position-sorted (pos, key): 10, 15, 20 (var_key first on tie), 20 (dense), 30.
        assert_eq!(
            merged,
            vec![(10, 100), (15, 150), (20, 200), (20, 250), (30, 300)]
        );
    }
}
