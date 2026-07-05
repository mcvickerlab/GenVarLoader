//! SVAR2 two-source variant provider: merge a hap's var_key ⋈ dense channels and
//! decode keys via svar2-codec, feeding the reconstruction kernel with no
//! intermediate variant table. Additive to the SVAR 1.0 global-table path.

use std::borrow::Cow;

use genoray_core::query::BatchResultSplit;
use ndarray::{Array2, ArrayView2};
use svar2_codec::{decode_key, DecodedKey};

pub mod store;

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

/// Per-hap applied-ilen diff for the two-source path, mirroring
/// `genotypes::get_diffs_sparse`'s q_start/q_end-clipped branch. Used to size the fused
/// SVAR2 reconstruct/track outputs. Serial (n is tiny; the fused callers already parallelize
/// the heavy reconstruct pass).
#[allow(clippy::too_many_arguments)]
pub fn hap_diffs_svar2(
    regions: ArrayView2<i32>, // (n_q, 3)
    ploidy: usize,
    vk_pos: &[i32],
    vk_key: &[i32],
    vk_off: &[i64], // (n_work+1)
    dense_pos: &[i32],
    dense_key: &[i32],
    dense_range: ArrayView2<i32>, // (n_q, 2)
    dense_present: &[u8],
    dense_present_off: &[i64], // (n_work+1) BIT offsets
    lut_bytes: &[u8],
    lut_off: &[i64],
) -> Array2<i32> {
    let n_q = regions.nrows();
    let mut diffs = Array2::<i32>::zeros((n_q, ploidy));
    for k in 0..(n_q * ploidy) {
        let query = k / ploidy;
        let hap = k % ploidy;
        let vk_lo = vk_off[k] as usize;
        let vk_hi = vk_off[k + 1] as usize;
        let ds = dense_range[[query, 0]] as usize;
        let de = dense_range[[query, 1]] as usize;
        let base_bit = dense_present_off[k] as usize;
        let present_bit = |j: usize| -> bool {
            let bit = base_bit + j;
            (dense_present[bit / 8] >> (bit % 8)) & 1 == 1
        };
        let merged = merge_hap(
            vk_pos,
            vk_key,
            vk_lo,
            vk_hi,
            dense_pos,
            dense_key,
            ds,
            de,
            present_bit,
        );
        if merged.is_empty() {
            continue;
        }
        let q_start = regions[[query, 1]] as i64;
        let q_end = regions[[query, 2]] as i64;
        let mut ref_idx = q_start;
        let mut acc: i64 = 0;
        for &(pos, key) in &merged {
            let v_start = pos as i64;
            let (mut v_ilen, _allele) = decode_alt(key, lut_bytes, lut_off);
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
    }
    diffs
}

/// The flat, single-dense-channel layout consumed by [`hap_diffs_svar2`] /
/// `reconstruct::reconstruct_haplotypes_from_svar2` — see [`split_to_flat`].
pub struct FlatChannels {
    pub vk_pos: Vec<i32>,
    pub vk_key: Vec<i32>,
    pub vk_off: Vec<i64>,
    pub dense_pos: Vec<i32>,
    pub dense_key: Vec<i32>,
    /// Flat, length `n_q*2`; view as `(n_q, 2)` at the call site.
    pub dense_range: Vec<i32>,
    pub dense_present: Vec<u8>,
    pub dense_present_off: Vec<i64>,
}

/// Marshal a read-bound [`BatchResultSplit`] (genoray's per-class-split dense
/// channels) into the flat single-dense-channel layout that the already-validated
/// [`hap_diffs_svar2`] / `reconstruct::reconstruct_haplotypes_from_svar2` kernels
/// consume — so read-bound gather can reuse those kernels unchanged instead of
/// duplicating the merge/decode logic for a three-source layout.
///
/// Per query `q` the combined dense window is `dense_snp[q] ++ dense_indel[q]`
/// (SNP entries first — matches genoray's `merge_keys(vec![vk, dense_snp,
/// dense_indel])` tie order, where the dense side is `dense_snp` before
/// `dense_indel`). Per hap the combined presence bits are `snp_bits ++
/// indel_bits` over that combined window, LSB-first packed into one bitstream,
/// so `dense_present_off`/`dense_range` line up with `dense_pos`/`dense_key`
/// exactly like the union path's single dense channel.
pub fn split_to_flat(br: &BatchResultSplit) -> FlatChannels {
    let ploidy = br.ploidy;
    let n_q = br.n_regions; // n_samples == 1 for read-bound gather
    let h_count = n_q * ploidy;

    let vk_pos: Vec<i32> = br.vk.iter().map(|k| k.position as i32).collect();
    let vk_key: Vec<i32> = br.vk.iter().map(|k| k.key as i32).collect();
    let vk_off: Vec<i64> = br.vk_off.iter().map(|&o| o as i64).collect();

    let mut dense_pos: Vec<i32> = Vec::new();
    let mut dense_key: Vec<i32> = Vec::new();
    let mut dense_range: Vec<i32> = Vec::with_capacity(n_q * 2);
    for q in 0..n_q {
        let base = dense_pos.len() as i32;
        let (ss, se) = br.dense_snp_range[q];
        for j in ss..se {
            dense_pos.push(br.dense_snp[j].position as i32);
            dense_key.push(br.dense_snp[j].key as i32);
        }
        let (is_, ie) = br.dense_indel_range[q];
        for j in is_..ie {
            dense_pos.push(br.dense_indel[j].position as i32);
            dense_key.push(br.dense_indel[j].key as i32);
        }
        dense_range.push(base);
        dense_range.push(dense_pos.len() as i32);
    }

    let mut dense_present: Vec<u8> = Vec::new();
    let mut dense_present_off: Vec<i64> = Vec::with_capacity(h_count + 1);
    let mut bit_acc: usize = 0;
    dense_present_off.push(0);
    for h in 0..h_count {
        let q = h / ploidy;
        let (ss, se) = br.dense_snp_range[q];
        let (is_, ie) = br.dense_indel_range[q];
        let snp_base = br.dense_snp_present_off[h];
        for k in 0..(se - ss) {
            if genoray_core::bits_get_bit(&br.dense_snp_present, snp_base + k) {
                let byte = bit_acc / 8;
                if dense_present.len() <= byte {
                    dense_present.resize(byte + 1, 0);
                }
                dense_present[byte] |= 1 << (bit_acc % 8);
            }
            bit_acc += 1;
        }
        let indel_base = br.dense_indel_present_off[h];
        for k in 0..(ie - is_) {
            if genoray_core::bits_get_bit(&br.dense_indel_present, indel_base + k) {
                let byte = bit_acc / 8;
                if dense_present.len() <= byte {
                    dense_present.resize(byte + 1, 0);
                }
                dense_present[byte] |= 1 << (bit_acc % 8);
            }
            bit_acc += 1;
        }
        dense_present_off.push(bit_acc as i64);
    }
    // The reused kernels read `dense_present[bit/8]` for EVERY window entry of
    // every hap, so the buffer must always be ceil(total_bits/8) bytes — even
    // when the last hap's window bits are all zero (the in-loop grow-on-set
    // above only extends up to the highest SET bit, leaving a trailing all-zero
    // byte unallocated → OOB panic downstream). genoray byte-sizes its presence
    // arrays via div_ceil unconditionally; match that here.
    dense_present.resize(bit_acc.div_ceil(8), 0);

    FlatChannels {
        vk_pos,
        vk_key,
        vk_off,
        dense_pos,
        dense_key,
        dense_range,
        dense_present,
        dense_present_off,
    }
}

/// Per-hap decoded variant SoA, matching genoray's `decode_hap` output layout —
/// see [`decode_variants_from_split`].
pub struct VariantsSoa {
    /// Per-variant (flat, hap-major) 0-based start position.
    pub pos: Vec<i32>,
    /// Per-variant (flat) `ilen` (ALT len - 1 for inline/lookup keys, negative
    /// deletion length for pure-deletion keys).
    pub ilen: Vec<i32>,
    /// Concatenated ALT bytes for all variants (pure-deletion ALT is empty).
    pub alt_bytes: Vec<u8>,
    /// Per-variant byte offsets into `alt_bytes`, len = `total_variants + 1`.
    pub str_off: Vec<i64>,
    /// Per-hap offsets into `pos`/`ilen`/`str_off`'s variant axis, len = `H + 1`.
    pub var_off: Vec<i64>,
}

/// Per-hap decode of a read-bound split into the [`VariantsSoa`], mirroring
/// genoray's `decode_hap`: merge each hap's `vk` with its present-dense entries
/// (position-sorted, `vk` before dense, snp before indel — see [`merge_hap`] via
/// [`split_to_flat`]), then decode each merged key via [`decode_alt`]. There is
/// NO overlap/clip filter here — the gather already restricts to overlapping
/// variants, unlike [`hap_diffs_svar2`]/reconstruct's ref_idx-consumed clipping,
/// which only matters for sizing a fixed-length output buffer.
pub fn decode_variants_from_split(
    br: &BatchResultSplit,
    lut_bytes: &[u8],
    lut_off: &[i64],
) -> VariantsSoa {
    let flat = split_to_flat(br);
    let ploidy = br.ploidy;
    let n_q = br.n_regions;
    let h_count = n_q * ploidy;

    let mut pos: Vec<i32> = Vec::new();
    let mut ilen: Vec<i32> = Vec::new();
    let mut alt_bytes: Vec<u8> = Vec::new();
    let mut str_off: Vec<i64> = vec![0];
    let mut var_off: Vec<i64> = Vec::with_capacity(h_count + 1);
    var_off.push(0);

    for h in 0..h_count {
        let q = h / ploidy;
        let vk_lo = flat.vk_off[h] as usize;
        let vk_hi = flat.vk_off[h + 1] as usize;
        let ds = flat.dense_range[q * 2] as usize;
        let de = flat.dense_range[q * 2 + 1] as usize;
        let base_bit = flat.dense_present_off[h] as usize;
        let present_bit = |k: usize| -> bool {
            let bit = base_bit + k;
            (flat.dense_present[bit / 8] >> (bit % 8)) & 1 == 1
        };

        let merged = merge_hap(
            &flat.vk_pos,
            &flat.vk_key,
            vk_lo,
            vk_hi,
            &flat.dense_pos,
            &flat.dense_key,
            ds,
            de,
            present_bit,
        );

        for &(p, key) in &merged {
            let (il, alt) = decode_alt(key, lut_bytes, lut_off);
            pos.push(p as i32);
            ilen.push(il as i32);
            alt_bytes.extend_from_slice(&alt);
            str_off.push(alt_bytes.len() as i64);
        }
        var_off.push(pos.len() as i64);
    }

    VariantsSoa {
        pos,
        ilen,
        alt_bytes,
        str_off,
        var_off,
    }
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

    #[test]
    fn test_hap_diffs_svar2_snp_and_del() {
        // 1 query, 1 hap, region [0, 100). Two var_key entries, no dense entries:
        // a SNP at pos 10 (ilen 0) and a single-base DEL at pos 20 (ilen -1), both
        // fully inside the region. Expected diff = 0 + (-1) = -1.
        let regions = ndarray::array![[0i32, 0, 100]];
        let ploidy = 1usize;

        let vk_pos = [10i32, 20];
        let vk_key = [
            svar2_codec::encode_alt_inline(b"A", 0) as i32,
            svar2_codec::encode_pure_del(-1) as i32,
        ];
        let vk_off: [i64; 2] = [0, 2];

        let dense_pos: [i32; 0] = [];
        let dense_key: [i32; 0] = [];
        let dense_range = ndarray::array![[0i32, 0]];
        let dense_present: [u8; 0] = [];
        let dense_present_off: [i64; 2] = [0, 0];

        let lut_bytes: [u8; 0] = [];
        let lut_off: [i64; 0] = [];

        let diffs = hap_diffs_svar2(
            regions.view(),
            ploidy,
            &vk_pos,
            &vk_key,
            &vk_off,
            &dense_pos,
            &dense_key,
            dense_range.view(),
            &dense_present,
            &dense_present_off,
            &lut_bytes,
            &lut_off,
        );

        assert_eq!(diffs[[0, 0]], -1);
    }

    #[test]
    fn test_split_to_flat_marshals_readbound_split() {
        use genoray_core::query::KeyRef;

        // ploidy=1, n_regions=1: one vk key, one dense_snp entry (present for
        // this hap), one dense_indel entry (absent for this hap).
        let br = BatchResultSplit {
            n_regions: 1,
            n_samples: 1,
            ploidy: 1,
            vk: vec![KeyRef {
                position: 5,
                key: 100,
            }],
            vk_off: vec![0, 1],
            dense_snp: vec![KeyRef {
                position: 10,
                key: 200,
            }],
            dense_snp_range: vec![(0, 1)],
            dense_snp_present: vec![0b1], // present
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![KeyRef {
                position: 15,
                key: 300,
            }],
            dense_indel_range: vec![(0, 1)],
            dense_indel_present: vec![0b0], // absent
            dense_indel_present_off: vec![0, 1],
        };

        let flat = split_to_flat(&br);

        assert_eq!(flat.vk_pos, vec![5]);
        assert_eq!(flat.vk_key, vec![100]);
        assert_eq!(flat.vk_off, vec![0, 1]);

        // Combined dense window: snp entries first, then indel entries.
        assert_eq!(flat.dense_pos, vec![10, 15]);
        assert_eq!(flat.dense_key, vec![200, 300]);
        assert_eq!(flat.dense_range, vec![0, 2]);

        // Combined presence bits: snp bit (present) then indel bit (absent),
        // LSB-first -> byte 0 = 0b01.
        assert_eq!(flat.dense_present, vec![0b01]);
        assert_eq!(flat.dense_present_off, vec![0, 2]);
    }

    #[test]
    fn test_split_to_flat_trailing_zero_byte_is_allocated() {
        use genoray_core::query::KeyRef;

        // Regression for the OOB defect: `dense_present` must always be
        // ceil(total_bits/8) bytes, even when the trailing byte is entirely
        // zero. 12 haps (ploidy=1, n_regions=12), each with a 1-entry dense/snp
        // window and no dense/indel -> 12 combined presence bits spanning 2
        // bytes. Only haps 0 and 3 carry the entry (both in byte 0); haps 8..12
        // (byte 1) are all UNSET. The grow-on-set path alone would leave byte 1
        // unallocated, so the reused kernels' `dense_present[bit/8]` read for
        // hap 8..12 would panic.
        let n = 12usize;

        // genoray's own bitstream: window size 1 per hap, bits set at 0 and 3.
        let mut dense_snp_present = vec![0u8; n.div_ceil(8)]; // 2 bytes
        dense_snp_present[0] = 0b0000_1001; // bits 0 and 3
        let dense_snp_present_off: Vec<usize> = (0..=n).collect();

        let br = BatchResultSplit {
            n_regions: n,
            n_samples: 1,
            ploidy: 1,
            vk: vec![],
            vk_off: vec![0; n + 1],
            dense_snp: vec![KeyRef {
                position: 42,
                key: 7,
            }],
            dense_snp_range: vec![(0, 1); n], // every query points at the lone entry
            dense_snp_present,
            dense_snp_present_off,
            dense_indel: vec![],
            dense_indel_range: vec![(0, 0); n], // no indel window
            dense_indel_present: vec![],
            dense_indel_present_off: vec![0; n + 1],
        };

        // Must not panic.
        let flat = split_to_flat(&br);

        // Buffer sized to ceil(total_bits/8) = ceil(12/8) = 2, NOT just up to
        // the highest set bit (byte 0).
        assert_eq!(flat.dense_present.len(), 2);
        assert_eq!(flat.dense_present, vec![0b0000_1001, 0b0000_0000]);
        // 12 haps, each contributing exactly 1 combined presence bit.
        assert_eq!(*flat.dense_present_off.last().unwrap(), n as i64);
        assert_eq!(
            flat.dense_present.len(),
            (*flat.dense_present_off.last().unwrap() as usize).div_ceil(8)
        );

        // The set bits land exactly where expected (haps 0 and 3), and the
        // trailing-byte haps are unset — proving no shift/corruption.
        for h in 0..n {
            let want = h == 0 || h == 3;
            let got = genoray_core::bits_get_bit(&flat.dense_present, h);
            assert_eq!(got, want, "hap {h} presence bit");
        }
    }

    #[test]
    fn test_decode_variants_from_split_merges_and_decodes() {
        use genoray_core::query::KeyRef;

        // 1 region, 1 sample (read-bound), ploidy 1 -> 1 hap. var_key: SNP at
        // pos 5 (inline ALT "T"). dense_snp: one entry at pos 8, PRESENT for
        // this hap (inline ALT "G"). dense_indel: one entry at pos 12, PRESENT
        // for this hap (pure DEL, ilen -2, empty ALT).
        let vk_key = svar2_codec::encode_alt_inline(b"T", 0);
        let dense_snp_key = svar2_codec::encode_alt_inline(b"G", 0);
        let dense_indel_key = svar2_codec::encode_pure_del(-2);

        let br = BatchResultSplit {
            n_regions: 1,
            n_samples: 1,
            ploidy: 1,
            vk: vec![KeyRef {
                position: 5,
                key: vk_key,
            }],
            vk_off: vec![0, 1],
            dense_snp: vec![KeyRef {
                position: 8,
                key: dense_snp_key,
            }],
            dense_snp_range: vec![(0, 1)],
            dense_snp_present: vec![0b1],
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![KeyRef {
                position: 12,
                key: dense_indel_key,
            }],
            dense_indel_range: vec![(0, 1)],
            dense_indel_present: vec![0b1],
            dense_indel_present_off: vec![0, 1],
        };

        let soa = decode_variants_from_split(&br, &[], &[0]);

        // Position-sorted: var_key SNP@5, dense/snp@8, dense/indel@12.
        assert_eq!(soa.var_off, vec![0, 3]);
        assert_eq!(soa.pos, vec![5, 8, 12]);
        assert_eq!(soa.ilen, vec![0, 0, -2]);
        assert_eq!(soa.alt_bytes, b"TG".to_vec());
        // Pure-del ALT is empty -> the 3rd variant's [start, end) is [2, 2).
        assert_eq!(soa.str_off, vec![0, 1, 2, 2]);
    }
}
