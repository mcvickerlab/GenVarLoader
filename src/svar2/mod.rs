//! SVAR2 two-source variant provider: merge a hap's var_key ⋈ dense channels and
//! decode keys via svar2-codec, feeding the reconstruction kernel with no
//! intermediate variant table. Additive to the SVAR 1.0 global-table path.

use std::borrow::Cow;
use std::ops::Range;

use genoray_core::query::{dense_abs_row, unpack_vk_src, BatchResultSplit, FieldView};
use ndarray::{Array2, ArrayView2};
use svar2_codec::{decode_key, DecodedKey};

pub mod store;
pub(crate) mod window;

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

/// LSB-first presence-bit lookup into a packed presence bitmap: returns whether bit
/// `base_bit + j` of `dense_present` is set, indexed from the least-significant bit
/// within each byte. Shared by the haplotype (`reconstruct`) and track (`tracks`)
/// per-hap `merge_hap` calls, which both read a hap's presence window this way.
#[inline]
pub fn present_bit(dense_present: &[u8], base_bit: usize, j: usize) -> bool {
    let bit = base_bit + j;
    (dense_present[bit / 8] >> (bit % 8)) & 1 == 1
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
    filter_exonic: bool,
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
            if filter_exonic && (v_start < q_start || v_end > q_end) {
                continue;
            }
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

    let dense_total: usize = (0..n_q)
        .map(|q| {
            let Range { start: ss, end: se } = br.dense_snp_range[q].clone();
            let Range { start: is_, end: ie } = br.dense_indel_range[q].clone();
            (se - ss) + (ie - is_)
        })
        .sum();
    let mut dense_pos: Vec<i32> = Vec::with_capacity(dense_total);
    let mut dense_key: Vec<i32> = Vec::with_capacity(dense_total);
    let mut dense_range: Vec<i32> = Vec::with_capacity(n_q * 2);
    for q in 0..n_q {
        let base = dense_pos.len() as i32;
        let Range { start: ss, end: se } = br.dense_snp_range[q].clone();
        for j in ss..se {
            dense_pos.push(br.dense_snp[j].position as i32);
            dense_key.push(br.dense_snp[j].key as i32);
        }
        let Range { start: is_, end: ie } = br.dense_indel_range[q].clone();
        for j in is_..ie {
            dense_pos.push(br.dense_indel[j].position as i32);
            dense_key.push(br.dense_indel[j].key as i32);
        }
        dense_range.push(base);
        dense_range.push(dense_pos.len() as i32);
    }

    // Per query `q`, every one of its `ploidy` haps has the exact same dense
    // window width `(se - ss) + (ie - is_)` — so summing per-h widths over
    // `0..h_count` (which recomputes `q = h / ploidy` via a runtime-value
    // hardware division on *every* h) is equivalent to summing the per-q
    // width once and multiplying by `ploidy`. Same total, no per-h division.
    let total_bits: usize = (0..n_q)
        .map(|q| {
            let Range { start: ss, end: se } = br.dense_snp_range[q].clone();
            let Range { start: is_, end: ie } = br.dense_indel_range[q].clone();
            ((se - ss) + (ie - is_)) * ploidy
        })
        .sum();
    let mut dense_present: Vec<u8> = vec![0u8; total_bits.div_ceil(8)];
    let mut dense_present_off: Vec<i64> = Vec::with_capacity(h_count + 1);
    let mut bit_acc: usize = 0;
    dense_present_off.push(0);
    // Same hoist as `decode_variants_from_split`: `q = h / ploidy` is
    // loop-invariant across each hap's ploidy-many iterations, so iterate `q`
    // in the outer loop and track `h` with a plain running counter instead of
    // recomputing `h / ploidy` (a runtime division) on every hap. Visits the
    // exact same `(h, q)` pairs in the exact same order (h = 0, 1, ...,
    // h_count-1, q = h/ploidy for each) — byte-identical — and also hoists the
    // per-query `(ss, se)`/`(is_, ie)` window lookup out of the
    // ploidy-many-times-redundant per-hap load.
    let mut h = 0usize;
    for q in 0..n_q {
        let Range { start: ss, end: se } = br.dense_snp_range[q].clone();
        let Range { start: is_, end: ie } = br.dense_indel_range[q].clone();
        for _hap in 0..ploidy {
            let snp_base = br.dense_snp_present_off[h];
            for k in 0..(se - ss) {
                if genoray_core::bits_get_bit(&br.dense_snp_present, snp_base + k) {
                    dense_present[bit_acc / 8] |= 1 << (bit_acc % 8);
                }
                bit_acc += 1;
            }
            let indel_base = br.dense_indel_present_off[h];
            for k in 0..(ie - is_) {
                if genoray_core::bits_get_bit(&br.dense_indel_present, indel_base + k) {
                    dense_present[bit_acc / 8] |= 1 << (bit_acc % 8);
                }
                bit_acc += 1;
            }
            dense_present_off.push(bit_acc as i64);
            h += 1;
        }
    }
    // `dense_present` was pre-sized to `total_bits.div_ceil(8)` above, and the
    // fill loop increments `bit_acc` exactly `total_bits` times, so this resize
    // is a defensive no-op (kept to document the ceil-byte invariant the reused
    // kernels rely on: they read `dense_present[bit/8]` for every window entry).
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

/// One INFO/FORMAT field's on-disk sidecars, opened for gather.
///
/// Contracts the caller MUST uphold when constructing this:
///  * `views` must be opened in [`genoray_core::layout::FieldSub::all`] order
///    (VkSnp, VkIndel, DenseSnp, DenseIndel) — `decode_variants_from_split`
///    indexes into it positionally, not by name. `FieldView` owns its mmap,
///    so this needs no lifetime.
///  * `cohort_n_samples` MUST equal the `n_samples` passed to
///    `FieldView::open` for every view in `views`. If they diverge, the dense
///    FORMAT stride computed here (`row * cohort_n_samples + orig_sample`)
///    silently disagrees with how the on-disk store was laid out — no panic,
///    just wrong values.
///  * `FieldView::bytes_at` panics on an empty sub-stream. This is only safe
///    because an empty sub-stream implies no emitted record ever resolves to
///    it (e.g. a store with zero indel calls never routes a variant to
///    `VkIndel`/`DenseIndel`). A future constructor that opens a placeholder
///    view for a sub-stream that CAN be referenced must not rely on this.
pub struct FieldGather {
    pub views: [FieldView; 4],
    pub is_format: bool,
    /// `dtype.width_bytes()`; consumed by the FFI caller, not here.
    pub width: usize,
    pub cohort_n_samples: usize,
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
#[allow(clippy::too_many_arguments)]
pub fn decode_variants_from_split(
    br: &BatchResultSplit,
    lut_bytes: &[u8],
    lut_off: &[i64],
    fields: &[FieldGather],
    on_disk_snp: &[Range<usize>],
    on_disk_indel: &[Range<usize>],
    orig_samples: &[usize],
) -> (VariantsSoa, Vec<Vec<u8>>) {
    // Fused decode straight from the split gather result: NO `split_to_flat`
    // marshaling copy, and NO per-hap `merge_hap` Vec+sort. The three per-hap
    // runs (var_key, present dense-snp, present dense-indel) are each already
    // position-sorted, so we stream a 3-way merge, decoding each key as we go.
    //
    // NOTE: dense keys are hap-independent, so pre-decoding them once per query
    // and copying across haps was tried (to kill the apparent per-(hap,variant)
    // re-decode). It REGRESSED — dense-SNP `decode_alt` is a 2-bit->1-byte no-op,
    // cheaper than the pre-decoded slice-copy + its allocation, so inline decode
    // wins. The real variant cost is the gather bit-extraction + the unavoidable
    // per-hap alt-byte emit, not the decode. (Measured 2026-07-06.)
    use genoray_core::bits_get_bit;
    let ploidy = br.ploidy;
    let n_q = br.n_regions;
    let h_count = n_q * ploidy;

    // Upper bound on total merged variants: every vk entry plus every dense
    // present bit (over-reserving is harmless).
    let vk_total = br.vk_off[h_count];
    let dense_bits = br.dense_snp_present_off[h_count] + br.dense_indel_present_off[h_count];
    let cap = vk_total + dense_bits;
    let mut pos: Vec<i32> = Vec::with_capacity(cap);
    let mut ilen: Vec<i32> = Vec::with_capacity(cap);
    let mut alt_bytes: Vec<u8> = Vec::with_capacity(cap);
    let mut str_off: Vec<i64> = Vec::with_capacity(cap + 1);
    str_off.push(0);
    let mut var_off: Vec<i64> = Vec::with_capacity(h_count + 1);
    var_off.push(0);
    let mut field_bufs: Vec<Vec<u8>> = fields
        .iter()
        .map(|f| Vec::with_capacity(cap * f.width))
        .collect();

    // A `BatchResultSplit` without provenance (`gather_haps_readbound`, not
    // `_src`) leaves `vk_src` empty; indexing it below would panic with an
    // opaque out-of-bounds error instead of naming the real cause. Check once,
    // O(1), instead of a per-variant `debug_assert` in the hot loop — this is
    // an unconditional `assert_eq!` (matches genoray's own contract check in
    // `BatchResultSplit`) so it fires in release builds too, before any wrong
    // provenance can attach a field value to the wrong variant.
    if !fields.is_empty() {
        assert_eq!(
            br.vk_src.len(),
            br.vk.len(),
            "fields require a BatchResultSplit from gather_haps_readbound_src \
             (vk_src must be populated 1:1 with vk)"
        );
    }

    let mut h = 0usize;
    for q in 0..n_q {
        let Range { start: ss, end: se } = br.dense_snp_range[q].clone();
        let Range { start: is_, end: ie } = br.dense_indel_range[q].clone();
        for _hap in 0..ploidy {
            let vk_lo = br.vk_off[h];
            let vk_hi = br.vk_off[h + 1];
            let snp_base = br.dense_snp_present_off[h];
            let indel_base = br.dense_indel_present_off[h];

            // 3-way merge, position-sorted. On equal positions the priority is
            // var_key < dense-snp < dense-indel, exactly the stable-sort tie
            // order of the previous collect-then-sort `merge_hap` (var_key
            // pushed first, then dense-snp, then dense-indel). Byte-identical.
            let mut i_vk = vk_lo;
            let mut i_sn = ss;
            let mut i_in = is_;
            loop {
                // Advance dense pointers to the next PRESENT entry. Pointers are
                // monotonic, so total skip work is O(window) per hap, not O(n^2).
                while i_sn < se && !bits_get_bit(&br.dense_snp_present, snp_base + (i_sn - ss)) {
                    i_sn += 1;
                }
                while i_in < ie
                    && !bits_get_bit(&br.dense_indel_present, indel_base + (i_in - is_))
                {
                    i_in += 1;
                }
                let has_vk = i_vk < vk_hi;
                let has_sn = i_sn < se;
                let has_in = i_in < ie;
                if !has_vk && !has_sn && !has_in {
                    break;
                }
                let p_vk = if has_vk { br.vk[i_vk].position } else { u32::MAX };
                let p_sn = if has_sn { br.dense_snp[i_sn].position } else { u32::MAX };
                let p_in = if has_in { br.dense_indel[i_in].position } else { u32::MAX };
                let (p, key, chan, cidx) = if has_vk && p_vk <= p_sn && p_vk <= p_in {
                    let e = &br.vk[i_vk];
                    let out = (e.position, e.key, 0u8, i_vk);
                    i_vk += 1;
                    out
                } else if has_sn && p_sn <= p_in {
                    let e = &br.dense_snp[i_sn];
                    let out = (e.position, e.key, 1u8, i_sn);
                    i_sn += 1;
                    out
                } else {
                    let e = &br.dense_indel[i_in];
                    let out = (e.position, e.key, 2u8, i_in);
                    i_in += 1;
                    out
                };
                let (il, alt) = decode_alt(key, lut_bytes, lut_off);
                pos.push(p as i32);
                ilen.push(il as i32);
                alt_bytes.extend_from_slice(&alt);
                str_off.push(alt_bytes.len() as i64);

                if !fields.is_empty() {
                    // Resolve (sub_ix, row) for this emitted variant.
                    let (sub_ix, row, is_dense) = match chan {
                        0 => {
                            let (is_indel, call_idx) = unpack_vk_src(br.vk_src[cidx]);
                            (if is_indel { 1 } else { 0 }, call_idx, false)
                        }
                        1 => (
                            2,
                            dense_abs_row(&on_disk_snp[q], &br.dense_snp_range[q], cidx),
                            true,
                        ),
                        _ => (
                            3,
                            dense_abs_row(&on_disk_indel[q], &br.dense_indel_range[q], cidx),
                            true,
                        ),
                    };
                    for (fi, f) in fields.iter().enumerate() {
                        // var_key entries are already per-(variant, sample) CALLS, so a
                        // FORMAT value is indexed by the call index directly. Only the
                        // DENSE channel, which is variant-major over the whole cohort,
                        // needs the sample stride.
                        let elem = if is_dense && f.is_format {
                            row * f.cohort_n_samples + orig_samples[q]
                        } else {
                            row
                        };
                        field_bufs[fi].extend_from_slice(f.views[sub_ix].bytes_at(elem));
                    }
                }
            }
            var_off.push(pos.len() as i64);
            h += 1;
        }
    }

    (
        VariantsSoa {
            pos,
            ilen,
            alt_bytes,
            str_off,
            var_off,
        },
        field_bufs,
    )
}

/// Fill `out_data`/`out_offsets` (cleared + refilled, capacity reused) with the
/// read-bound reconstruction of `n_q = region_starts_v.len()` regions x `ploidy`
/// haplotypes. Identical logic to the original `reconstruct_haplotypes_from_svar2_readbound`
/// py.detach body (`src/ffi/mod.rs`); `parallel` is passed straight to the kernel. Runs
/// GIL-free — callers wrap in `py.detach`. `out_offsets` has length `n_q*ploidy + 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn svar2_readbound_chain(
    reader: &genoray_core::query::ContigReader,
    region_starts_v: &[u32],
    orig_samples_v: &[usize],
    vk_snp_range_v: &[Range<usize>],
    vk_indel_range_v: &[Range<usize>],
    dense_snp_range_v: &[Range<usize>],
    dense_indel_range_v: &[Range<usize>],
    regions: numpy::ndarray::ArrayView2<i32>,
    shifts_a: numpy::ndarray::ArrayView2<i32>,
    ref_a: numpy::ndarray::ArrayView1<u8>,
    ref_offsets_a: numpy::ndarray::ArrayView1<i64>,
    pad_char: u8,
    output_length: i64,
    parallel: bool,
    filter_exonic: bool,
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<i64>,
) {
    use numpy::ndarray::{ArrayView1, ArrayView2};
    let ploidy = shifts_a.ncols();
    let n_q = regions.nrows();

    let rb = genoray_core::query::HapRanges::new(
        region_starts_v,
        orig_samples_v,
        vk_snp_range_v,
        vk_indel_range_v,
        dense_snp_range_v,
        dense_indel_range_v,
        ploidy,
    );
    let br = genoray_core::query::gather_haps_readbound(reader, &rb);
    let (lut_bytes, lut_off_u64) = reader.lut_arrays();
    let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();
    let flat = crate::svar2::split_to_flat(&br);
    let dense_range_a = ArrayView2::from_shape((n_q, 2), &flat.dense_range).unwrap();

    let diffs = crate::svar2::hap_diffs_svar2(
        regions,
        ploidy,
        &flat.vk_pos,
        &flat.vk_key,
        &flat.vk_off,
        &flat.dense_pos,
        &flat.dense_key,
        dense_range_a,
        &flat.dense_present,
        &flat.dense_present_off,
        &lut_bytes,
        &lut_off,
        filter_exonic,
    );

    // Offsets (prefix sum) into the reused buffer.
    let n_work = n_q * ploidy;
    out_offsets.clear();
    out_offsets.resize(n_work + 1, 0);
    let mut acc: i64 = 0;
    out_offsets[0] = 0;
    for k in 0..n_work {
        let query = k / ploidy;
        let hap = k % ploidy;
        let len: i64 = if output_length >= 0 {
            output_length
        } else {
            let ref_len = (regions[[query, 2]] - regions[[query, 1]]) as i64;
            let diff = diffs[[query, hap]] as i64;
            (ref_len + diff).max(0)
        };
        acc += len;
        out_offsets[k + 1] = acc;
    }

    // Output buffer (reused capacity; fully overwritten by reconstruct).
    let total = out_offsets[n_work] as usize;
    out_data.clear();
    out_data.resize(total, 0u8);

    let out_offsets_view = ArrayView1::from(out_offsets.as_slice());
    let out_bounds = crate::reconstruct::bounds_from_offsets(out_offsets_view);
    let out_data_view = numpy::ndarray::ArrayViewMut1::from(out_data.as_mut_slice());
    crate::reconstruct::reconstruct_haplotypes_from_svar2(
        out_data_view,
        out_bounds.view(),
        regions,
        shifts_a,
        ArrayView1::from(flat.vk_pos.as_slice()),
        ArrayView1::from(flat.vk_key.as_slice()),
        ArrayView1::from(flat.vk_off.as_slice()),
        ArrayView1::from(flat.dense_pos.as_slice()),
        ArrayView1::from(flat.dense_key.as_slice()),
        dense_range_a,
        ArrayView1::from(flat.dense_present.as_slice()),
        ArrayView1::from(flat.dense_present_off.as_slice()),
        ArrayView1::from(lut_bytes.as_slice()),
        ArrayView1::from(lut_off.as_slice()),
        ref_a,
        ref_offsets_a,
        pad_char,
        parallel,
        filter_exonic,
    );
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
            false,
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
            dense_snp_range: vec![0..1],
            dense_snp_present: vec![0b1], // present
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![KeyRef {
                position: 15,
                key: 300,
            }],
            dense_indel_range: vec![0..1],
            dense_indel_present: vec![0b0], // absent
            dense_indel_present_off: vec![0, 1],
            ..Default::default()
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
            dense_snp_range: vec![0..1; n], // every query points at the lone entry
            dense_snp_present,
            dense_snp_present_off,
            dense_indel: vec![],
            dense_indel_range: vec![0..0; n], // no indel window
            dense_indel_present: vec![],
            dense_indel_present_off: vec![0; n + 1],
            ..Default::default()
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
    fn test_split_to_flat_ploidy_gt1_reuses_per_query_window_across_haps() {
        use genoray_core::query::KeyRef;

        // Regression for the q=h/ploidy hoist: ploidy=2, n_regions=2 (h_count=4)
        // so each query's dense_snp/dense_indel window is shared by 2 haps, but
        // each hap has its own presence bits at its own `*_present_off` offset.
        // This exercises the outer-q/inner-hap loop restructuring (the old code
        // recomputed `q = h / ploidy` per hap via a runtime division; the new
        // code hoists the per-query window lookup out of the hap loop) — proving
        // it doesn't mix up which hap's presence bits attach to which query's
        // window. The 12 combined presence bits (3/hap * 4 haps) also cross a
        // byte boundary with mixed set/unset bits, like the trailing-zero test,
        // but here across queries+haps instead of a single-bit-per-hap window.
        let dense_snp: Vec<KeyRef> = (0..4)
            .map(|i| KeyRef {
                position: 10 + i,
                key: 200 + i,
            })
            .collect();
        let dense_indel: Vec<KeyRef> = (0..2)
            .map(|i| KeyRef {
                position: 50 + i,
                key: 500 + i,
            })
            .collect();

        let br = BatchResultSplit {
            n_regions: 2,
            n_samples: 1,
            ploidy: 2,
            vk: vec![],
            vk_off: vec![0; 5],
            dense_snp,
            // query 0 owns snp[0..2), query 1 owns snp[2..4) — width 2/query,
            // shared by both of that query's haps.
            dense_snp_range: vec![0..2, 2..4],
            // hap0 bits(0,1)=(1,0), hap1 bits(2,3)=(0,1), hap2 bits(4,5)=(1,1),
            // hap3 bits(6,7)=(0,0) -> byte 0b0011_1001.
            dense_snp_present: vec![0b0011_1001],
            dense_snp_present_off: vec![0, 2, 4, 6, 8],
            dense_indel,
            // query 0 owns indel[0..1), query 1 owns indel[1..2) — width 1/query.
            dense_indel_range: vec![0..1, 1..2],
            // hap0=1, hap1=0, hap2=1, hap3=1 -> byte 0b0000_1101.
            dense_indel_present: vec![0b0000_1101],
            dense_indel_present_off: vec![0, 1, 2, 3, 4],
            ..Default::default()
        };

        let flat = split_to_flat(&br);

        // snp-then-indel per query, queries in order.
        assert_eq!(flat.dense_pos, vec![10, 11, 50, 12, 13, 51]);
        assert_eq!(flat.dense_key, vec![200, 201, 500, 202, 203, 501]);
        assert_eq!(flat.dense_range, vec![0, 3, 3, 6]);

        // 4 haps * 3 bits/hap = 12 bits -> 2 bytes, spanning a byte boundary.
        assert_eq!(flat.dense_present_off, vec![0, 3, 6, 9, 12]);
        assert_eq!(flat.dense_present, vec![0b1101_0101, 0b0000_1001]);
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
            dense_snp_range: vec![0..1],
            dense_snp_present: vec![0b1],
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![KeyRef {
                position: 12,
                key: dense_indel_key,
            }],
            dense_indel_range: vec![0..1],
            dense_indel_present: vec![0b1],
            dense_indel_present_off: vec![0, 1],
            ..Default::default()
        };

        let (soa, _bufs) = decode_variants_from_split(&br, &[], &[0], &[], &[], &[], &[]);

        // Position-sorted: var_key SNP@5, dense/snp@8, dense/indel@12.
        assert_eq!(soa.var_off, vec![0, 3]);
        assert_eq!(soa.pos, vec![5, 8, 12]);
        assert_eq!(soa.ilen, vec![0, 0, -2]);
        assert_eq!(soa.alt_bytes, b"TG".to_vec());
        // Pure-del ALT is empty -> the 3rd variant's [start, end) is [2, 2).
        assert_eq!(soa.str_off, vec![0, 1, 2, 2]);
    }

    /// Exercises the two things the asm-inspection pass touched: (1) `q =
    /// h / ploidy` computed via an incrementing counter instead of a division
    /// (needs `ploidy > 1` so some consecutive haps share a `q`, which the
    /// single-hap tests above never trigger), and (2) the `present_bit`
    /// closure's read of `dense_present`, with a mix of
    /// present/absent bits whose per-hap `base_bit` windows straddle a byte
    /// boundary (hap 1's 5-bit window covers global bits 5..10, i.e. bytes 0
    /// and 1).
    #[test]
    fn test_decode_variants_from_split_byte_identical_presence_edge() {
        use genoray_core::query::KeyRef;

        let k = |b: &[u8]| svar2_codec::encode_alt_inline(b, 0);

        // 2 regions x ploidy 2 = 4 haps, no var_key entries (isolates the
        // dense/present-bit path). Region 0's dense window is 3 snp + 2 indel
        // (width 5, shared by haps 0 & 1); region 1's is 3 snp + 0 indel
        // (width 3, shared by haps 2 & 3). Combined presence bitstream is 16
        // bits = 2 bytes, with hap 1's window (global bits 5..10) crossing
        // the byte-0/byte-1 boundary at bit 8.
        let br = BatchResultSplit {
            n_regions: 2,
            n_samples: 1,
            ploidy: 2,
            vk: vec![],
            vk_off: vec![0, 0, 0, 0, 0],
            dense_snp: vec![
                KeyRef { position: 10, key: k(b"A") },
                KeyRef { position: 11, key: k(b"C") },
                KeyRef { position: 12, key: k(b"G") },
                KeyRef { position: 50, key: k(b"T") },
                KeyRef { position: 51, key: k(b"A") },
                KeyRef { position: 52, key: k(b"C") },
            ],
            dense_snp_range: vec![0..3, 3..6],
            // Per-hap snp-bit widths 3,3,3,3 -> offsets 0,3,6,9,12. Bitstream
            // (idx0..11): 1,0,1, 0,1,0, 1,1,0, 0,0,1 -> byte0 = 0b1101_0101
            // (bits0-7: 1,0,1,0,1,0,1,1 -> 1+4+16+64+128=213), byte1 low
            // nibble (bits8-11: 0,0,0,1 -> 8).
            dense_snp_present: vec![213, 8],
            dense_snp_present_off: vec![0, 3, 6, 9, 12],
            dense_indel: vec![
                KeyRef { position: 13, key: svar2_codec::encode_pure_del(-2) },
                KeyRef { position: 14, key: svar2_codec::encode_pure_del(-5) },
            ],
            dense_indel_range: vec![0..2, 2..2],
            // Per-hap indel-bit widths 2,2,0,0 -> offsets 0,2,4,4,4.
            // Bitstream (idx0..3): 1,0, 0,1 -> byte0 = 0b1001 (1+8=9).
            dense_indel_present: vec![9],
            dense_indel_present_off: vec![0, 2, 4, 4, 4],
            ..Default::default()
        };

        let (soa, _bufs) = decode_variants_from_split(&br, &[], &[0], &[], &[], &[], &[]);

        // hap0 (q0): snp present [1,0,1] -> keeps pos10("A"),pos12("G");
        //   indel present [1,0] -> keeps pos13(ilen -2).
        // hap1 (q0): snp present [0,1,0] -> keeps pos11("C");
        //   indel present [0,1] -> keeps pos14(ilen -5).
        // hap2 (q1): snp present [1,1,0] -> keeps pos50("T"),pos51("A").
        // hap3 (q1): snp present [0,0,1] -> keeps pos52("C").
        assert_eq!(soa.pos, vec![10, 12, 13, 11, 14, 50, 51, 52]);
        assert_eq!(soa.ilen, vec![0, 0, -2, 0, -5, 0, 0, 0]);
        assert_eq!(soa.alt_bytes, b"AGCTAC".to_vec());
        assert_eq!(soa.str_off, vec![0, 1, 2, 2, 3, 3, 4, 5, 6]);
        assert_eq!(soa.var_off, vec![0, 3, 5, 7, 8]);
    }

    /// Build 4 FieldViews over a store where each sub-stream's content is
    /// distinguishable: element `i` of sub-stream `sub_ix` (in
    /// `FieldSub::all()` order, i.e. `[VkSnp=0, VkIndel=1, DenseSnp=2,
    /// DenseIndel=3]`) has value `100 * sub_ix + i`. A test that decodes value
    /// `V` therefore reveals BOTH which sub-stream (`V / 100`) and which row
    /// (`V % 100`) the decoder attributed to that variant — a same-content
    /// store (as an earlier version of this fixture used) cannot distinguish
    /// "read the right sub-stream" from "read some sub-stream".
    /// Returns the TempDir too — it must outlive the views (dropping it deletes the mmapped files).
    fn make_identity_i32_fields() -> (tempfile::TempDir, Vec<FieldGather>) {
        use genoray_core::field::StorageDtype;
        use genoray_core::layout::{ContigPaths, FieldSub};

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_str().unwrap();
        let paths = ContigPaths::new(base, "chr1");

        const N: usize = 8; // enough for call idx 5 and dense row 3/4

        let mut views = Vec::with_capacity(4);
        for (sub_ix, sub) in FieldSub::all().into_iter().enumerate() {
            let bytes: Vec<u8> = (0..N as i32)
                .flat_map(|i| (100 * sub_ix as i32 + i).to_le_bytes())
                .collect();
            let p = paths.field_values("info", "X", sub);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, &bytes).unwrap();
            views.push(FieldView::open(&paths, "info", "X", sub, StorageDtype::I32, 1).unwrap());
        }
        let views: [FieldView; 4] = views.try_into().map_err(|_| ()).unwrap();

        (
            tmp,
            vec![FieldGather {
                views,
                is_format: false, // INFO -> element index is the row itself
                width: 4,
                cohort_n_samples: 1,
            }],
        )
    }

    #[test]
    fn test_decode_fields_provenance_identity() {
        use genoray_core::query::{pack_vk_src, KeyRef};

        // One query, ploidy 1, all four provenance channels represented:
        //   var_key SNP   entry at pos 10, provenance = (snp,   call idx 5)  -> sub VkSnp=0
        //   var_key indel entry at pos 15, provenance = (indel, call idx 2)  -> sub VkIndel=1
        //   dense-snp     entry at pos 20, output window 0..1, ON-DISK window 3..4 -> abs row 3, sub DenseSnp=2
        //   dense-indel   entry at pos 25, output window 0..1, ON-DISK window 4..5 -> abs row 4, sub DenseIndel=3
        let br = BatchResultSplit {
            n_regions: 1,
            n_samples: 1,
            ploidy: 1,
            vk: vec![
                KeyRef {
                    position: 10,
                    key: svar2_codec::encode_pure_del(-1),
                },
                KeyRef {
                    position: 15,
                    key: svar2_codec::encode_pure_del(-1),
                },
            ],
            vk_off: vec![0, 2],
            vk_src: vec![pack_vk_src(false, 5), pack_vk_src(true, 2)],
            dense_snp: vec![KeyRef {
                position: 20,
                key: svar2_codec::encode_pure_del(-1),
            }],
            dense_snp_range: vec![0..1],
            dense_snp_present: vec![0b1],
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![KeyRef {
                position: 25,
                key: svar2_codec::encode_pure_del(-1),
            }],
            dense_indel_range: vec![0..1],
            dense_indel_present: vec![0b1],
            dense_indel_present_off: vec![0, 1],
            ..Default::default()
        };

        let (_tmp, fields) = make_identity_i32_fields(); // keep _tmp alive: it owns the tempdir

        let (soa, bufs) = decode_variants_from_split(
            &br,
            &[],
            &[0i64],
            &fields,
            &[3..4], // on_disk_snp: the dense-snp window really lives at rows 3..4 on disk
            &[4..5], // on_disk_indel: the dense-indel window really lives at rows 4..5 on disk
            &[0],    // orig_samples
        );

        // Position-sorted: var_key(10,15) before dense(20,25); snp before indel on ties (none here).
        assert_eq!(soa.pos, vec![10, 15, 20, 25]);

        // value = 100*sub_ix + row, so each decoded value pins BOTH the
        // sub-stream routing and the row/offset arithmetic for its channel:
        //   var_key snp   -> sub 0, call idx 5 -> 100*0+5   = 5
        //   var_key indel -> sub 1, call idx 2 -> 100*1+2   = 102
        //   dense-snp     -> sub 2, abs row  3 -> 100*2+3   = 203
        //   dense-indel   -> sub 3, abs row  4 -> 100*3+4   = 304
        let vals: Vec<i32> = bufs[0]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![5, 102, 203, 304]);
    }

    #[test]
    fn test_decode_fields_format_dense_stride_and_var_key_unstrided() {
        use genoray_core::field::StorageDtype;
        use genoray_core::layout::{ContigPaths, FieldSub};
        use genoray_core::query::{pack_vk_src, KeyRef};

        // A FORMAT field over a 3-sample cohort. Store is plain identity
        // (element i has value i) across all four subs — this test isolates
        // the *stride* arithmetic, not sub-stream routing (that's pinned by
        // `test_decode_fields_provenance_identity`).
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_str().unwrap();
        let paths = ContigPaths::new(base, "chr1");

        const N: usize = 16; // covers dense elem 3*3+2=11 and var_key call idx 5
        let bytes: Vec<u8> = (0..N as i32).flat_map(|i| i.to_le_bytes()).collect();

        let mut views = Vec::with_capacity(4);
        for sub in FieldSub::all() {
            let p = paths.field_values("format", "DP", sub);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, &bytes).unwrap();
            // n_samples passed to `open` MUST match `cohort_n_samples` below.
            views.push(FieldView::open(&paths, "format", "DP", sub, StorageDtype::I32, 3).unwrap());
        }
        let views: [FieldView; 4] = views.try_into().map_err(|_| ()).unwrap();

        let fields = vec![FieldGather {
            views,
            is_format: true,
            width: 4,
            cohort_n_samples: 3,
        }];

        // var_key SNP entry at pos 10, call idx 5; dense-snp entry at pos 20,
        // output window 0..1, ON-DISK window 3..4 -> abs row 3.
        let br = BatchResultSplit {
            n_regions: 1,
            n_samples: 1,
            ploidy: 1,
            vk: vec![KeyRef {
                position: 10,
                key: svar2_codec::encode_pure_del(-1),
            }],
            vk_off: vec![0, 1],
            vk_src: vec![pack_vk_src(false, 5)],
            dense_snp: vec![KeyRef {
                position: 20,
                key: svar2_codec::encode_pure_del(-1),
            }],
            dense_snp_range: vec![0..1],
            dense_snp_present: vec![0b1],
            dense_snp_present_off: vec![0, 1],
            dense_indel: vec![],
            dense_indel_range: vec![0..0],
            dense_indel_present: vec![],
            dense_indel_present_off: vec![0, 0],
            ..Default::default()
        };

        let (soa, bufs) = decode_variants_from_split(
            &br,
            &[],
            &[0i64],
            &fields,
            &[3..4], // on_disk_snp
            &[0..0], // on_disk_indel
            &[2],    // orig_samples: this query's original cohort sample index is 2
        );

        assert_eq!(soa.pos, vec![10, 20]);

        let vals: Vec<i32> = bufs[0]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // var_key FORMAT is UNSTRIDED: a var_key entry is already a per-CALL
        // value, so the element index is the call index directly (5), NOT
        // `call_idx * cohort_n_samples + orig_sample` (5*3+2=17). This pins
        // the `is_dense && f.is_format` gate — the part most likely to
        // silently regress.
        // dense FORMAT IS strided: abs row 3, orig_sample 2, cohort_n_samples 3
        // -> element 3*3+2 = 11.
        assert_eq!(vals, vec![5, 11]);
    }
}
