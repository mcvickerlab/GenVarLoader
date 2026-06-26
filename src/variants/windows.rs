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

/// Assembled flat buffers returned by the mode orchestrators. `byte_bufs` carry
/// raw allele bytes (u8); `tok_bufs` carry LUT-applied tokens (`Tok`). Each
/// tuple is `(field_name, data, seq_offsets)`.
pub struct VariantBufs<Tok> {
    pub byte_bufs: Vec<(&'static str, Array1<u8>, Array1<i64>)>,
    pub tok_bufs: Vec<(&'static str, Array1<Tok>, Array1<i64>)>,
}

/// Gather per-selected-variant `start`/`ilen` from the GLOBAL arrays via `v_idxs`.
fn gather_starts_ilens(
    v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i32>) {
    let n = v_idxs.len();
    let mut s = Array1::<i32>::zeros(n);
    let mut il = Array1::<i32>::zeros(n);
    for i in 0..n {
        let v = v_idxs[i] as usize;
        s[i] = v_starts[v];
        il[i] = ilens[v];
    }
    (s, il)
}

/// Plain-`variants` assembly tail: raw alt bytes (always), raw ref bytes
/// (optional), `flank_tokens` ride-along (optional). Mirrors the variants tail
/// of `get_variants_flat` (gather_alleles + compute_flank_tokens).
#[allow(clippy::too_many_arguments)]
pub fn assemble_variants_mode<Tok: Copy>(
    v_idxs: ArrayView1<i32>,
    row_offsets: ArrayView1<i64>,
    alt_global: ArrayView1<u8>,
    alt_off_global: ArrayView1<i64>,
    ref_global: Option<ArrayView1<u8>>,
    ref_off_global: Option<ArrayView1<i64>>,
    want_flank: bool,
    flank_len: i64,
    lut: Option<ArrayView1<Tok>>,
    v_contigs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> VariantBufs<Tok> {
    let mut byte_bufs = Vec::new();
    let mut tok_bufs = Vec::new();

    let (alt_data, alt_seq_off) =
        crate::variants::gather_alleles(v_idxs, alt_global, alt_off_global);
    byte_bufs.push(("alt", alt_data, alt_seq_off));

    if let (Some(rg), Some(ro)) = (ref_global, ref_off_global) {
        let (ref_data, ref_seq_off) = crate::variants::gather_alleles(v_idxs, rg, ro);
        byte_bufs.push(("ref", ref_data, ref_seq_off));
    }

    if want_flank {
        let lut = lut.expect("flank tokens requested but no token LUT supplied");
        let (starts_v, ilens_v) = gather_starts_ilens(v_idxs, v_starts, ilens);
        let (rw_data, rw_off) = fetch_windows(
            v_contigs, starts_v.view(), ilens_v.view(), flank_len, reference, ref_offsets,
            pad_char,
        );
        let l = flank_len as usize;
        let (f5, f3) = slice_flanks(rw_data.view(), rw_off.view(), l);
        // Concatenate [f5 | f3] per variant (2L tokens, variant-major), tokenize.
        let n = f5.len() / l;
        let mut flank_bytes: Vec<u8> = Vec::with_capacity(n * 2 * l);
        for i in 0..n {
            for k in 0..l {
                flank_bytes.push(f5[i * l + k]);
            }
            for k in 0..l {
                flank_bytes.push(f3[i * l + k]);
            }
        }
        let fb = Array1::from_vec(flank_bytes);
        let tok = tokenize(fb.view(), lut);
        // flank_tokens offsets are the variant-level row_offsets (fixed 2L inner
        // axis carried separately Python-side as a trailing regular dim).
        tok_bufs.push(("flank_tokens", tok, row_offsets.to_owned()));
    }

    VariantBufs { byte_bufs, tok_bufs }
}

/// `variant-windows` assembly tail. `ref_mode`/`alt_mode`: 1 = flanked window
/// (`[start-L,end+L)` for ref; `flank5.alt.flank3` for alt), 2 = bare tokenized
/// allele. Produces only token buffers (scalar fields are handled Python-side).
/// Mirrors the windows branch of `get_variants_flat` (incl. the single fused
/// fetch shared by ref_window + alt_window).
#[allow(clippy::too_many_arguments)]
pub fn assemble_windows_mode<Tok: Copy>(
    v_idxs: ArrayView1<i32>,
    _row_offsets: ArrayView1<i64>,
    ref_mode: i64,
    alt_mode: i64,
    alt_global: ArrayView1<u8>,
    alt_off_global: ArrayView1<i64>,
    ref_global: Option<ArrayView1<u8>>,
    ref_off_global: Option<ArrayView1<i64>>,
    flank_len: i64,
    lut: ArrayView1<Tok>,
    v_contigs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> VariantBufs<Tok> {
    let mut tok_bufs = Vec::new();
    let l = flank_len as usize;

    // alt alleles are always gathered (needed for alt window or bare alt).
    let (alt_data, alt_seq_off) =
        crate::variants::gather_alleles(v_idxs, alt_global, alt_off_global);

    // One fused fetch if either side needs a window read.
    let need_fetch = ref_mode == 1 || alt_mode == 1;
    let fetched = if need_fetch {
        let (starts_v, ilens_v) = gather_starts_ilens(v_idxs, v_starts, ilens);
        Some(fetch_windows(
            v_contigs, starts_v.view(), ilens_v.view(), flank_len, reference, ref_offsets,
            pad_char,
        ))
    } else {
        None
    };

    // ref side (ordered first to match Python field insertion order).
    if ref_mode == 1 {
        let (rw_data, rw_off) = fetched.as_ref().expect("ref window needs a fetch");
        let tok = tokenize(rw_data.view(), lut);
        tok_bufs.push(("ref_window", tok, rw_off.clone()));
    } else if ref_mode == 2 {
        let rg = ref_global.expect("bare ref allele needs ref byte buffer");
        let ro = ref_off_global.expect("bare ref allele needs ref offsets");
        let (ref_data, ref_seq_off) = crate::variants::gather_alleles(v_idxs, rg, ro);
        let tok = tokenize(ref_data.view(), lut);
        tok_bufs.push(("ref", tok, ref_seq_off));
    }

    // alt side.
    if alt_mode == 1 {
        let (rw_data, rw_off) = fetched.as_ref().expect("alt window needs a fetch");
        let (f5, f3) = slice_flanks(rw_data.view(), rw_off.view(), l);
        let (alt_bytes, alt_off) = assemble_alt_window(
            f5.view(),
            f3.view(),
            alt_data.view(),
            alt_seq_off.view(),
            l,
        );
        let tok = tokenize(alt_bytes.view(), lut);
        tok_bufs.push(("alt_window", tok, alt_off));
    } else if alt_mode == 2 {
        let tok = tokenize(alt_data.view(), lut);
        tok_bufs.push(("alt", tok, alt_seq_off));
    }

    VariantBufs { byte_bufs: Vec::new(), tok_bufs }
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

    #[test]
    fn test_assemble_windows_mode_both_windows() {
        use ndarray::Array1 as A1;
        // Global alt alleles: v0="A"(65). offsets [0,1].
        let alt_global = arr1(&[65u8]);
        let alt_off = arr1(&[0i64, 1]);
        let v_idxs = arr1(&[0i32]);
        let row_offsets = arr1(&[0i64, 1]);
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let v_contigs = arr1(&[0i32]);
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect()); // identity

        let bufs = assemble_windows_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            1, // ref_mode = window
            1, // alt_mode = window
            alt_global.view(),
            alt_off.view(),
            None,
            None,
            1, // flank_len
            lut.view(),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        // SNP start=5 ilen=0 → end=6; read [4,7) = [4,5,6]. L=1.
        // ref_window tokens (identity) = [4,5,6], off [0,3].
        // alt_window = f5[4] . alt[65] . f3[6] = [4,65,6], off [0,3].
        assert_eq!(bufs.byte_bufs.len(), 0);
        let names: Vec<&str> = bufs.tok_bufs.iter().map(|t| t.0).collect();
        assert_eq!(names, vec!["ref_window", "alt_window"]);
        assert_eq!(bufs.tok_bufs[0].1.to_vec(), vec![4u8, 5, 6]);
        assert_eq!(bufs.tok_bufs[0].2.to_vec(), vec![0i64, 3]);
        assert_eq!(bufs.tok_bufs[1].1.to_vec(), vec![4u8, 65, 6]);
        assert_eq!(bufs.tok_bufs[1].2.to_vec(), vec![0i64, 3]);
    }

    #[test]
    fn test_assemble_windows_mode_bare_alleles() {
        use ndarray::Array1 as A1;
        // alt v0="AC"(65,67); ref v0="G"(71).
        let alt_global = arr1(&[65u8, 67]);
        let alt_off = arr1(&[0i64, 2]);
        let ref_global = arr1(&[71u8]);
        let ref_off = arr1(&[0i64, 1]);
        let v_idxs = arr1(&[0i32]);
        let row_offsets = arr1(&[0i64, 1]);
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let v_contigs = arr1(&[0i32]);
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect());

        let bufs = assemble_windows_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            2, // ref_mode = allele (bare)
            2, // alt_mode = allele (bare)
            alt_global.view(),
            alt_off.view(),
            Some(ref_global.view()),
            Some(ref_off.view()),
            1,
            lut.view(),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        let names: Vec<&str> = bufs.tok_bufs.iter().map(|t| t.0).collect();
        assert_eq!(names, vec!["ref", "alt"]);
        // bare ref tokens = [71], off [0,1]; bare alt tokens = [65,67], off [0,2].
        assert_eq!(bufs.tok_bufs[0].1.to_vec(), vec![71u8]);
        assert_eq!(bufs.tok_bufs[0].2.to_vec(), vec![0i64, 1]);
        assert_eq!(bufs.tok_bufs[1].1.to_vec(), vec![65u8, 67]);
        assert_eq!(bufs.tok_bufs[1].2.to_vec(), vec![0i64, 2]);
    }

    #[test]
    fn test_assemble_variants_mode_alt_and_flank() {
        use ndarray::Array1 as A1;
        // Global alleles: v0="A"(65), v1="CG"(67,71). offsets [0,1,3].
        let alt_global = arr1(&[65u8, 67, 71]);
        let alt_off = arr1(&[0i64, 1, 3]);
        // Select v_idxs [1, 0] in one row.
        let v_idxs = arr1(&[1i32, 0]);
        let row_offsets = arr1(&[0i64, 2]);
        // Reference 0..20, single contig. v_starts/ilens are GLOBAL (indexed by v_idx).
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32, 8]); // global per-variant
        let ilens = arr1(&[0i32, 0]);
        let v_contigs = arr1(&[0i32, 0]); // per-selected-variant contig
        // L=1, token LUT: identity-ish u8 (byte value -> itself for the test).
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect());

        let bufs = assemble_variants_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            alt_global.view(),
            alt_off.view(),
            None, // no ref alleles
            None,
            true, // want_flank
            1,    // flank_len
            Some(lut.view()),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        // byte_bufs: only "alt". v_idxs [1,0] → "CG" then "A" → [67,71,65], off [0,2,3].
        assert_eq!(bufs.byte_bufs.len(), 1);
        let (name, data, off) = &bufs.byte_bufs[0];
        assert_eq!(*name, "alt");
        assert_eq!(data.to_vec(), vec![67u8, 71, 65]);
        assert_eq!(off.to_vec(), vec![0i64, 2, 3]);
        // tok_bufs: only "flank_tokens". Each variant: [f5(1) | f3(1)] = 2 tokens.
        // var0 = v_idx 1: start=8, ilen=0 → end=9, read [7,10) = [7,8,9]; f5=[7], f3=[9].
        // var1 = v_idx 0: start=5, ilen=0 → end=6, read [4,7) = [4,5,6]; f5=[4], f3=[6].
        // tokens (identity lut) = [7,9, 4,6]; offsets = row_offsets [0,2].
        assert_eq!(bufs.tok_bufs.len(), 1);
        let (tname, tdata, toff) = &bufs.tok_bufs[0];
        assert_eq!(*tname, "flank_tokens");
        assert_eq!(tdata.to_vec(), vec![7u8, 9, 4, 6]);
        assert_eq!(toff.to_vec(), vec![0i64, 2]);
    }
}
