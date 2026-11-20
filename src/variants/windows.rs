//! Variant-windows / variants flat-buffer assembly cores (pure ndarray).
//! PyO3 lives in `crate::ffi`. Mirrors the Python helpers in
//! `_dataset/_flat_flanks.py` (`tokenize_alleles`, `_slice_flanks`,
//! `_assemble_alt_windows`, `compute_*`) — byte-identical by construction.
use ndarray::{Array1, ArrayView1};

/// Apply a 256-entry byte->token lookup table. `out[i] = lut[bytes[i]]`.
/// Mirrors numpy `lut[bytes]`. `Tok` is the token dtype (u8 or i32).
pub fn tokenize<Tok: Copy>(bytes: ArrayView1<u8>, lut: ArrayView1<Tok>) -> Array1<Tok> {
    let bytes_s = bytes.as_slice().expect("tokenize: bytes must be contiguous");
    let lut_s = lut.as_slice().expect("tokenize: lut must be contiguous");
    // One upfront assertion lets the compiler prove every `b as usize` (< 256) is
    // in-bounds for lut_s, eliminating the per-element bounds check.
    assert!(lut_s.len() >= 256, "tokenize: lut must have >= 256 entries");
    // Using raw slices instead of ArrayView1 removes the per-element ndarray stride
    // multiply (imul rax, stride) that appeared in the indexed loop. collect() uses
    // TrustedLen and pre-allocates, removing the per-element Vec capacity check.
    let out: Vec<Tok> = bytes_s.iter().map(|&b| lut_s[b as usize]).collect();
    Array1::from_vec(out)
}

/// Derive per-variant (f5, f3) fixed-`flank_len` flanks from a contiguous
/// per-variant window read `[start-L, end+L)`. `f5` = first `L` bytes of each
/// row, `f3` = last `L`. Both returned flat `(n*L,)`, variant-major. Mirrors
/// `_slice_flanks` (`f5 = data[rw_off[:-1,None]+cols]`,
/// `f3 = data[rw_off[1:,None]-L+cols]`).
#[cfg(test)] // unfused oracle for the fused kernels
pub fn slice_flanks(
    data: ArrayView1<u8>,
    rw_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<u8>) {
    let n = rw_off.len() - 1;
    // Hoist contiguous slices upfront: eliminates the per-element ndarray stride
    // multiply (imul) and bounds check (cmp/jae) that appeared in both inner
    // k-loops. Using raw &[u8]/&[i64] lets LLVM see the loop as a plain copy.
    let data_s = data.as_slice().expect("slice_flanks: data must be contiguous");
    let rw_off_s = rw_off.as_slice().expect("slice_flanks: rw_off must be contiguous");
    let mut f5: Vec<u8> = Vec::with_capacity(n * flank_len);
    let mut f3: Vec<u8> = Vec::with_capacity(n * flank_len);
    for i in 0..n {
        let s = rw_off_s[i] as usize;
        let e = rw_off_s[i + 1] as usize;
        // extend_from_slice replaces flank_len individual push calls with a
        // single slice-bounds check + memcpy, removing the per-byte capacity
        // check and enabling vectorisation.
        f5.extend_from_slice(&data_s[s..s + flank_len]);
        f3.extend_from_slice(&data_s[e - flank_len..e]);
    }
    (Array1::from_vec(f5), Array1::from_vec(f3))
}

/// Copy `dst.len()` reference bases starting at contig coordinate `from` into
/// `dst`, padding positions outside `[0, contig.len())` with `pad_char`
/// (absolute-coordinate padding, same as `reference::get_reference`).
#[inline]
fn copy_ref_padded(dst: &mut [u8], contig: &[u8], from: i64, pad_char: u8) {
    let n = dst.len() as i64;
    let clen = contig.len() as i64;
    let lo = from.clamp(0, clen);
    let hi = (from + n).clamp(0, clen);
    if hi <= lo {
        dst.fill(pad_char);
        return;
    }
    let d_lo = (lo - from) as usize;
    let d_hi = (hi - from) as usize;
    dst[..d_lo].fill(pad_char);
    dst[d_lo..d_hi].copy_from_slice(&contig[lo as usize..hi as usize]);
    dst[d_hi..].fill(pad_char);
}

#[inline(always)]
fn var_ref_end(start: i32, ilen: i32) -> i64 {
    start as i64 - (ilen.min(0) as i64) + 1
}

/// Append `n` reference bases starting at contig coordinate `from` to `buf`
/// (padded like [`copy_ref_padded`]).
#[inline]
fn push_ref_padded(buf: &mut Vec<u8>, contig: &[u8], from: i64, n: usize, pad_char: u8) {
    let at = buf.len();
    buf.resize(at + n, 0);
    copy_ref_padded(&mut buf[at..], contig, from, pad_char);
}

/// Per-row inputs shared by the haplotype-flank helpers. All per-variant slices
/// are indexed by *selected* variant (already gathered).
struct RowCtx<'a> {
    contig: &'a [u8],
    starts: &'a [i32],
    ilens: &'a [i32],
    alt: &'a [u8],
    alt_off: &'a [i64],
    applied: &'a [bool],
    pad_char: u8,
}

impl RowCtx<'_> {
    /// Allele bytes of variant `k` as written into the haplotype. Empty-ALT pure
    /// deletions (SVAR2) get their anchor base `ref[start]` back, written into
    /// `anchor` (mirrors [`anchor_empty_del_alts`]).
    #[inline]
    fn allele<'s>(&'s self, k: usize, anchor: &'s mut [u8; 1]) -> &'s [u8] {
        let (a0, a1) = (self.alt_off[k] as usize, self.alt_off[k + 1] as usize);
        if a0 == a1 && self.ilens[k] < 0 {
            let p = self.starts[k] as i64;
            anchor[0] = if (0..self.contig.len() as i64).contains(&p) {
                self.contig[p as usize]
            } else {
                self.pad_char
            };
            &anchor[..]
        } else {
            &self.alt[a0..a1]
        }
    }

    /// Flanks of a focal variant `i` the haplotype does NOT carry (shadowed by an
    /// overlapping earlier ALT). The focal variant is forced in and conflicting
    /// neighbours are dropped: walk outward over applied neighbours until `L`
    /// bases are filled on each side. `O(L + k)`; only used for shadowed variants.
    fn walk(&self, i: usize, rs: usize, re: usize, f5: &mut [u8], f3: &mut [u8]) {
        let l = f5.len();
        let (starts, ilens) = (self.starts, self.ilens);
        let mut anchor = [0u8; 1];

        // 5': fill right-to-left, walking back from start_i.
        let mut rem = l;
        let mut cur = starts[i] as i64;
        let mut j = i;
        while rem > 0 {
            let mut found = None;
            while j > rs {
                j -= 1;
                if self.applied[j] && var_ref_end(starts[j], ilens[j]) <= cur {
                    found = Some(j);
                    break;
                }
            }
            let gap = match found {
                Some(k) => cur - var_ref_end(starts[k], ilens[k]),
                None => i64::MAX,
            };
            let n_ref = (gap as u64).min(rem as u64) as usize;
            copy_ref_padded(&mut f5[rem - n_ref..rem], self.contig, cur - n_ref as i64, self.pad_char);
            rem -= n_ref;
            let Some(k) = found else { break };
            if rem == 0 {
                break;
            }
            let a = self.allele(k, &mut anchor);
            let take = a.len().min(rem);
            f5[rem - take..rem].copy_from_slice(&a[a.len() - take..]);
            rem -= take;
            cur = starts[k] as i64;
        }

        // 3': fill left-to-right, walking forward from end_i.
        let mut pos = 0usize;
        let mut cur = var_ref_end(starts[i], ilens[i]);
        let mut j = i + 1;
        while pos < l {
            let mut found = None;
            while j < re {
                let k = j;
                j += 1;
                if self.applied[k] && starts[k] as i64 >= cur {
                    found = Some(k);
                    break;
                }
            }
            let gap = match found {
                Some(k) => starts[k] as i64 - cur,
                None => i64::MAX,
            };
            let n_ref = (gap as u64).min((l - pos) as u64) as usize;
            copy_ref_padded(&mut f3[pos..pos + n_ref], self.contig, cur, self.pad_char);
            pos += n_ref;
            let Some(k) = found else { break };
            if pos == l {
                break;
            }
            let a = self.allele(k, &mut anchor);
            let take = a.len().min(l - pos);
            f3[pos..pos + take].copy_from_slice(&a[..take]);
            pos += take;
            cur = var_ref_end(starts[k], ilens[k]);
        }
    }
}

/// Haplotype ("personalized") per-variant flanks: for each selected variant, the
/// `L` *haplotype* bases immediately 5' and 3' of it, with the other ALT calls
/// in the same `(instance, ploid)` row applied. Returns one flat buffer laid out
/// `[f5 | f3]` per variant (`2L` bytes each, variant-major) -- the exact layout
/// of the ride-along `flank_tokens` before tokenization.
///
/// Rows (`row_offsets`, len `rows+1`) must be position-sorted, as they are
/// everywhere variants are gathered. Which neighbours count as applied mirrors
/// `reconstruct_haplotype_core`: a single greedy left-to-right pass where a
/// variant overlapping an already-applied one is skipped ("first ALT wins").
/// So for every variant the haplotype carries, `f5 . ALT . f3` is exactly the
/// reconstructed haplotype around it. A shadowed variant gets its flanks with
/// itself forced in and the conflicting neighbours dropped.
///
/// Cost: per row, one pass builds a *gap-compressed* haplotype: applied ALTs
/// separated by their reference gaps, where any gap longer than `2L` is cut to
/// its first `L` and last `L` bases (no flank can see further). Every flank is
/// then a single `L`-byte slice of that buffer, so the work is `~(2L + |ALT|)`
/// bytes per variant -- the same as the reference path's `[start-L, end+L)`
/// read -- and dense rows are *cheaper* because short gaps are copied once.
///
/// Empty-ALT pure deletions (SVAR2) are re-anchored with `ref[start]`.
/// `alt_data`/`alt_off`/`starts_v`/`ilens_v`/`v_contigs` are per *selected*
/// variant (already gathered).
#[allow(clippy::too_many_arguments)]
pub fn haplotype_flanks<Tok: Copy>(
    row_offsets: ArrayView1<i64>,
    v_contigs: ArrayView1<i32>,
    starts_v: ArrayView1<i32>,
    ilens_v: ArrayView1<i32>,
    alt_data: ArrayView1<u8>,
    alt_off: ArrayView1<i64>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    flank_len: usize,
    lut: &[Tok],
) -> Array1<Tok> {
    assert!(lut.len() >= 256, "haplotype_flanks: lut must have >= 256 entries");
    let n = starts_v.len();
    let l = flank_len;
    let mut out: Vec<Tok> = vec![lut[pad_char as usize]; n * 2 * l];
    if n == 0 || l == 0 {
        return Array1::from_vec(out);
    }
    let ref_s = reference.as_slice().expect("haplotype_flanks: reference must be contiguous");
    let rof = ref_offsets.as_slice().expect("haplotype_flanks: ref_offsets must be contiguous");
    let alt_s = alt_data.as_slice().expect("haplotype_flanks: alt_data must be contiguous");
    let aof = alt_off.as_slice().expect("haplotype_flanks: alt_off must be contiguous");
    let starts = starts_v.as_slice().expect("haplotype_flanks: starts must be contiguous");
    let ilens = ilens_v.as_slice().expect("haplotype_flanks: ilens must be contiguous");
    let contigs = v_contigs.as_slice().expect("haplotype_flanks: contigs must be contiguous");
    let rows = row_offsets.as_slice().expect("haplotype_flanks: row_offsets must be contiguous");

    let mut applied = vec![false; n];
    // hap-buffer position of each applied variant's allele start (reused per row)
    let mut hpos: Vec<usize> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    // `buf` LUT-mapped once per row; flanks are copied out of it as tokens.
    let mut tbuf: Vec<Tok> = Vec::new();
    let mut walk_buf = vec![0u8; 2 * l];
    let mut anchor = [0u8; 1];
    for r in 0..rows.len().saturating_sub(1) {
        let (rs, re) = (rows[r] as usize, rows[r + 1] as usize);
        if re <= rs {
            continue;
        }
        let c = contigs[rs] as usize;
        let contig = &ref_s[rof[c] as usize..rof[c + 1] as usize];

        // Greedy forward pass: which variants does this haplotype carry?
        let mut cursor = i64::MIN;
        for j in rs..re {
            if starts[j] as i64 >= cursor {
                applied[j] = true;
                cursor = var_ref_end(starts[j], ilens[j]);
            }
        }
        let ctx = RowCtx {
            contig,
            starts,
            ilens,
            alt: alt_s,
            alt_off: aof,
            applied: &applied,
            pad_char,
        };

        // Build the gap-compressed haplotype for this row.
        buf.clear();
        hpos.clear();
        hpos.resize(re - rs, 0);
        let mut cur = i64::MIN; // ref coordinate just past the last applied ALT
        for j in rs..re {
            if !applied[j] {
                continue;
            }
            let s = starts[j] as i64;
            if cur == i64::MIN {
                push_ref_padded(&mut buf, contig, s - l as i64, l, pad_char);
            } else {
                let gap = (s - cur) as usize;
                if gap <= 2 * l {
                    push_ref_padded(&mut buf, contig, cur, gap, pad_char);
                } else {
                    push_ref_padded(&mut buf, contig, cur, l, pad_char);
                    push_ref_padded(&mut buf, contig, s - l as i64, l, pad_char);
                }
            }
            hpos[j - rs] = buf.len();
            buf.extend_from_slice(ctx.allele(j, &mut anchor));
            cur = var_ref_end(starts[j], ilens[j]);
        }
        push_ref_padded(&mut buf, contig, cur, l, pad_char);
        tbuf.clear();
        tbuf.extend(buf.iter().map(|&b| lut[b as usize]));

        // Slice every applied variant's flanks; walk only for shadowed ones.
        for j in rs..re {
            let (f5, f3) = out[j * 2 * l..(j + 1) * 2 * l].split_at_mut(l);
            if applied[j] {
                let p = hpos[j - rs];
                let q = p + ctx.allele(j, &mut anchor).len();
                f5.copy_from_slice(&tbuf[p - l..p]);
                f3.copy_from_slice(&tbuf[q..q + l]);
            } else {
                let (w5, w3) = walk_buf.split_at_mut(l);
                ctx.walk(j, rs, re, w5, w3);
                for (d, &b) in f5.iter_mut().zip(w5.iter()) {
                    *d = lut[b as usize];
                }
                for (d, &b) in f3.iter_mut().zip(w3.iter()) {
                    *d = lut[b as usize];
                }
            }
        }
    }
    Array1::from_vec(out)
}

/// A tokenized ragged buffer: (tokens, int64 offsets).
type TokBuf<Tok> = (Array1<Tok>, Array1<i64>);

/// `variant-windows` with haplotype flanks: from the `[f5 | f3]` buffer of
/// [`haplotype_flanks`], assemble `ref_window = f5 . REF . f3` (REF read
/// straight from the reference) and/or `alt_window = f5 . ALT . f3` (empty-ALT
/// deletions re-anchored). Tokenizes while copying (one pass, no reference
/// window fetch). Returns `(ref_window, alt_window)` as `(tokens, offsets)`.
#[allow(clippy::too_many_arguments)]
fn hap_windows<Tok: Copy>(
    flanks: &[Tok],
    want_ref: bool,
    want_alt: bool,
    v_contigs: &[i32],
    starts: &[i32],
    ilens: &[i32],
    alt: &[u8],
    alt_off: &[i64],
    reference: &[u8],
    ref_offsets: &[i64],
    pad_char: u8,
    l: usize,
    lut: &[Tok],
) -> (Option<TokBuf<Tok>>, Option<TokBuf<Tok>>) {
    assert!(lut.len() >= 256, "hap_windows: lut must have >= 256 entries");
    let n = starts.len();
    let tok = |dst: &mut Vec<Tok>, src: &[u8]| dst.extend(src.iter().map(|&b| lut[b as usize]));
    let contig_of = |i: usize| {
        let c = v_contigs[i] as usize;
        &reference[ref_offsets[c] as usize..ref_offsets[c + 1] as usize]
    };
    let mut scratch: Vec<u8> = Vec::new();

    let ref_w = want_ref.then(|| {
        let mut off = Vec::with_capacity(n + 1);
        off.push(0i64);
        let mut total = 0i64;
        for i in 0..n {
            total += 2 * l as i64 + (var_ref_end(starts[i], ilens[i]) - starts[i] as i64);
            off.push(total);
        }
        let mut data: Vec<Tok> = Vec::with_capacity(total as usize);
        for i in 0..n {
            let fl = &flanks[i * 2 * l..(i + 1) * 2 * l];
            let s = starts[i] as i64;
            let len = (var_ref_end(starts[i], ilens[i]) - s) as usize;
            scratch.clear();
            push_ref_padded(&mut scratch, contig_of(i), s, len, pad_char);
            data.extend_from_slice(&fl[..l]);
            tok(&mut data, &scratch);
            data.extend_from_slice(&fl[l..]);
        }
        (Array1::from_vec(data), Array1::from_vec(off))
    });

    let alt_w = want_alt.then(|| {
        let mut off = Vec::with_capacity(n + 1);
        off.push(0i64);
        let mut total = 0i64;
        for i in 0..n {
            let a_len = alt_off[i + 1] - alt_off[i];
            let a_len = if a_len == 0 && ilens[i] < 0 { 1 } else { a_len };
            total += 2 * l as i64 + a_len;
            off.push(total);
        }
        let mut data: Vec<Tok> = Vec::with_capacity(total as usize);
        for i in 0..n {
            let fl = &flanks[i * 2 * l..(i + 1) * 2 * l];
            let (a0, a1) = (alt_off[i] as usize, alt_off[i + 1] as usize);
            data.extend_from_slice(&fl[..l]);
            if a0 == a1 && ilens[i] < 0 {
                scratch.clear();
                push_ref_padded(&mut scratch, contig_of(i), starts[i] as i64, 1, pad_char);
                tok(&mut data, &scratch);
            } else {
                tok(&mut data, &alt[a0..a1]);
            }
            data.extend_from_slice(&fl[l..]);
        }
        (Array1::from_vec(data), Array1::from_vec(off))
    });
    (ref_w, alt_w)
}

/// Append `n` LUT-mapped reference bases starting at contig coordinate `from`.
/// Positions outside `[0, contig.len())` get `pad_tok` (= `lut[pad_char]`), so
/// the result equals `tokenize(copy_ref_padded(..))` without the byte buffer.
#[inline]
fn push_ref_tok<Tok: Copy>(
    out: &mut Vec<Tok>,
    contig: &[u8],
    from: i64,
    n: usize,
    pad_tok: Tok,
    lut: &[Tok],
) {
    let clen = contig.len() as i64;
    let to = from + n as i64;
    let lo = from.clamp(0, clen);
    let hi = to.clamp(0, clen);
    if hi <= lo {
        out.extend(std::iter::repeat_n(pad_tok, n));
        return;
    }
    out.extend(std::iter::repeat_n(pad_tok, (lo - from) as usize));
    out.extend(contig[lo as usize..hi as usize].iter().map(|&b| lut[b as usize]));
    out.extend(std::iter::repeat_n(pad_tok, (to - hi) as usize));
}

/// Reference flanks in the interleaved `[f5 | f3]` layout (2L per variant,
/// variant-major), LUT-mapped while copying: `f5 = ref[start-L, start)`,
/// `f3 = ref[end, end+L)`. One pass, no intermediate window read; the REF span
/// between the flanks (long for big deletions) is never touched. Pass an
/// identity LUT to get raw bytes. Equal to `fetch_windows` + `slice_flanks` +
/// interleave + `tokenize`.
#[allow(clippy::too_many_arguments)]
pub fn ref_flanks<Tok: Copy>(
    v_contigs: &[i32],
    starts: &[i32],
    ilens: &[i32],
    reference: &[u8],
    ref_offsets: &[i64],
    pad_char: u8,
    l: usize,
    lut: &[Tok],
) -> Array1<Tok> {
    assert!(lut.len() >= 256, "ref_flanks: lut must have >= 256 entries");
    let pad_tok = lut[pad_char as usize];
    let n = starts.len();
    let mut out: Vec<Tok> = Vec::with_capacity(n * 2 * l);
    for i in 0..n {
        let c = v_contigs[i] as usize;
        let contig = &reference[ref_offsets[c] as usize..ref_offsets[c + 1] as usize];
        let s = starts[i] as i64;
        push_ref_tok(&mut out, contig, s - l as i64, l, pad_tok, lut);
        push_ref_tok(&mut out, contig, var_ref_end(starts[i], ilens[i]), l, pad_tok, lut);
    }
    Array1::from_vec(out)
}

/// Tokenized reference windows `ref[start-L, end+L)` per variant, LUT-mapped
/// while copying. Equal to `tokenize(fetch_windows(..).0)` with the same offsets.
#[allow(clippy::too_many_arguments)]
fn ref_window_tokens<Tok: Copy>(
    v_contigs: &[i32],
    starts: &[i32],
    ilens: &[i32],
    reference: &[u8],
    ref_offsets: &[i64],
    pad_char: u8,
    l: usize,
    lut: &[Tok],
) -> TokBuf<Tok> {
    assert!(lut.len() >= 256, "ref_window_tokens: lut must have >= 256 entries");
    let pad_tok = lut[pad_char as usize];
    let n = starts.len();
    let mut off: Vec<i64> = Vec::with_capacity(n + 1);
    off.push(0);
    let mut total = 0i64;
    for i in 0..n {
        total += 2 * l as i64 + (var_ref_end(starts[i], ilens[i]) - starts[i] as i64);
        off.push(total);
    }
    let mut out: Vec<Tok> = Vec::with_capacity(total as usize);
    for i in 0..n {
        let c = v_contigs[i] as usize;
        let contig = &reference[ref_offsets[c] as usize..ref_offsets[c + 1] as usize];
        let from = starts[i] as i64 - l as i64;
        let len = (off[i + 1] - off[i]) as usize;
        push_ref_tok(&mut out, contig, from, len, pad_tok, lut);
    }
    (Array1::from_vec(out), Array1::from_vec(off))
}

/// Identity byte LUT, for kernel calls that want raw bytes.
#[cfg(test)]
fn identity_lut() -> [u8; 256] {
    let mut t = [0u8; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = i as u8;
    }
    t
}

/// Concatenate `flank5 . alt . flank3` per variant into a flat byte buffer.
/// `f5`/`f3` are `(n*flank_len,)` variant-major. Mirrors numba
/// `_assemble_alt_windows`. Returns `(out_bytes, out_offsets)`.
#[cfg(test)] // unfused oracle for the fused kernels
pub fn assemble_alt_window(
    f5: ArrayView1<u8>,
    f3: ArrayView1<u8>,
    alt_data: ArrayView1<u8>,
    alt_seq_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<i64>) {
    let n = alt_seq_off.len() - 1;
    // Hoist contiguous slices upfront: eliminates per-element ndarray stride
    // multiply (imul) and bounds checks (cmp/jae) in both the offset-build loop
    // and the assembly loop. Raw &[T] lets LLVM see the inner copies as plain
    // memcpy, matching the slice_flanks pattern already applied to this file.
    let f5_s = f5.as_slice().expect("assemble_alt_window: f5 must be contiguous");
    let f3_s = f3.as_slice().expect("assemble_alt_window: f3 must be contiguous");
    let alt_data_s =
        alt_data.as_slice().expect("assemble_alt_window: alt_data must be contiguous");
    let alt_seq_off_s =
        alt_seq_off.as_slice().expect("assemble_alt_window: alt_seq_off must be contiguous");

    let mut out_off: Vec<i64> = Vec::with_capacity(n + 1);
    out_off.push(0);
    for i in 0..n {
        let alt_len = alt_seq_off_s[i + 1] - alt_seq_off_s[i];
        out_off.push(out_off[i] + 2 * flank_len as i64 + alt_len);
    }
    let total = out_off[n] as usize;
    let mut out: Vec<u8> = Vec::with_capacity(total);
    for i in 0..n {
        // extend_from_slice: single bounds check + memcpy, not per-byte push.
        out.extend_from_slice(&f5_s[i * flank_len..(i + 1) * flank_len]);
        let a = alt_seq_off_s[i] as usize;
        let b = alt_seq_off_s[i + 1] as usize;
        out.extend_from_slice(&alt_data_s[a..b]);
        out.extend_from_slice(&f3_s[i * flank_len..(i + 1) * flank_len]);
    }
    (Array1::from_vec(out), Array1::from_vec(out_off))
}

/// Re-insert the anchor base into empty-ALT pure deletions before alt-window
/// assembly. SVAR2 stores an atomized pure DEL with an empty ALT (genoray keeps
/// no anchor in the key) at the VCF anchor position, so `[start, end)` spans the
/// anchor plus the deleted bases exactly as for SVAR1 (`GTA>G`). Haplotype
/// reconstruction keeps the anchor (`ref[pos]`), so the alt window must too, or
/// `flank5 . alt . flank3` silently drops a base and disagrees with both the
/// haplotype and the SVAR1 alt window. The anchor is the first byte of each
/// variant's ref-window span, `rw_data[rw_off[i] + L]`.
///
/// Returns `None` when no variant needs re-anchoring (the common SVAR1 case), so
/// callers keep the gathered buffers without a copy. Mirrors
/// `_anchor_empty_del_alts` in `_dataset/_flat_flanks.py`.
#[cfg(test)] // unfused oracle for the fused kernels
pub fn anchor_empty_del_alts(
    alt_data: ArrayView1<u8>,
    alt_seq_off: ArrayView1<i64>,
    ilens_v: ArrayView1<i32>,
    rw_data: ArrayView1<u8>,
    rw_off: ArrayView1<i64>,
    flank_len: usize,
) -> Option<(Array1<u8>, Array1<i64>)> {
    let n = alt_seq_off.len() - 1;
    let needs = |i: usize| alt_seq_off[i + 1] == alt_seq_off[i] && ilens_v[i] < 0;
    if !(0..n).any(needs) {
        return None;
    }
    let alt_s = alt_data.as_slice().expect("anchor_empty_del_alts: alt_data must be contiguous");
    let rw_s = rw_data.as_slice().expect("anchor_empty_del_alts: rw_data must be contiguous");
    let mut out: Vec<u8> = Vec::with_capacity(alt_s.len() + n);
    let mut off: Vec<i64> = Vec::with_capacity(n + 1);
    off.push(0);
    for i in 0..n {
        if needs(i) {
            out.push(rw_s[rw_off[i] as usize + flank_len]);
        } else {
            out.extend_from_slice(&alt_s[alt_seq_off[i] as usize..alt_seq_off[i + 1] as usize]);
        }
        off.push(out.len() as i64);
    }
    Some((Array1::from_vec(out), Array1::from_vec(off)))
}

/// Fetch the per-variant reference window `[start-L, end+L)` into one flat
/// buffer, with `ends = starts - min(ilen, 0) + 1`. Returns `(data, rw_off)`
/// where `rw_off` are per-variant byte boundaries (len `n+1`). Reuses
/// `reference::get_reference`'s padded core (absolute-coordinate OOB padding).
/// Mirrors `reference.fetch(v_contigs, starts-L, ends+L)`.
#[cfg(test)] // unfused oracle for the fused kernels
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
    let mut regions = ndarray::Array2::<i32>::zeros((n, 3));
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
        None,  // to_rc: window/flank fetch is always forward; strand RC handled elsewhere
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
    hap_flanks: bool,
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
        let l = flank_len as usize;
        // [f5 | f3] tokens per variant (2L, variant-major).
        let tok = if hap_flanks {
            // alt bytes are already gathered per selected variant (byte_bufs[0]);
            // no reference window fetch is needed.
            let (_, a_data, a_off) = &byte_bufs[0];
            haplotype_flanks(
                row_offsets, v_contigs, starts_v.view(), ilens_v.view(), a_data.view(),
                a_off.view(), reference, ref_offsets, pad_char, l,
                lut.as_slice().expect("lut must be contiguous"),
            )
        } else {
            // Read straight from the reference into the interleaved layout,
            // tokenizing while copying.
            ref_flanks(
                v_contigs.as_slice().expect("v_contigs must be contiguous"),
                starts_v.as_slice().unwrap(),
                ilens_v.as_slice().unwrap(),
                reference.as_slice().expect("reference must be contiguous"),
                ref_offsets.as_slice().expect("ref_offsets must be contiguous"),
                pad_char,
                l,
                lut.as_slice().expect("lut must be contiguous"),
            )
        };
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
    row_offsets: ArrayView1<i64>,
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
    hap_flanks: bool,
) -> VariantBufs<Tok> {
    let mut tok_bufs = Vec::new();
    let l = flank_len as usize;

    // alt alleles are always gathered (needed for alt window or bare alt).
    let (alt_data, alt_seq_off) =
        crate::variants::gather_alleles(v_idxs, alt_global, alt_off_global);

    let (ref_w, alt_w) = if ref_mode == 1 || alt_mode == 1 {
        let (starts_v, ilens_v) = gather_starts_ilens(v_idxs, v_starts, ilens);
        let (contigs_s, starts_s, ilens_s) = (
            v_contigs.as_slice().expect("v_contigs must be contiguous"),
            starts_v.as_slice().unwrap(),
            ilens_v.as_slice().unwrap(),
        );
        let ref_s = reference.as_slice().expect("reference must be contiguous");
        let roff_s = ref_offsets.as_slice().expect("ref_offsets must be contiguous");
        let lut_s = lut.as_slice().expect("lut must be contiguous");
        // [f5 | f3] flank tokens: from the per-row haplotype, or straight from
        // the reference. Skipped for a reference ref_window alone.
        let flanks = if hap_flanks {
            Some(haplotype_flanks(
                row_offsets, v_contigs, starts_v.view(), ilens_v.view(), alt_data.view(),
                alt_seq_off.view(), reference, ref_offsets, pad_char, l, lut_s,
            ))
        } else if alt_mode == 1 {
            Some(ref_flanks(contigs_s, starts_s, ilens_s, ref_s, roff_s, pad_char, l, lut_s))
        } else {
            None
        };
        // A reference ref_window is one contiguous read of [start-L, end+L).
        let ref_direct = (!hap_flanks && ref_mode == 1).then(|| {
            ref_window_tokens(contigs_s, starts_s, ilens_s, ref_s, roff_s, pad_char, l, lut_s)
        });
        // Remaining windows: f5 . REF . f3 and/or f5 . ALT . f3, tokenized while
        // copying (empty-ALT deletions re-anchored to ref[start]).
        let (ref_hw, alt_hw) = match &flanks {
            Some(fl) => hap_windows(
                fl.as_slice().unwrap(),
                hap_flanks && ref_mode == 1,
                alt_mode == 1,
                contigs_s,
                starts_s,
                ilens_s,
                alt_data.as_slice().unwrap(),
                alt_seq_off.as_slice().unwrap(),
                ref_s,
                roff_s,
                pad_char,
                l,
                lut_s,
            ),
            None => (None, None),
        };
        (ref_direct.or(ref_hw), alt_hw)
    } else {
        (None, None)
    };

    // ref side first (Python field insertion order), then alt side.
    if let Some((d, o)) = ref_w {
        tok_bufs.push(("ref_window", d, o));
    } else if ref_mode == 2 {
        let rg = ref_global.expect("bare ref allele needs ref byte buffer");
        let ro = ref_off_global.expect("bare ref allele needs ref offsets");
        let (ref_data, ref_seq_off) = crate::variants::gather_alleles(v_idxs, rg, ro);
        tok_bufs.push(("ref", tokenize(ref_data.view(), lut), ref_seq_off));
    }
    if let Some((d, o)) = alt_w {
        tok_bufs.push(("alt_window", d, o));
    } else if alt_mode == 2 {
        tok_bufs.push(("alt", tokenize(alt_data.view(), lut), alt_seq_off));
    }

    VariantBufs { byte_bufs: Vec::new(), tok_bufs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Tiny deterministic LCG (no rand dev-dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// Random variants over two contigs (lengths 50, 30), including starts at
    /// both contig edges, deletions running past the contig end, and empty-ALT
    /// (SVAR2-style) pure deletions.
    #[allow(clippy::type_complexity)]
    fn random_case(
        seed: u64,
        n: usize,
    ) -> (Array1<u8>, Array1<i64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<u8>, Vec<i64>, Vec<u8>, Vec<i64>)
    {
        let mut r = Lcg(seed);
        let lens = [50usize, 30];
        let reference: Vec<u8> = (0..80).map(|_| b"ACGTN"[r.next(5) as usize]).collect();
        let ref_offsets = arr1(&[0i64, 50, 80]);
        let (mut contigs, mut starts, mut ilens) = (vec![], vec![], vec![]);
        let (mut alt, mut aoff, mut refa, mut roff) = (vec![], vec![0i64], vec![], vec![0i64]);
        for _ in 0..n {
            let c = r.next(2) as usize;
            let s = match r.next(6) {
                0 => 0,
                1 => lens[c] as i64 - 1,
                _ => r.next(lens[c] as u64) as i64,
            };
            let il = r.next(9) as i32 - 5; // -5..=3
            contigs.push(c as i32);
            starts.push(s as i32);
            ilens.push(il);
            let alen = if il < 0 && r.next(2) == 0 { 0 } else { 1 + il.max(0) as usize };
            for _ in 0..alen {
                alt.push(b"ACGT"[r.next(4) as usize]);
            }
            aoff.push(alt.len() as i64);
            refa.push(b'A');
            roff.push(refa.len() as i64);
        }
        (Array1::from_vec(reference), ref_offsets, contigs, starts, ilens, alt, aoff, refa, roff)
    }

    fn check_fused_matches_unfused<Tok: Copy + PartialEq + std::fmt::Debug>(lut: Array1<Tok>) {
        for seed in 0..40u64 {
            for &l in &[1usize, 4, 9] {
                let n = 25;
                let (reference, ref_offsets, contigs, starts, ilens, alt, aoff, refa, roff) =
                    random_case(seed, n);
                let (contigs, starts, ilens) =
                    (Array1::from_vec(contigs), Array1::from_vec(starts), Array1::from_vec(ilens));
                let (alt, aoff) = (Array1::from_vec(alt), Array1::from_vec(aoff));
                let (refa, roff) = (Array1::from_vec(refa), Array1::from_vec(roff));
                let v_idxs = Array1::from_vec((0..n as i32).collect());
                let rows = arr1(&[0i64, 10, n as i64]);
                let pad = b'N';

                // Unfused oracle: fetch -> slice -> (interleave | anchor + assemble) -> tokenize.
                let (rw, rw_off) = fetch_windows(
                    contigs.view(), starts.view(), ilens.view(), l as i64, reference.view(),
                    ref_offsets.view(), pad,
                );
                let (f5, f3) = slice_flanks(rw.view(), rw_off.view(), l);
                let mut fb = Vec::new();
                for i in 0..n {
                    fb.extend_from_slice(&f5.as_slice().unwrap()[i * l..(i + 1) * l]);
                    fb.extend_from_slice(&f3.as_slice().unwrap()[i * l..(i + 1) * l]);
                }
                let exp_flank = tokenize(Array1::from_vec(fb).view(), lut.view());
                let exp_rw = tokenize(rw.view(), lut.view());
                let anchored = anchor_empty_del_alts(
                    alt.view(), aoff.view(), ilens.view(), rw.view(), rw_off.view(), l,
                );
                let (a_d, a_o) = anchored.unwrap_or_else(|| (alt.clone(), aoff.clone()));
                let (ab, ao) = assemble_alt_window(f5.view(), f3.view(), a_d.view(), a_o.view(), l);
                let exp_aw = tokenize(ab.view(), lut.view());

                let got = assemble_variants_mode(
                    v_idxs.view(), rows.view(), alt.view(), aoff.view(), None, None, true,
                    l as i64, Some(lut.view()), contigs.view(), starts.view(), ilens.view(),
                    reference.view(), ref_offsets.view(), pad, false,
                );
                assert_eq!(got.tok_bufs[0].1, exp_flank, "flank_tokens seed={seed} l={l}");

                // Haplotype flanks: LUT applied while copying == tokenize(bytes).
                let hap_bytes = haplotype_flanks(
                    rows.view(), contigs.view(), starts.view(), ilens.view(), alt.view(),
                    aoff.view(), reference.view(), ref_offsets.view(), pad, l, &identity_lut(),
                );
                let got = assemble_variants_mode(
                    v_idxs.view(), rows.view(), alt.view(), aoff.view(), None, None, true,
                    l as i64, Some(lut.view()), contigs.view(), starts.view(), ilens.view(),
                    reference.view(), ref_offsets.view(), pad, true,
                );
                assert_eq!(
                    got.tok_bufs[0].1,
                    tokenize(hap_bytes.view(), lut.view()),
                    "hap flank_tokens seed={seed} l={l}"
                );

                for (rm, am) in [(1i64, 1i64), (1, 2), (2, 1), (2, 2)] {
                    let got = assemble_windows_mode(
                        v_idxs.view(), rows.view(), rm, am, alt.view(), aoff.view(),
                        Some(refa.view()), Some(roff.view()), l as i64, lut.view(),
                        contigs.view(), starts.view(), ilens.view(), reference.view(),
                        ref_offsets.view(), pad, false,
                    );
                    let names: Vec<_> = got.tok_bufs.iter().map(|b| b.0).collect();
                    let want_names = [
                        if rm == 1 { "ref_window" } else { "ref" },
                        if am == 1 { "alt_window" } else { "alt" },
                    ];
                    assert_eq!(names, want_names, "seed={seed} l={l} modes=({rm},{am})");
                    if rm == 1 {
                        assert_eq!(got.tok_bufs[0].1, exp_rw, "ref_window seed={seed} l={l}");
                        assert_eq!(got.tok_bufs[0].2, rw_off, "ref_window off seed={seed} l={l}");
                    }
                    if am == 1 {
                        assert_eq!(got.tok_bufs[1].1, exp_aw, "alt_window seed={seed} l={l}");
                        assert_eq!(got.tok_bufs[1].2, ao, "alt_window off seed={seed} l={l}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_fused_ref_path_matches_unfused_u8() {
        // A=0 C=1 G=2 T=3, everything else (N, pad) = 7.
        let mut lut = vec![7u8; 256];
        for (i, &b) in b"ACGT".iter().enumerate() {
            lut[b as usize] = i as u8;
        }
        check_fused_matches_unfused(Array1::from_vec(lut));
    }

    #[test]
    fn test_fused_ref_path_matches_unfused_i32() {
        // Distinct token for N vs other unknown bytes, so padding mistakes show.
        let mut lut = vec![300i32; 256];
        for (i, &b) in b"ACGT".iter().enumerate() {
            lut[b as usize] = i as i32;
        }
        lut[b'N' as usize] = 299;
        check_fused_matches_unfused(Array1::from_vec(lut));
    }

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
    fn test_anchor_empty_del_alts() {
        // L=1. var0: SNP alt "A"; var1: SVAR2 pure DEL (ilen -2, empty ALT) whose
        // ref window is [9 | 30 31 32 | 8] -> anchor 30; var2: empty ALT but
        // ilen 0 (not a DEL) stays empty.
        let alt_data = arr1(&[65u8]);
        let alt_seq_off = arr1(&[0i64, 1, 1, 1]);
        let ilens = arr1(&[0i32, -2, 0]);
        let rw = arr1(&[1u8, 2, 3, 9, 30, 31, 32, 8, 4, 5, 6]);
        let rw_off = arr1(&[0i64, 3, 8, 11]);
        let (d, o) = anchor_empty_del_alts(
            alt_data.view(),
            alt_seq_off.view(),
            ilens.view(),
            rw.view(),
            rw_off.view(),
            1,
        )
        .expect("a pure DEL needs re-anchoring");
        assert_eq!(d.to_vec(), vec![65u8, 30]);
        assert_eq!(o.to_vec(), vec![0i64, 1, 2, 2]);

        // SVAR1-style anchored DEL: nothing to do, no copy.
        let alt_data = arr1(&[30u8]);
        let alt_seq_off = arr1(&[0i64, 1]);
        let ilens = arr1(&[-2i32]);
        let rw = arr1(&[9u8, 30, 31, 32, 8]);
        let rw_off = arr1(&[0i64, 5]);
        assert!(anchor_empty_del_alts(
            alt_data.view(),
            alt_seq_off.view(),
            ilens.view(),
            rw.view(),
            rw_off.view(),
            1,
        )
        .is_none());
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
            false,
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
            false,
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
            false,
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

    /// Run `haplotype_flanks` on ref = 0..20 (one contig), pad = 255.
    /// `vars` = (start, ilen, alt bytes); `rows` = row offsets.
    fn hap_flanks(vars: &[(i32, i32, &[u8])], rows: &[i64], l: usize) -> (Vec<u8>, Vec<u8>) {
        let reference: Array1<u8> = Array1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let starts = Array1::from_vec(vars.iter().map(|v| v.0).collect());
        let ilens = Array1::from_vec(vars.iter().map(|v| v.1).collect());
        let mut alt = Vec::new();
        let mut off = vec![0i64];
        for v in vars {
            alt.extend_from_slice(v.2);
            off.push(alt.len() as i64);
        }
        let contigs = Array1::<i32>::zeros(vars.len());
        let fl = haplotype_flanks(
            arr1(rows).view(),
            contigs.view(),
            starts.view(),
            ilens.view(),
            Array1::from_vec(alt).view(),
            Array1::from_vec(off).view(),
            reference.view(),
            ref_offsets.view(),
            255,
            l,
            &identity_lut(),
        );
        // Split the interleaved [f5 | f3] per-variant layout.
        let (mut f5, mut f3) = (Vec::new(), Vec::new());
        for c in fl.as_slice().unwrap().chunks(2 * l) {
            f5.extend_from_slice(&c[..l]);
            f3.extend_from_slice(&c[l..]);
        }
        (f5, f3)
    }

    #[test]
    fn test_haplotype_flanks_snps() {
        // SNPs at 5 (->88) and 7 (->99), one row, L=3.
        let (f5, f3) = hap_flanks(&[(5, 0, &[88]), (7, 0, &[99])], &[0, 2], 3);
        assert_eq!(f5, vec![2, 3, 4, /* v@7 */ 4, 88, 6]);
        assert_eq!(f3, vec![6, 99, 8, /* v@7 */ 8, 9, 10]);
    }

    #[test]
    fn test_haplotype_flanks_rows_isolated() {
        // Same two SNPs but in separate rows: flanks are pure reference.
        let (f5, f3) = hap_flanks(&[(5, 0, &[88]), (7, 0, &[99])], &[0, 1, 2], 3);
        assert_eq!(f5, vec![2, 3, 4, 4, 5, 6]);
        assert_eq!(f3, vec![6, 7, 8, 8, 9, 10]);
    }

    #[test]
    fn test_haplotype_flanks_deletion_and_edge_pad() {
        // DEL at 3 (ilen -2, alt = anchor 100) spans [3,6); SNP at 7 (->200). L=4.
        let (f5, f3) = hap_flanks(&[(3, -2, &[100]), (7, 0, &[200])], &[0, 2], 4);
        // del f5: [-1,3) -> pad,0,1,2 ; SNP f5: ref6, DEL alt 100, ref 2, ref 1.
        assert_eq!(f5, vec![255, 0, 1, 2, 1, 2, 100, 6]);
        // del f3: from 6: ref6, SNP 200, ref8, ref9 ; SNP f3: 8..12.
        assert_eq!(f3, vec![6, 200, 8, 9, 8, 9, 10, 11]);
    }

    #[test]
    fn test_haplotype_flanks_empty_alt_deletion_reanchored() {
        // SVAR2 bare-ALT deletion: empty alt -> anchor ref[3] = 3 is restored.
        let (f5, _) = hap_flanks(&[(3, -2, &[]), (7, 0, &[200])], &[0, 2], 4);
        assert_eq!(&f5[4..], &[1, 2, 3, 6]);
    }

    #[test]
    fn test_haplotype_flanks_insertion() {
        // INS at 2 (alt = anchor 2 + 50,51), SNP at 4. L=3.
        let (f5, f3) = hap_flanks(&[(2, 2, &[2, 50, 51]), (4, 0, &[99])], &[0, 2], 3);
        assert_eq!(&f5[3..], &[50, 51, 3]);
        assert_eq!(&f3[..3], &[3, 99, 5]);
    }

    #[test]
    fn test_haplotype_flanks_shadowed_overlap() {
        // DEL at 3 (ilen -3) spans [3,7); SNP at 5 is shadowed (first ALT wins).
        let (f5, f3) = hap_flanks(&[(3, -3, &[100]), (5, 0, &[99])], &[0, 2], 2);
        // DEL ignores the shadowed SNP: f3 = ref[7,8].
        assert_eq!(&f3[..2], &[7, 8]);
        // Shadowed SNP as focal: the conflicting DEL is dropped -> ref flanks.
        assert_eq!(&f5[2..], &[3, 4]);
        assert_eq!(&f3[2..], &[6, 7]);
    }

    #[test]
    fn test_haplotype_flanks_long_neighbour_allele_truncated() {
        // INS right before focal longer than L: only its last L bases are used.
        let (f5, _) = hap_flanks(&[(4, 4, &[4, 60, 61, 62, 63]), (5, 0, &[99])], &[0, 2], 3);
        assert_eq!(&f5[3..], &[61, 62, 63]);
    }
}
