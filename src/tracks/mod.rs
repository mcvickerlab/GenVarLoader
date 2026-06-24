//! Track-realignment PRNG primitives and insertion-fill strategies.
//!
//! PRNG functions mirror the numba implementations in
//! `python/genvarloader/_dataset/_tracks.py` (`_xorshift64`, `_hash4`) exactly.
//! All arithmetic is on `u64` with wrapping shifts/xors to match numba's
//! `np.uint64` overflow semantics.
//!
//! `apply_insertion_fill` mirrors `_apply_insertion_fill` in the same file
//! (lines 56-138), statement-by-statement, including float promotion points.

use ndarray::{ArrayView1, ArrayViewMut1};

// Strategy IDs — mirror _insertion_fill.py exactly.
pub const REPEAT_5P: i64 = 0;
pub const REPEAT_5P_NORM: i64 = 1;
pub const CONSTANT: i64 = 2;
pub const FLANK_SAMPLE: i64 = 3;
pub const INTERPOLATE: i64 = 4;

/// Single round of xorshift64.
///
/// Mirrors numba `_xorshift64` on `np.uint64`:
/// ```text
/// x ^= x << 13
/// x ^= x >> 7
/// x ^= x << 17
/// ```
/// Left shifts use `wrapping_shl` to replicate `np.uint64` truncation-to-64-bits.
#[inline(always)]
pub fn xorshift64(mut x: u64) -> u64 {
    x ^= x.wrapping_shl(13);
    x ^= x >> 7;
    x ^= x.wrapping_shl(17);
    x
}

/// Hash four `u64` values into one.
///
/// Mirrors numba `_hash4`:
/// ```text
/// h = a
/// h = xorshift64(h ^ b)
/// h = xorshift64(h ^ c)
/// h = xorshift64(h ^ d)
/// ```
#[inline(always)]
pub fn hash4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut h = a;
    h = xorshift64(h ^ b);
    h = xorshift64(h ^ c);
    h = xorshift64(h ^ d);
    h
}

/// Fill `writable_length` values starting at `out[out_idx]` using the given
/// insertion-fill strategy.
///
/// Mirrors numba `_apply_insertion_fill` (lines 56-138 of `_tracks.py`)
/// statement-by-statement, including float promotion points:
///
/// - `REPEAT_5P_NORM`: division is f32 / f32 (v_len cast to f32), result stored
///   as f32. Mirrors numba where `track` is f32 and `v_len` is an int —
///   numpy promotes f32/int → f32.
/// - `CONSTANT`: `params[0]` is f64; stored into f32 `out` (cast on store).
/// - `INTERPOLATE`: all anchor/Lagrange arithmetic in f64 (`xs`, `ys` are f64);
///   `ys[j] = track[ref_idx]` promotes f32 → f64 on assignment; final `acc`
///   stored into f32 `out` (cast on store).
///
/// # Parameters
/// - `out`: output track buffer (f32)
/// - `out_idx`: starting write index within `out`
/// - `writable_length`: number of positions to write
/// - `v_len`: total insertion length (v_diff + 1)
/// - `track`: reference track values (f32)
/// - `v_rel_pos`: variant position relative to the query region
/// - `strategy_id`: one of `REPEAT_5P`, `REPEAT_5P_NORM`, `CONSTANT`,
///   `FLANK_SAMPLE`, `INTERPOLATE`
/// - `params`: per-strategy parameter slot (f64); `params[0]` = flank_width,
///   constant value, or interpolation order depending on strategy
/// - `base_seed`, `query`, `hap`: seed components for `FLANK_SAMPLE`
pub fn apply_insertion_fill(
    out: &mut ArrayViewMut1<f32>,
    out_idx: usize,
    writable_length: usize,
    v_len: i64,
    track: ArrayView1<f32>,
    v_rel_pos: i64,
    strategy_id: i64,
    params: ArrayView1<f64>,
    base_seed: u64,
    query: u64,
    hap: u64,
) {
    let track_len = track.len() as i64;

    if strategy_id == REPEAT_5P {
        // Numba comment: "unreachable from outer kernel (which short-circuits this
        // strategy before calling). Kept for completeness and direct-helper-call safety."
        let val = track[v_rel_pos as usize];
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == REPEAT_5P_NORM {
        // Numba: val = track[v_rel_pos] / v_len
        // track is f32, v_len is int → numpy promotes f32/int → f32.
        // Mirror: cast v_len to f32, divide f32/f32 → f32.
        let val = track[v_rel_pos as usize] / (v_len as f32);
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == CONSTANT {
        // Numba: val = params[0] (f64), stored into f32 out on assignment.
        let val = params[0] as f32;
        for i in 0..writable_length {
            out[out_idx + i] = val;
        }
    } else if strategy_id == FLANK_SAMPLE {
        // Numba: width = np.int64(params[0])
        let width = params[0] as i64;
        let pool_lo = (v_rel_pos - width).max(0);
        let pool_hi = (v_rel_pos + width).min(track_len - 1);
        let pool_size = (pool_hi - pool_lo + 1) as u64;
        for i in 0..writable_length {
            // Numba: seed = _hash4(base_seed, np.uint64(query), np.uint64(hap), np.uint64(out_idx + i))
            let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
            // Numba: offset = np.int64(seed % np.uint64(pool_size))
            let offset = (seed % pool_size) as i64;
            out[out_idx + i] = track[(pool_lo + offset) as usize];
        }
    } else if strategy_id == INTERPOLATE {
        // Numba: order = np.int64(params[0])
        let order = params[0] as i64;
        // k = ceil((order+1)/2)
        // Numba: k = (order + 1 + 1) // 2
        let k = (order + 1 + 1) / 2;
        let n_anchors = (2 * k) as usize;

        // Anchors: xs and ys are f64 (numba: np.empty(..., dtype=np.float64))
        let mut xs = vec![0.0f64; n_anchors];
        let mut ys = vec![0.0f64; n_anchors];

        // 5' side: xs[j] = -j, ys[j] = track[max(v_rel_pos - j, 0)]
        // Numba: xs[j] = -float(j), ys[j] = track[ref_idx]
        // track[ref_idx] is f32; ys is f64 → f32 promoted to f64 on assignment.
        for j in 0..k as usize {
            let ref_idx = (v_rel_pos - j as i64).max(0) as usize;
            xs[j] = -(j as f64);
            ys[j] = track[ref_idx] as f64;
        }
        // 3' side: xs[k+j] = v_len + j, ys[k+j] = track[min(v_rel_pos+1+j, track_len-1)]
        // Numba: xs[k + j] = float(v_len) + float(j), ys[k + j] = track[ref_idx]
        for j in 0..k as usize {
            let ref_idx = (v_rel_pos + 1 + j as i64).min(track_len - 1) as usize;
            xs[k as usize + j] = (v_len as f64) + (j as f64);
            ys[k as usize + j] = track[ref_idx] as f64;
        }

        // Lagrange interpolation: mirror numba loop nesting exactly.
        // outer: a over n_anchors; inner: b over n_anchors, skip b==a
        for i in 0..writable_length {
            // Numba: x = float(i) — this is the insertion-local coordinate
            let x = i as f64;
            // Numba: acc = 0.0 (float64 literal)
            let mut acc = 0.0f64;
            for a in 0..n_anchors {
                // Numba: term = ys[a]
                let mut term = ys[a];
                for b in 0..n_anchors {
                    if b == a {
                        continue;
                    }
                    // Numba: term *= (x - xs[b]) / (xs[a] - xs[b])
                    term *= (x - xs[b]) / (xs[a] - xs[b]);
                }
                // Numba: acc += term
                acc += term;
            }
            // Numba: out[out_idx + i] = acc — f64 acc stored into f32 out
            out[out_idx + i] = acc as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Expected values hand-derived from the numba algorithm (verified by running
    /// the Python reference implementation with np.uint64 arithmetic).
    #[test]
    fn test_xorshift64_vectors() {
        // xorshift64(1):
        //   x=1; x ^= 1<<13=0x2000 → 0x2001
        //   x ^= 0x2001>>7=0x40   → 0x2041
        //   x ^= 0x2041<<17=0x408200000 → 0x40822041 = 1082269761
        assert_eq!(xorshift64(1), 1_082_269_761_u64);

        // xorshift64(2) = 2164539522 (verified via Python np.uint64)
        assert_eq!(xorshift64(2), 2_164_539_522_u64);

        // xorshift64(42) = 45454805674
        assert_eq!(xorshift64(42), 45_454_805_674_u64);

        // xorshift64(0xdeadbeef) = 4018790486776397394
        assert_eq!(xorshift64(0xdeadbeef), 4_018_790_486_776_397_394_u64);

        // xorshift64(u64::MAX) — wrapping behaviour: 2**64-1 = 0xffffffffffffffff
        // result = 0x3f801fc0 = 1065361344 (verified via Python np.uint64)
        assert_eq!(xorshift64(u64::MAX), 1_065_361_344_u64);
    }

    #[test]
    fn test_hash4_vectors() {
        // hash4(1,2,3,4) = 11323120931611735037 (verified via Python)
        assert_eq!(hash4(1, 2, 3, 4), 11_323_120_931_611_735_037_u64);

        // hash4(0,0,0,0): h=0; xorshift64(0)=0 at each step → 0
        assert_eq!(hash4(0, 0, 0, 0), 0_u64);

        // hash4(0xdeadbeef, 0xcafe, 0xbabe, 1) = 5244362157944750963
        assert_eq!(
            hash4(0xdeadbeef, 0xcafe, 0xbabe, 1),
            5_244_362_157_944_750_963_u64
        );
    }

    // ------------------------------------------------------------------ //
    // apply_insertion_fill tests                                           //
    // ------------------------------------------------------------------ //

    /// Helper: allocate out, run apply_insertion_fill, return the filled slice.
    fn run_fill(
        out_size: usize,
        out_idx: usize,
        writable_length: usize,
        v_len: i64,
        track: &[f32],
        v_rel_pos: i64,
        strategy_id: i64,
        params: &[f64],
        base_seed: u64,
        query: u64,
        hap: u64,
    ) -> Vec<f32> {
        let mut out_arr = Array1::<f32>::zeros(out_size);
        {
            let mut out_view = out_arr.view_mut();
            let track_arr = Array1::from_vec(track.to_vec());
            let params_arr = Array1::from_vec(params.to_vec());
            apply_insertion_fill(
                &mut out_view,
                out_idx,
                writable_length,
                v_len,
                track_arr.view(),
                v_rel_pos,
                strategy_id,
                params_arr.view(),
                base_seed,
                query,
                hap,
            );
        }
        out_arr.to_vec()
    }

    /// REPEAT_5P_NORM: val = track[v_rel_pos] / v_len (f32/f32 → f32).
    ///
    /// track = [1.0, 6.0, 2.0], v_rel_pos = 1 → track[1] = 6.0f32
    /// v_len = 3 → val = 6.0f32 / 3f32 = 2.0f32
    /// writable_length = 3 → out[0..3] = [2.0, 2.0, 2.0]
    /// sum = 6.0 = track[v_rel_pos] ✓ (sum-preserving)
    #[test]
    fn test_repeat_5p_norm() {
        let track = [1.0f32, 6.0, 2.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        // val = 6.0f32 / 3f32 = 2.0f32  (exact in f32)
        let expected_val = 6.0f32 / 3.0f32;
        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            REPEAT_5P_NORM,
            &[0.0],
            0,
            0,
            0,
        );
        assert_eq!(result.len(), writable_length);
        for &v in &result {
            assert_eq!(v, expected_val, "REPEAT_5P_NORM: expected {expected_val}, got {v}");
        }
        // Sum preservation check
        let sum: f32 = result.iter().sum();
        assert_eq!(sum, track[v_rel_pos as usize]);
    }

    /// REPEAT_5P_NORM with non-divisible values: verifies f32 precision.
    ///
    /// track = [0.0, 1.0, 0.0], v_rel_pos = 1, v_len = 3
    /// val = 1.0f32 / 3f32 (not exactly representable)
    #[test]
    fn test_repeat_5p_norm_precision() {
        let track = [0.0f32, 1.0, 0.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        let expected_val = 1.0f32 / 3.0f32; // same f32 division as numba
        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            REPEAT_5P_NORM,
            &[0.0],
            0,
            0,
            0,
        );
        for &v in &result {
            assert_eq!(v, expected_val);
        }
    }

    /// CONSTANT: fills every position with params[0] cast to f32.
    ///
    /// params[0] = 3.14 (f64), writable_length = 4
    /// expected: each position = 3.14f64 as f32 = 3.14f32
    #[test]
    fn test_constant() {
        let track = [0.0f32, 0.0, 0.0, 0.0, 0.0];
        let result = run_fill(5, 1, 4, 1, &track, 0, CONSTANT, &[3.14f64], 0, 0, 0);
        let expected = 3.14f64 as f32;
        for i in 1..5 {
            assert_eq!(result[i], expected, "CONSTANT at position {i}");
        }
        // position 0 should be untouched (still 0)
        assert_eq!(result[0], 0.0f32);
    }

    /// CONSTANT with NaN: the default Constant(value=NaN) should write NaN.
    #[test]
    fn test_constant_nan() {
        let track = [0.0f32];
        let result = run_fill(3, 0, 3, 1, &track, 0, CONSTANT, &[f64::NAN], 0, 0, 0);
        for &v in &result {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    /// FLANK_SAMPLE: deterministic given seed.
    ///
    /// Setup: track = [10.0, 20.0, 30.0, 40.0, 50.0], v_rel_pos=2, flank_width=1
    /// pool: pool_lo = max(0, 2-1)=1, pool_hi = min(4, 2+1)=3, pool_size=3
    /// pool values: track[1..=3] = [20.0, 30.0, 40.0]
    ///
    /// For base_seed=42, query=7, hap=1, out_idx=0, writable_length=4:
    ///
    /// Hand-derived using verified hash4:
    ///   i=0: seed = hash4(42, 7, 1, 0); offset = seed % 3; track[1+offset]
    ///   i=1: seed = hash4(42, 7, 1, 1); offset = seed % 3; track[1+offset]
    ///   i=2: seed = hash4(42, 7, 1, 2); offset = seed % 3; track[1+offset]
    ///   i=3: seed = hash4(42, 7, 1, 3); offset = seed % 3; track[1+offset]
    ///
    /// Computed by applying xorshift64 chain:
    ///   hash4(42, 7, 1, 0) = xorshift64(xorshift64(xorshift64(42^7) ^ 1) ^ 0)
    ///   We compute all hash values first and derive offsets below.
    #[test]
    fn test_flank_sample_deterministic() {
        let track = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let v_rel_pos = 2i64;
        let flank_width = 1i64; // pool_lo=1, pool_hi=3, pool_size=3
        let pool_lo = 1i64;
        let pool_size = 3u64;

        let base_seed = 42u64;
        let query = 7u64;
        let hap = 1u64;
        let out_idx = 0usize;
        let writable_length = 4;

        // Hand-compute the expected hash values and pool indices:
        // This uses our verified hash4 function.
        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
                let offset = (seed % pool_size) as i64;
                track[(pool_lo + offset) as usize]
            })
            .collect();

        let result = run_fill(
            writable_length,
            out_idx,
            writable_length,
            1,
            &track,
            v_rel_pos,
            FLANK_SAMPLE,
            &[flank_width as f64],
            base_seed,
            query,
            hap,
        );

        assert_eq!(result, expected, "FLANK_SAMPLE: result did not match expected");

        // Spot-check the first index by computing hash4 explicitly:
        // hash4(42, 7, 1, 0):
        //   h = 42
        //   h = xorshift64(42 ^ 7) = xorshift64(45) = ?
        let h0 = xorshift64(42 ^ 7); // xorshift64(45)
        let h1 = xorshift64(h0 ^ 1);
        let h2 = xorshift64(h1 ^ 0);
        let offset0 = (h2 % pool_size) as i64;
        assert_eq!(
            result[0],
            track[(pool_lo + offset0) as usize],
            "FLANK_SAMPLE spot-check i=0 failed"
        );
    }

    /// FLANK_SAMPLE with out_idx > 0: verifies that out_idx+i is used, not just i.
    #[test]
    fn test_flank_sample_out_idx_offset() {
        let track = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let v_rel_pos = 2i64;
        let flank_width = 1i64;
        let pool_lo = 1i64;
        let pool_size = 3u64;
        let base_seed = 100u64;
        let query = 3u64;
        let hap = 0u64;
        let out_idx = 5usize;
        let writable_length = 3;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let seed = hash4(base_seed, query, hap, (out_idx + i) as u64);
                let offset = (seed % pool_size) as i64;
                track[(pool_lo + offset) as usize]
            })
            .collect();

        let mut out_arr = Array1::<f32>::zeros(out_idx + writable_length);
        {
            let mut out_view = out_arr.view_mut();
            let track_arr = Array1::from_vec(track.to_vec());
            let params_arr = Array1::from_vec(vec![flank_width as f64]);
            apply_insertion_fill(
                &mut out_view,
                out_idx,
                writable_length,
                1,
                track_arr.view(),
                v_rel_pos,
                FLANK_SAMPLE,
                params_arr.view(),
                base_seed,
                query,
                hap,
            );
        }
        let result: Vec<f32> = out_arr.iter().skip(out_idx).cloned().collect();
        assert_eq!(result, expected, "FLANK_SAMPLE out_idx offset test failed");
    }

    /// INTERPOLATE order=1 (linear interpolation).
    ///
    /// order=1 → k = ceil(2/2) = 1, n_anchors = 2
    /// track = [0.0, 4.0, 8.0] (indices 0,1,2), v_rel_pos=1, v_len=3
    ///
    /// Anchors (5' then 3' side):
    ///   xs[0] = -0.0 = 0.0, ys[0] = track[max(1-0,0)=1] = 4.0
    ///   xs[1] = 3.0+0.0 = 3.0, ys[1] = track[min(1+1+0,2)=2] = 8.0
    ///
    /// Lagrange at x=0: term_0 = 4.0 * (0-3)/(0-3) = 4.0*(-3/-3) = 4.0*1.0 = 4.0
    ///                  term_1 = 8.0 * (0-0)/(3-0) = 8.0*0 = 0.0; acc=4.0
    /// Lagrange at x=1: term_0 = 4.0 * (1-3)/(0-3) = 4.0*(-2/-3) = 4.0*0.6667 = 2.6667
    ///                  term_1 = 8.0 * (1-0)/(3-0) = 8.0*(1/3) = 2.6667; acc=5.3333
    /// Lagrange at x=2: term_0 = 4.0 * (2-3)/(0-3) = 4.0*(1/3) = 1.3333
    ///                  term_1 = 8.0 * (2-0)/(3-0) = 8.0*(2/3) = 5.3333; acc=6.6667
    ///
    /// Check endpoints: at x=0 → 4.0 = track[1] ✓; at x=3 → 8.0 = track[2] ✓
    #[test]
    fn test_interpolate_order1() {
        let track = [0.0f32, 4.0, 8.0];
        let v_rel_pos = 1i64;
        let v_len = 3i64;
        let writable_length = 3;

        // Hand-computed Lagrange values (f64 arithmetic, stored to f32):
        // xs = [0.0, 3.0], ys = [4.0, 8.0]
        // x=0: acc = 4.0*(0-3)/(0-3) + 8.0*(0-0)/(3-0) = 4.0 + 0.0 = 4.0
        // x=1: acc = 4.0*(1-3)/(0-3) + 8.0*(1-0)/(3-0) = 4.0*(2/3) + 8.0*(1/3)
        //           = 8.0/3.0 + 8.0/3.0 = 16.0/3.0
        // x=2: acc = 4.0*(2-3)/(0-3) + 8.0*(2-0)/(3-0) = 4.0*(1/3) + 8.0*(2/3)
        //           = 4.0/3.0 + 16.0/3.0 = 20.0/3.0
        let xs = [0.0f64, 3.0f64];
        let ys = [4.0f64, 8.0f64];
        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..2usize {
                    let mut term = ys[a];
                    for b in 0..2usize {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[1.0f64], // order=1
            0,
            0,
            0,
        );

        assert_eq!(result.len(), writable_length);
        // Endpoint check: at i=0, result should equal ys[0]=track[v_rel_pos]=4.0
        assert_eq!(result[0], 4.0f32, "order=1 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=1 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE order=2.
    ///
    /// order=2 → k = ceil(3/2) = 2, n_anchors = 4
    /// track = [1.0, 2.0, 4.0, 8.0, 16.0], v_rel_pos=2, v_len=2
    ///
    /// Anchors:
    ///   5' side (j=0,1):
    ///     xs[0]=-0.0=0.0, ys[0]=track[max(2-0,0)=2]=4.0
    ///     xs[1]=-1.0,     ys[1]=track[max(2-1,0)=1]=2.0
    ///   3' side (j=0,1):
    ///     xs[2]=2.0+0.0=2.0, ys[2]=track[min(2+1+0,4)=3]=8.0
    ///     xs[3]=2.0+1.0=3.0, ys[3]=track[min(2+1+1,4)=4]=16.0
    ///
    /// Lagrange at x=0,1 hand-computed via the same formula.
    #[test]
    fn test_interpolate_order2() {
        let track = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        let v_rel_pos = 2i64;
        let v_len = 2i64;
        let writable_length = 2;

        // Anchors: xs=[0.0, -1.0, 2.0, 3.0], ys=[4.0, 2.0, 8.0, 16.0]
        let xs = [0.0f64, -1.0f64, 2.0f64, 3.0f64];
        let ys = [4.0f64, 2.0f64, 8.0f64, 16.0f64];
        let n = 4usize;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..n {
                    let mut term = ys[a];
                    for b in 0..n {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[2.0f64], // order=2
            0,
            0,
            0,
        );

        // At x=0, result should equal ys[0] = track[v_rel_pos] = 4.0
        assert_eq!(result[0], 4.0f32, "order=2 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=2 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE order=3.
    ///
    /// order=3 → k = ceil(4/2) = 2, n_anchors = 4 (same as order=2)
    /// (The numba formula k=(order+1+1)//2 gives k=2 for both order=2 and order=3)
    /// track = [3.0, 1.0, 5.0, 9.0, 2.0, 6.0], v_rel_pos=2, v_len=4
    ///
    /// Anchors:
    ///   5' side (j=0,1):
    ///     xs[0]=0.0, ys[0]=track[2]=5.0
    ///     xs[1]=-1.0, ys[1]=track[1]=1.0
    ///   3' side (j=0,1):
    ///     xs[2]=4.0, ys[2]=track[3]=9.0
    ///     xs[3]=5.0, ys[3]=track[4]=2.0
    #[test]
    fn test_interpolate_order3() {
        let track = [3.0f32, 1.0, 5.0, 9.0, 2.0, 6.0];
        let v_rel_pos = 2i64;
        let v_len = 4i64;
        let writable_length = 4;

        // k=2, n_anchors=4
        let xs = [0.0f64, -1.0f64, 4.0f64, 5.0f64];
        let ys = [5.0f64, 1.0f64, 9.0f64, 2.0f64];
        let n = 4usize;

        let expected: Vec<f32> = (0..writable_length)
            .map(|i| {
                let x = i as f64;
                let mut acc = 0.0f64;
                for a in 0..n {
                    let mut term = ys[a];
                    for b in 0..n {
                        if b == a { continue; }
                        term *= (x - xs[b]) / (xs[a] - xs[b]);
                    }
                    acc += term;
                }
                acc as f32
            })
            .collect();

        let result = run_fill(
            writable_length,
            0,
            writable_length,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[3.0f64], // order=3
            0,
            0,
            0,
        );

        // At x=0, result should equal track[v_rel_pos]=5.0
        assert_eq!(result[0], 5.0f32, "order=3 left endpoint must equal track[v_rel_pos]");
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "INTERPOLATE order=3 at i={i}: got {got}, expected {exp}");
        }
    }

    /// INTERPOLATE: verify that order=1 at x=v_len gives the 3' anchor value.
    ///
    /// With track=[2.0, 10.0, 6.0], v_rel_pos=1, v_len=2:
    ///   xs=[0.0, 2.0], ys=[10.0, 6.0]
    ///   At x=0: acc = 10.0*(0-2)/(0-2) + 6.0*(0-0)/(2-0) = 10.0 + 0.0 = 10.0 ✓
    ///   At x=1: acc = 10.0*(1-2)/(0-2) + 6.0*(1-0)/(2-0) = 10.0*0.5 + 6.0*0.5 = 8.0
    ///   (Note: x=v_len=2 would be exactly 6.0 but writable_length=2 so we test x=0,1)
    #[test]
    fn test_interpolate_order1_endpoints() {
        let track = [2.0f32, 10.0, 6.0];
        let v_rel_pos = 1i64;
        let v_len = 2i64;

        // writable_length = v_len = 2, covering x=0,1
        let result = run_fill(
            2,
            0,
            2,
            v_len,
            &track,
            v_rel_pos,
            INTERPOLATE,
            &[1.0f64],
            0,
            0,
            0,
        );

        // x=0 must equal track[v_rel_pos] = 10.0
        assert_eq!(result[0], 10.0f32, "left endpoint");

        // x=1: hand-computed
        // xs=[0.0, 2.0], ys=[10.0, 6.0]
        // term_0 = 10.0 * (1-2)/(0-2) = 10.0 * 0.5 = 5.0
        // term_1 = 6.0 * (1-0)/(2-0) = 6.0 * 0.5 = 3.0; acc=8.0
        let x = 1.0f64;
        let xs = [0.0f64, 2.0f64];
        let ys = [10.0f64, 6.0f64];
        let mut acc = 0.0f64;
        for a in 0..2 {
            let mut term = ys[a];
            for b in 0..2 {
                if b == a { continue; }
                term *= (x - xs[b]) / (xs[a] - xs[b]);
            }
            acc += term;
        }
        assert_eq!(result[1], acc as f32, "midpoint check");
    }

    /// REPEAT_5P: fills with track[v_rel_pos] directly.
    #[test]
    fn test_repeat_5p() {
        let track = [5.0f32, 11.0, 7.0];
        let v_rel_pos = 1i64;
        let result = run_fill(4, 0, 4, 4, &track, v_rel_pos, REPEAT_5P, &[0.0], 0, 0, 0);
        for &v in &result {
            assert_eq!(v, 11.0f32, "REPEAT_5P: expected 11.0");
        }
    }
}
