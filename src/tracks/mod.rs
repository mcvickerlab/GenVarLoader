//! Track-realignment PRNG primitives.
//!
//! Both functions mirror the numba implementations in
//! `python/genvarloader/_dataset/_tracks.py` (`_xorshift64`, `_hash4`) exactly.
//! All arithmetic is on `u64` with wrapping shifts/xors to match numba's
//! `np.uint64` overflow semantics.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
