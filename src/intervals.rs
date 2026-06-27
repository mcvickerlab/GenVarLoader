use ndarray::{ArrayView1, ArrayViewMut1};
use rayon::prelude::*;

/// Paint base-pair-resolution tracks from pre-sorted intervals.
///
/// Mirrors the numba kernel `intervals_to_tracks` exactly:
/// - Zeroes the entire `out` buffer first.
/// - Skips queries that have no intervals.
/// - Replicates numpy slice semantics: start is clamped to 0 (intervals may
///   begin before the query origin under max_jitter>0; see #242) and end is
///   clamped to `length`; no-op when `min(end, length) <= max(start, 0)`.
/// - Breaks out of the interval loop when `start >= length` (intervals are
///   sorted by start, so all subsequent intervals are also out of range).
/// - Values are copied (f32 → f32), never reduced.
///
/// When `parallel=true` the outer query loop is dispatched via rayon using the
/// split_at_mut cursor idiom (same as C1/C2) so per-query out slices are
/// provably disjoint — no raw `*mut` in the closure.
pub fn intervals_to_tracks(
    offset_idxs: ArrayView1<i64>,
    starts: ArrayView1<i32>,
    itv_starts: ArrayView1<i32>,
    itv_ends: ArrayView1<i32>,
    itv_values: ArrayView1<f32>,
    itv_offsets: ArrayView1<i64>,
    mut out: ArrayViewMut1<f32>,
    out_offsets: ArrayView1<i64>,
    parallel: bool,
) {
    // Hoist all inputs to raw slices before any loop — eliminates ndarray's
    // per-element stride multiplication and bounds-check branches that would
    // otherwise appear in every inner-loop iteration.
    let offset_idxs = offset_idxs.as_slice().unwrap();
    let starts = starts.as_slice().unwrap();
    let itv_starts = itv_starts.as_slice().unwrap();
    let itv_ends = itv_ends.as_slice().unwrap();
    let itv_values = itv_values.as_slice().unwrap();
    let itv_offsets = itv_offsets.as_slice().unwrap();
    let out_offsets = out_offsets.as_slice().unwrap();

    // Step 1: zero the whole output buffer, exactly like `out[:] = 0.0`.
    // The out buffer is freshly allocated and contiguous; address it as a raw
    // &mut [f32] so per-interval writes avoid ndarray SliceInfo construction.
    let out_slice = out.as_slice_mut().unwrap();
    out_slice.fill(0.0);

    let n_queries = starts.len();

    // Inner per-query paint logic. Takes a mutable slice for this query's
    // output region (already offset-addressed) plus the query index.
    // All read-only slices are captured by shared reference — they are
    // Send+Sync so this closure is safe to use in rayon.
    let paint_query = |query: usize, out_chunk: &mut [f32]| {
        let idx = offset_idxs[query] as usize;
        let itv_s = itv_offsets[idx] as usize;
        let itv_e = itv_offsets[idx + 1] as usize;

        if itv_s == itv_e {
            // No intervals for this query — out slice stays 0 (already zeroed).
            return;
        }

        let length = out_chunk.len() as i64;
        let query_start = starts[query] as i64;

        for interval in itv_s..itv_e {
            // start/end computed in i64 (avoids i32 overflow for large coords).
            let start = itv_starts[interval] as i64 - query_start;
            let end = itv_ends[interval] as i64 - query_start;
            let value = itv_values[interval];

            if start >= length {
                // start >= length: intervals are sorted, all remaining are
                // also out of range — break.
                break;
            }
            // Clip to the query window. Intervals may start before query_start
            // (jitter-expanded interval storage vs. the per-read query origin;
            // see issue #242) or end past it. Keep s/e as i64 until after the
            // guard so that negative values don't wrap when cast to usize.
            let s = start.max(0);
            let e = end.min(length);
            if e > s {
                out_chunk[s as usize..e as usize].fill(value);
            }
        }
    };

    if parallel {
        // Build disjoint per-query mutable slices using the split_at_mut
        // cursor idiom (mirrors C1 reconstruct_haplotypes_from_sparse).
        let bounds: Vec<(usize, usize)> = (0..n_queries)
            .map(|q| (out_offsets[q] as usize, out_offsets[q + 1] as usize))
            .collect();

        let mut out_chunks: Vec<&mut [f32]> = Vec::with_capacity(n_queries);
        {
            let mut rest = &mut out_slice[..];
            let mut cursor = 0usize;
            for &(s, e) in &bounds {
                debug_assert!(
                    s >= cursor && e >= s,
                    "out_offsets must be monotonically non-decreasing (got s={s}, e={e}, cursor={cursor})"
                );
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                out_chunks.push(mid);
                rest = tail2;
                cursor = e;
            }
        }

        out_chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(query, out_chunk)| {
                paint_query(query, out_chunk);
            });
    } else {
        for query in 0..n_queries {
            let out_s = out_offsets[query] as usize;
            let out_e = out_offsets[query + 1] as usize;
            paint_query(query, &mut out_slice[out_s..out_e]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run the kernel on owned arrays, return the out buffer.
    fn run(
        offset_idxs: &[i64],
        starts: &[i32],
        itv_starts: &[i32],
        itv_ends: &[i32],
        itv_values: &[f32],
        itv_offsets: &[i64],
        out_len: usize,
        out_offsets: &[i64],
    ) -> Vec<f32> {
        use ndarray::Array1;
        let mut out = Array1::<f32>::zeros(out_len);
        intervals_to_tracks(
            Array1::from_vec(offset_idxs.to_vec()).view(),
            Array1::from_vec(starts.to_vec()).view(),
            Array1::from_vec(itv_starts.to_vec()).view(),
            Array1::from_vec(itv_ends.to_vec()).view(),
            Array1::from_vec(itv_values.to_vec()).view(),
            Array1::from_vec(itv_offsets.to_vec()).view(),
            out.view_mut(),
            Array1::from_vec(out_offsets.to_vec()).view(),
            false, // serial path — unit tests don't need rayon overhead
        );
        out.to_vec()
    }

    /// Basic paint: one query, one interval fully inside → bp-range equals
    /// value, rest 0.
    #[test]
    fn test_basic_paint() {
        // query_start = 10, interval [11, 13) → start=1, end=3 in a length-5 out
        let result = run(
            &[0],          // offset_idxs
            &[10],         // starts (query_start)
            &[11],         // itv_starts
            &[13],         // itv_ends
            &[2.0],        // itv_values
            &[0, 1],       // itv_offsets
            5,             // out_len
            &[0, 5],       // out_offsets
        );
        assert_eq!(result, vec![0.0, 2.0, 2.0, 0.0, 0.0]);
    }

    /// Empty intervals: a query whose itv slice is empty → its out slice all 0.
    #[test]
    fn test_empty_intervals() {
        let result = run(
            &[0],
            &[0],
            &[],       // no intervals at all
            &[],
            &[],
            &[0, 0],   // itv_offsets: 0..0 → empty
            4,
            &[0, 4],
        );
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// End-clamp: interval end beyond length → painted only up to length, no
    /// overflow into the next query's region.
    #[test]
    fn test_end_clamp() {
        // query_start=0, interval [2, 100) on a length-5 out → painted [2..5)
        let result = run(
            &[0],
            &[0],
            &[2],
            &[100],
            &[7.0],
            &[0, 1],
            5,
            &[0, 5],
        );
        assert_eq!(result, vec![0.0, 0.0, 7.0, 7.0, 7.0]);
    }

    /// Break-on-start>=length: an interval with start >= length followed by
    /// another interval → second interval NOT painted (loop broke).
    #[test]
    fn test_break_on_start_ge_length() {
        // length=5, interval[0]: start=5 (== length) → break immediately
        // interval[1] at start=6 should never be visited.
        let result = run(
            &[0],
            &[0],
            &[5, 6],          // both out of range
            &[7, 8],
            &[3.0, 5.0],
            &[0, 2],
            5,
            &[0, 5],
        );
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// #242: interval starts before query_start, fully covers the window.
    #[test]
    fn test_interval_starts_before_query_full_cover() {
        // query_start=100, interval [96,114) on length-10 out -> all 5.0
        let result = run(&[0], &[100], &[96], &[114], &[5.0], &[0, 1], 10, &[0, 10]);
        assert_eq!(result, vec![5.0; 10]);
    }

    /// #242: partial left overlap -> clipped at 0.
    #[test]
    fn test_interval_starts_before_query_partial() {
        // query_start=10, interval [8,13) on length-5 out -> [5,5,5,0,0]
        let result = run(&[0], &[10], &[8], &[13], &[5.0], &[0, 1], 5, &[0, 5]);
        assert_eq!(result, vec![5.0, 5.0, 5.0, 0.0, 0.0]);
    }

    /// #242: interval ends at/below query_start -> no paint.
    #[test]
    fn test_interval_fully_left_of_query() {
        let result = run(&[0], &[10], &[2], &[6], &[5.0], &[0, 1], 5, &[0, 5]);
        assert_eq!(result, vec![0.0; 5]);
    }

    /// Multi-query disjoint: two queries with different offset_idxs/out slices
    /// paint independently.
    #[test]
    fn test_multi_query_disjoint() {
        // Query 0: offset_idx=0, itv_offsets[0..1]=[0,1] → itv 0, start=1,end=3 in out[0..5]
        // Query 1: offset_idx=1, itv_offsets[1..2]=[1,2] → itv 1, start=0,end=2 in out[5..10]
        let result = run(
            &[0, 1],       // offset_idxs
            &[10, 20],     // query starts
            &[11, 20],     // itv_starts
            &[13, 22],     // itv_ends
            &[2.0, 4.0],   // itv_values
            &[0, 1, 2],    // itv_offsets
            10,            // out_len (two 5-bp slices)
            &[0, 5, 10],   // out_offsets
        );
        // Query 0: out[1..3] = 2.0; rest 0
        // Query 1: out[5+0..5+2] = 4.0; rest 0
        assert_eq!(
            result,
            vec![0.0, 2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0]
        );
    }
}
