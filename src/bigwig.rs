use anyhow::Result;
use bigtools::{BigWigRead, Value};
use itertools::{izip, Itertools};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::path::PathBuf;

pub fn count_intervals(
    paths: &Vec<PathBuf>,
    contig: &str,
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
) -> Result<Array2<i32>> {
    let n_samples = paths.len();
    let n_regions = starts.len();

    // layout is (samples, regions)
    let mut n_per_region_sample = Array2::<i32>::uninit((n_samples, n_regions));
    let n_per_region_sample_slice = n_per_region_sample.as_slice_mut().unwrap();

    paths
        .par_iter()
        .zip(n_per_region_sample_slice.par_chunks_exact_mut(n_regions))
        .for_each(|(path, n_slice)| {
            let mut bw = BigWigRead::open_file(path).expect("Error opening file");

            let (max_len, contig) = bw
                .chroms()
                .iter()
                .filter_map(|chrom| {
                    if chrom.name == contig || chrom.name == format!("chr{contig}") {
                        Some((chrom.length, chrom.name.clone()))
                    } else {
                        None
                    }
                })
                .exactly_one()
                .expect("Contig not found or multiple contigs match");

            starts
                .as_slice()
                .expect("Starts array is not contiguous")
                .iter()
                .zip(ends.as_slice().expect("Ends array is not contiguous"))
                .zip(n_slice.iter_mut())
                .for_each(|((&r_start, &r_end), n)| {
                    let r_start = r_start.max(0) as u32;
                    let r_end = (r_end as u32).min(max_len);

                    *n = MaybeUninit::new(
                        bw.get_interval(contig.as_str(), r_start, r_end)
                            .expect("Error starting interval reading")
                            .into_iter()
                            .count() as i32,
                    );
                })
        });

    // convert layout to (regions, samples)
    unsafe {
        let n_per_region_sample = n_per_region_sample
            .assume_init()
            .t()
            .as_standard_layout()
            .to_owned();
        Ok(n_per_region_sample)
    }
}

/// This is an UNSAFE function because it assumes the offsets exactly correspond to the intervals
/// in the bigwig files. If arbitrary offsets are used, this function can cause data races or
/// segfault by writing to unallocated memory. The only offsets that are valid are those from
/// [`count_intervals`] for the exact same arguments sans `offsets`.
pub unsafe fn intervals(
    paths: &Vec<PathBuf>,
    contig: &str,
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
    offsets: ArrayView1<i64>,
) -> Result<(Array2<u32>, Array1<f32>)> {
    let n_samples = paths.len();
    // flattened (regions, samples)
    let n_intervals = offsets[offsets.len() - 1] as usize;

    // layout is Ragged<(intervals, 2)> with shape (regions, samples)
    let coords = Array2::<u32>::uninit((n_intervals, 2));
    let values = Array1::<f32>::uninit(n_intervals);

    paths.par_iter().enumerate().for_each(|(s_idx, path)| {
        let mut bw = BigWigRead::open_file(path).expect("Error opening file");
        let (max_len, contig) = bw
            .chroms()
            .iter()
            .filter_map(|chrom| {
                if chrom.name == contig || chrom.name == format!("chr{contig}") {
                    Some((chrom.length, chrom.name.clone()))
                } else {
                    None
                }
            })
            .exactly_one()
            .expect("Contig not found or multiple contigs match");

        izip!(starts, ends)
            .enumerate()
            .for_each(|(r_idx, (&s, &e))| {
                let coords_ptr = coords.as_ptr() as *mut MaybeUninit<u32>;
                let values_ptr = values.as_ptr() as *mut MaybeUninit<f32>;
                let offset = offsets[r_idx * n_samples + s_idx] as usize;

                let r_start = s.max(0) as u32;
                let r_end = (e as u32).min(max_len);

                bw.get_interval(contig.as_str(), r_start, r_end)
                    .expect("Begin reading intervals")
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, itv)| {
                        let Value { start, end, value } = itv.expect("Read interval");
                        unsafe {
                            coords_ptr
                                .add((offset + i) * 2)
                                .write(MaybeUninit::new(start - 1));
                            coords_ptr
                                .add((offset + i) * 2 + 1)
                                .write(MaybeUninit::new(end));
                            values_ptr.add(i + offset).write(MaybeUninit::new(value));
                        }
                    });
            })
    });

    unsafe {
        let coords = coords.assume_init();
        let values = values.assume_init();
        Ok((coords, values))
    }
}
