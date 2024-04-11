use anyhow::Result;
use bigtools::{BigWigRead, Value};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::stack;
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::path::PathBuf;

pub fn intervals(
    paths: Vec<PathBuf>,
    contig: &str,
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
) -> Result<(Array2<u32>, Array1<f32>, Array2<i32>)> {
    let n_samples = paths.len();
    let n_regions = starts.len();

    // layouts are (samples, regions)
    let mut n_per_region_sample = Array2::<i32>::uninit((n_samples, n_regions));
    let n_per_region_sample_slice = n_per_region_sample.as_slice_mut().unwrap();

    let mut i_starts = vec![Vec::<u32>::new(); n_samples * n_regions];
    let mut i_ends = vec![Vec::<u32>::new(); n_samples * n_regions];
    let mut values = vec![Vec::<f32>::new(); n_samples * n_regions];

    paths
        .par_iter()
        .zip(n_per_region_sample_slice.par_chunks_exact_mut(n_regions))
        .zip(i_starts.par_chunks_exact_mut(n_regions))
        .zip(i_ends.par_chunks_exact_mut(n_regions))
        .zip(values.par_chunks_exact_mut(n_regions))
        .for_each(|((((path, n_slice), i_s_slice), i_e_slice), i_v_slice)| {
            let mut bw = BigWigRead::open_file(path.to_str().expect("Path is not unicode"))
                .expect("Error opening file");

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
                .zip(i_s_slice.iter_mut())
                .zip(i_e_slice.iter_mut())
                .zip(i_v_slice.iter_mut())
                .for_each(|(((((&r_start, &r_end), n), s), e), v)| {
                    let r_start = r_start.max(0) as u32;
                    let r_end = (r_end as u32).min(max_len);

                    *n = MaybeUninit::new(
                        bw.get_interval(contig.as_str(), r_start, r_end)
                            .expect("Error starting interval reading")
                            .into_iter()
                            .try_fold(0, |acc, itv| {
                                itv.map(|Value { start, end, value }| {
                                    s.push(start - 1);  // Wiggle is 1-indexed, but we want 0-based coordinates
                                    e.push(end);
                                    v.push(value);
                                    acc + 1
                                })
                            })
                            .expect("Error reading intervals"),
                    );
                })
        });

    let i_starts = interleave_and_flatten_vec_of_vec(i_starts, n_samples, n_regions);
    let i_ends = interleave_and_flatten_vec_of_vec(i_ends, n_samples, n_regions);
    let values = interleave_and_flatten_vec_of_vec(values, n_samples, n_regions);

    let coords = stack(Axis(1), &[i_starts.view(), i_ends.view()])
        .expect("Different number of starts and ends");

    unsafe {
        let n_per_region_sample = n_per_region_sample
            .assume_init()
            .t()
            .as_standard_layout()
            .to_owned();
        Ok((coords, values, n_per_region_sample))
    }
}

fn interleave_and_flatten_vec_of_vec<T: Clone + Send>(
    v: Vec<Vec<T>>,
    n_samples: usize,
    n_regions: usize,
) -> Array1<T> {
    Array1::from(
        Array2::from_shape_vec((n_samples, n_regions), v)
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned()
            .into_raw_vec()
            .into_par_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )
}
