use anyhow::Result;
use bigtools::BigWigRead;
use itertools::{multiunzip, Itertools};
use ndarray::prelude::*;
use ndarray::stack;
use rayon::prelude::*;
use std::path::PathBuf;

pub fn intervals(
    paths: Vec<PathBuf>,
    contig: &str,
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
) -> Result<(Array2<u32>, Array1<f32>, Array2<i32>)> {
    // (samples) of tuple (s e v n)
    // (s e v n) are each vec with len regions
    let mut intervals_queries = Vec::with_capacity(paths.len());
    paths
        .par_iter()
        .map(|path| {
            let mut bw = BigWigRead::open_file(path.to_str().expect("Path is not"))
                .expect("Error opening file");

            let max_len = bw
                .chroms()
                .iter()
                .filter_map(|chrom| {
                    if chrom.name == contig {
                        Some(chrom.length)
                    } else if chrom.name == format!("chr{}", contig) {
                        Some(chrom.length)
                    } else {
                        None
                    }
                })
                .next()
                .expect("Contig not found");

            let (intervals, n_per_region): (Vec<Vec<(u32, u32, f32)>>, Vec<i32>) = starts
                .iter()
                .zip(ends.iter())
                .map(|(&start, &end)| {
                    let start = start.max(0) as u32;
                    let end = (end as u32).min(max_len);
                    let itvs = bw
                        .get_interval(contig, start, end)
                        .expect("Error starting interval reading")
                        .into_iter()
                        .filter_map_ok(|itv| Some((itv.start, itv.end, itv.value)))
                        .collect::<Result<Vec<_>, _>>()
                        .expect("Error reading intervals");

                    let n = itvs.len() as i32;
                    (itvs, n)
                })
                .unzip();

            let (i_s, i_e, vals): (Vec<u32>, Vec<u32>, Vec<f32>) =
                multiunzip(intervals.into_iter().flatten().map(|(a, b, c)| (a, b, c)));

            (i_s, i_e, vals, n_per_region)
        })
        .collect_into_vec(&mut intervals_queries);

    let (i_starts, i_ends, values, n_per_query): (
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<Vec<f32>>,
        Vec<Vec<i32>>,
    ) = multiunzip(
        intervals_queries
            .into_iter()
            .map(|(a, b, c, d)| (a, b, c, d)),
    );

    let i_starts = Array1::from(i_starts.into_iter().flatten().collect::<Vec<_>>());
    let i_ends = Array1::from(i_ends.into_iter().flatten().collect::<Vec<_>>());
    let coords = stack(Axis(1), &[i_starts.view(), i_ends.view()])
        .expect("Different number of starts and ends");

    let values = values.into_iter().flatten().collect::<Vec<_>>();

    let n_per_query = n_per_query.into_iter().flatten().collect::<Vec<_>>();
    let n_per_query = Array2::from_shape_vec((paths.len(), starts.len()), n_per_query)
        .expect("Wrong number of queries");

    Ok((coords, values.into(), n_per_query))
}
