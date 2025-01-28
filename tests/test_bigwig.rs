use std::path::PathBuf;

use genvarloader::bigwig::{count_intervals, intervals};
use ndarray::prelude::*;
use rstest::*;

#[fixture]
fn test_dir() -> PathBuf {
    PathBuf::from(file!())
        .parent()
        .expect("Parent directory for file")
        .to_path_buf()
}

#[fixture]
fn bws(test_dir: PathBuf) -> Vec<PathBuf> {
    vec![
        test_dir.join("data/bigwig/sample_0.bw"),
        test_dir.join("data/bigwig/sample_1.bw"),
    ]
}

#[rstest]
#[case::check_zero_indexing("chr1", array![0], array![1], array![[0, 0]])]
#[case("chr1", array![0], array![2], array![[1, 1]])]
fn test_count_intervals(
    bws: Vec<PathBuf>,
    #[case] chrom: &str,
    #[case] starts: Array1<i32>,
    #[case] ends: Array1<i32>,
    #[case] desired: Array2<i32>,
) {
    let actual = count_intervals(&bws, chrom, starts.view(), ends.view()).unwrap();

    assert_eq!(actual, desired);
}

#[rstest]
#[case::one_itv_one_region("chr1", array![0], array![5], array![[0, 5], [0, 5],], array![1.0, 1.0])]
#[case::two_itvs_one_region("chr1", array![0], array![105], array![[0, 5], [99, 105], [0, 5], [99, 105]], array![1.0, 2.0, 1.0, 2.0])]
fn test_intervals(
    bws: Vec<PathBuf>,
    #[case] chrom: &str,
    #[case] starts: Array1<i32>,
    #[case] ends: Array1<i32>,
    #[case] des_coords: Array2<u32>,
    #[case] des_values: Array1<f32>,
) {
    // (regions, samples)
    let n_per_query = count_intervals(&bws, chrom, starts.view(), ends.view()).unwrap();
    // 0..cumsum
    let offsets = Array::from_iter((0..1).chain(n_per_query.as_slice().unwrap().iter().scan(
        0,
        |acc, &x| {
            *acc += x as i64;
            Some(*acc)
        },
    )));

    let (coords, values) =
        unsafe { intervals(&bws, chrom, starts.view(), ends.view(), offsets.view()) }.unwrap();

    assert_eq!(coords, des_coords);
    assert_eq!(values, des_values);
}
