use anyhow::{Context, Result};
use bigtools::{BigWigRead, Value};
use itertools::{izip, Itertools};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};

const REGION_BATCH: usize = 512;

// If this alias is wrong for the pinned bigtools, the build error names the real type.
type BwReader = BigWigRead<bigtools::utils::reopen::ReopenableFile>;

thread_local! {
    static READERS: RefCell<HashMap<PathBuf, BwReader>> =
        RefCell::new(HashMap::new());
}

/// Decoded intervals for one region across all samples: per-sample Vec<Value>.
type RegionDecoded = Vec<Vec<Value>>;

pub fn write_track(
    paths: &[PathBuf],
    contigs: &[String],
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
    max_mem: usize,
    out_dir: &Path,
    _sample_less: bool,
) -> Result<()> {
    let n_regions = starts.len();
    let n_samples = paths.len();
    let starts = starts.as_slice().expect("starts contiguous");
    let ends = ends.as_slice().expect("ends contiguous");

    let mut itv_writer = BufWriter::new(File::create(out_dir.join("intervals.npy"))?);
    // offsets accumulated in memory; region-major, sample-minor; final total appended.
    let mut offsets: Vec<i64> = Vec::with_capacity(n_regions * n_samples + 1);
    offsets.push(0);
    let mut acc: i64 = 0;

    let mut batch_start = 0usize;
    while batch_start < n_regions {
        let batch_end = (batch_start + REGION_BATCH).min(n_regions);
        let batch: Vec<usize> = (batch_start..batch_end).collect();

        // Parallel decode each region (all samples), preserving order via collect.
        let decoded: Vec<Result<RegionDecoded>> = batch
            .par_iter()
            .map(|&r| {
                READERS.with(|cell| {
                    let mut readers = cell.borrow_mut();
                    let contig = &contigs[r];
                    let mut per_sample: RegionDecoded = Vec::with_capacity(n_samples);
                    for path in paths.iter() {
                        let bw = match readers.entry(path.clone()) {
                            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                let reader = BigWigRead::open_file(path)
                                    .with_context(|| format!("Error opening bigWig {}", path.display()))?;
                                e.insert(reader)
                            }
                        };
                        let (max_len, name) = bw
                            .chroms()
                            .iter()
                            .filter_map(|c| {
                                if &c.name == contig || c.name == format!("chr{contig}") {
                                    Some((c.length, c.name.clone()))
                                } else {
                                    None
                                }
                            })
                            .exactly_one()
                            .map_err(|_| anyhow::anyhow!("Contig {:?} not found or multiple contigs match in {}", contig, path.display()))?;
                        let r_start = starts[r].max(0) as u32;
                        let r_end = (ends[r] as u32).min(max_len);
                        let vals: Vec<Value> = bw
                            .get_interval(name.as_str(), r_start, r_end)
                            .with_context(|| format!("Error reading intervals for contig {:?} in {}", contig, path.display()))?
                            .into_iter()
                            .map(|v| v.with_context(|| "Error reading interval value"))
                            .collect::<Result<Vec<_>>>()?;
                        per_sample.push(vals);
                    }
                    let region_bytes: usize =
                        per_sample.iter().map(|v| v.len()).sum::<usize>() * 12;
                    if region_bytes > max_mem {
                        anyhow::bail!(
                            "Memory usage per region exceeds max_mem ({} > {}).",
                            region_bytes,
                            max_mem
                        );
                    }
                    Ok(per_sample)
                })
            })
            .collect();

        for region in decoded {
            let per_sample = region?;
            for sample_vals in per_sample {
                for v in sample_vals {
                    itv_writer.write_all(&(v.start as i32).to_le_bytes())?;
                    itv_writer.write_all(&(v.end as i32).to_le_bytes())?;
                    itv_writer.write_all(&v.value.to_le_bytes())?;
                    acc += 1;
                }
                offsets.push(acc);
            }
        }
        batch_start = batch_end;
    }
    itv_writer.flush()?;

    let mut off_writer = BufWriter::new(File::create(out_dir.join("offsets.npy"))?);
    for o in &offsets {
        off_writer.write_all(&o.to_le_bytes())?;
    }
    off_writer.flush()?;
    Ok(())
}

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
                                .write(MaybeUninit::new(start));
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::fs;

    fn fixture_paths() -> Vec<PathBuf> {
        // tests/data/bigwig/sample_{0,1}.bw (chr1 len 2000, chr2 len 1000)
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/bigwig");
        vec![base.join("sample_0.bw"), base.join("sample_1.bw")]
    }

    #[test]
    fn write_track_matches_count_and_intervals_oracle() {
        let paths = fixture_paths();
        let contigs = vec!["chr1".to_string(), "chr1".to_string()];
        let starts = array![0i32, 50];
        let ends = array![200i32, 110];
        let tmp = std::env::temp_dir().join("gvl_bw_write_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        write_track(
            &paths,
            &contigs,
            starts.view(),
            ends.view(),
            1 << 30,
            &tmp,
            false,
        )
        .unwrap();

        // Oracle: count_intervals (per contig) + intervals, replicating the Python path.
        // Region 0 and 1 are both on chr1; build expected offsets + packed bytes.
        let n0 = count_intervals(&paths, "chr1", array![0i32, 50].view(), array![200i32, 110].view())
            .unwrap(); // (regions, samples)
        let offsets: Vec<i64> = {
            let mut acc = 0i64;
            let mut v = vec![0i64];
            for r in 0..n0.nrows() {
                for s in 0..n0.ncols() {
                    acc += n0[[r, s]] as i64;
                    v.push(acc);
                }
            }
            v
        };
        let (coords, vals) = unsafe {
            intervals(
                &paths,
                "chr1",
                array![0i32, 50].view(),
                array![200i32, 110].view(),
                ndarray::aview1(&offsets),
            )
        }
        .unwrap();

        // Expected intervals.npy bytes: [i32 start, i32 end, f32 value] per row.
        let mut expected = Vec::new();
        for i in 0..vals.len() {
            expected.extend_from_slice(&(coords[[i, 0]] as i32).to_le_bytes());
            expected.extend_from_slice(&(coords[[i, 1]] as i32).to_le_bytes());
            expected.extend_from_slice(&vals[i].to_le_bytes());
        }
        let got = fs::read(tmp.join("intervals.npy")).unwrap();
        assert_eq!(got, expected, "intervals.npy bytes mismatch");

        // Expected offsets.npy bytes: i64 little-endian, full offsets vec.
        let mut expected_off = Vec::new();
        for o in &offsets {
            expected_off.extend_from_slice(&o.to_le_bytes());
        }
        let got_off = fs::read(tmp.join("offsets.npy")).unwrap();
        assert_eq!(got_off, expected_off, "offsets.npy bytes mismatch");
    }

    #[test]
    fn write_track_errors_when_region_exceeds_max_mem() {
        let paths = fixture_paths();
        let contigs = vec!["chr1".to_string()];
        let starts = array![0i32];
        let ends = array![2000i32];
        let tmp = std::env::temp_dir().join("gvl_bw_write_oom");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // max_mem = 1 byte -> any region with >=1 interval exceeds it
        let res = write_track(&paths, &contigs, starts.view(), ends.view(), 1, &tmp, false);
        assert!(res.is_err());
    }
}
