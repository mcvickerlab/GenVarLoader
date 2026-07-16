pub mod store;

/// Flattened sparse-genotype CSR for a batch of `(region, sample)` pairs,
/// produced by `Svar1Store::read_window`. Mirrors the layout
/// `reconstruct_haplotypes_from_sparse` expects: one CSR row per
/// `(batch-row, hap)`, `o_starts`/`o_stops` bracketing that row's slice of
/// `geno_v_idxs` (which holds GLOBAL variant indices), and
/// `geno_offset_idx[batch_row, hap]` giving the CSR row number
/// (`batch_row * ploidy + hap`). `geno_offset_idx` has shape `(batch, ploidy)`,
/// where `batch` is the number of `(region, sample)` pairs the caller
/// flattened — see `Svar1Store::read_window`.
pub struct Sparse {
    pub geno_v_idxs: Vec<i32>,
    pub o_starts: Vec<i64>,
    pub o_stops: Vec<i64>,
    pub geno_offset_idx: ndarray::Array2<i64>,
}
