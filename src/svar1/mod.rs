pub mod store;

/// One window's sparse-genotype CSR geometry, produced by `Svar1Store::read_window`.
///
/// **This is the degenerate case of the SVAR1-style window buffer: it holds only
/// offsets.** SVAR1's on-disk layout is already hap-major sparse CSR of sorted global
/// variant ids, so there is nothing to decode and no table to materialize —
/// `geno_v_idxs` is borrowed straight from the `variant_idxs` mmap
/// (`Svar1Reader::variant_idxs`) and handed to the kernel as-is. VCF/PGEN backends
/// (#276) DO materialize an owned buffer; do not take this as their template.
///
/// A window is CARTESIAN: `n_regions x n_samples x ploidy`. `o_starts`/`o_stops` are
/// `n_regions * n_samples * ploidy` long in C-order `(region, sample, ploid)` —
/// absolute indices into `variant_idxs`. The batch-row -> CSR-row map is the identity
/// `bi * ploidy + p`, so callers (e.g. `svar1_generate_batch`) rebuild it locally over
/// just their batch slice rather than carrying a window-scale copy here.
pub struct Svar1Window {
    pub o_starts: Vec<i64>,
    pub o_stops: Vec<i64>,
}

#[cfg(test)]
mod link_tests {
    /// Smoke: the ungated genoray SVAR1 query API is visible from gvl's crate.
    #[test]
    fn svar1_query_symbols_are_linkable() {
        use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
        let _ = Svar1Reader::open;
        let _ = var_ranges;
        let _ = find_ranges;
    }
}
