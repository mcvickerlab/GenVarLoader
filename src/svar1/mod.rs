pub mod store;

/// One window's sparse-genotype CSR geometry, produced by `Svar1Store::read_window`.
///
/// **The buffer itself holds only offsets.** SVAR1's on-disk layout is already
/// hap-major sparse CSR of sorted global variant ids, so there is nothing to decode
/// and no table to materialize — the variant ids these offsets index into live in the
/// `variant_idxs` mmap (`Svar1Reader::variant_idxs`), reached via the shared OS page
/// cache at generate time, and handed to the kernel as-is. VCF/PGEN backends (#276) DO
/// materialize an owned buffer; do not take this as their template.
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

impl Default for Svar1Window {
    fn default() -> Self {
        Svar1Window { o_starts: Vec::new(), o_stops: Vec::new() }
    }
}

#[cfg(test)]
mod window_default_tests {
    #[test]
    fn svar1_window_default_is_empty() {
        let w = super::Svar1Window::default();
        assert!(w.o_starts.is_empty() && w.o_stops.is_empty());
    }
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
