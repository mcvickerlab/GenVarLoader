"""#202: written `with_seqs("variants")` must clip variants to the region window."""

import numpy as np
import polars as pl

import genvarloader as gvl

# `pgen_snp_ins_del_multi`'s `.regions` is a single full-contig region
# `[0, 250)` -- every variant overlaps it, so the #202 leak cannot manifest
# there (it would be a tautology). Use narrow, disjoint regions on the same
# contig instead so out-of-window variants genuinely leak with PGEN's
# contig-scoped genotype query.
#
# Variants (0-based pos, ilen): pos=29 (SNP, ilen 0), pos=69 (INS, ilen +2),
# pos=109 (DEL, ilen -3), pos=149 (SNP split x2, ilen 0).
#   region 0 [0,90)   -> contains pos 29, 69
#   region 1 [90,170) -> contains pos 109 (extent [109,113)) and both pos-149 atoms
#   region 2 [170,250) -> no variants
_NARROW_REGIONS = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr1"],
        "chromStart": [0, 90, 170],
        "chromEnd": [90, 170, 250],
    }
)


def test_written_variants_are_clipped_to_window(pgen_snp_ins_del_multi, tmp_path):
    """Every returned variant's extent overlaps its cell's region window (#202)."""
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", _NARROW_REGIONS, variants=str(f.pgen))
    ds = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs("variants")

    regions = np.asarray(_NARROW_REGIONS.select(["chromStart", "chromEnd"]).to_numpy())
    n_regions, n_samples = ds.shape

    for r in range(n_regions):
        r_start, r_end = int(regions[r, 0]), int(regions[r, 1])
        for s in range(n_samples):
            rv = ds[r, s]
            for h in range(ds.ploidy):
                starts = np.asarray(rv.start[h]).astype(np.int64)
                ilens = np.asarray(rv.ilen[h]).astype(np.int64)
                v_end = (
                    starts - np.minimum(ilens, 0) + 1
                )  # matches reconstruct/mod.rs:705
                overlaps = (starts < r_end) & (v_end > r_start)
                assert overlaps.all(), (
                    f"cell ({r},{s}) hap {h}: variant outside window "
                    f"[{r_start},{r_end}); starts={starts.tolist()}"
                )


def test_annotated_variants_are_a_subset_of_clipped_variants(
    pgen_snp_ins_del_multi, tmp_path
):
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", _NARROW_REGIONS, variants=str(f.pgen))
    base = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta)
    ann = base.with_seqs("annotated")
    var = base.with_seqs("variants")

    n_regions, n_samples = base.shape
    v_starts = np.asarray(base._seqs.ffi_static.v_starts).astype(np.int64)  # type: ignore[attr-defined]

    for r in range(n_regions):
        for s in range(n_samples):
            a = ann[r, s]
            v = var[r, s]
            for h in range(base.ploidy):
                a_ids = np.asarray(a.var_idxs[h])
                a_ids = a_ids[a_ids >= 0]
                appeared_starts = set(v_starts[a_ids].tolist())
                clipped_starts = set(np.asarray(v.start[h]).astype(np.int64).tolist())
                assert appeared_starts <= clipped_starts, (
                    f"cell ({r},{s}) hap {h}: annotated used a variant the clipped "
                    f"variants output dropped"
                )
