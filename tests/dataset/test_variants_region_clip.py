"""#202: written `with_seqs("variants")` must clip variants to the region window."""

import subprocess

import numpy as np
import polars as pl

import genvarloader as gvl
from genvarloader import Table

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
# Use the `vcf_snp_ins_del_multi_regions` fixture: narrow disjoint regions on
# the same contig with exactly this breakdown (it and `pgen_snp_ins_del_multi`
# both derive from `_VCF_PARITY_REF` / the same underlying data).


def test_written_variants_are_clipped_to_window(
    pgen_snp_ins_del_multi, vcf_snp_ins_del_multi_regions, tmp_path
):
    """Every returned variant's extent overlaps its cell's region window (#202)."""
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", vcf_snp_ins_del_multi_regions, variants=str(f.pgen))
    ds = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs("variants")

    regions = np.asarray(
        vcf_snp_ins_del_multi_regions.select(["chromStart", "chromEnd"]).to_numpy()
    )
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
    pgen_snp_ins_del_multi, vcf_snp_ins_del_multi_regions, tmp_path
):
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", vcf_snp_ins_del_multi_regions, variants=str(f.pgen))
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


# --- #314 -------------------------------------------------------------------
# The clip above is reached only on the DIRECT dispatch,
# `_haps.py::_get_haps -> get_variants_flat(self, idx, regions)`. When tracks
# are active the same query instead routes through
# `Haps.get_haps_and_shifts`, whose call site passed no `regions` at all -- so
# the clip short-circuited on its own `regions is None` guard and the raw
# genotype cell came back unfiltered.
#
# The fixtures above cannot show it: their written cells happen to hold only
# in-window variants, so clipping is a no-op. Provoking it needs a cell that
# genuinely holds an out-of-window variant, which `extend_to_length` (on by
# default at write time) produces: a deletion inside the window makes the
# writer read past `chromEnd` to meet the length target, pulling in variants
# the query-time window then has to clip back out.


def _write_extend_to_length_ds(tmp_path):
    """A dataset whose region-0 genotype cell holds a variant outside [50, 100).

    300bp of `ACGT` repeats on chr1, het in `s0` only:
      pos 60 (0-based): 21bp REF -> 1bp ALT deletion, inside region 0
      pos 110 (0-based): SNP, outside region 0 but inside region 1

    Region 0 is `[50, 100)`; the deletion shortens `s0`'s haplotype, so
    `extend_to_length` widens the write window past `chromEnd` and the pos-110
    SNP lands in region 0's cell.
    """
    ref = "ACGT" * 75
    del_ref = ref[60 : 60 + 21]
    del_alt = del_ref[0]
    snp_ref = ref[110]
    snp_alt = {"A": "C", "C": "G", "G": "T", "T": "A"}[snp_ref]

    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">chr1\n{ref}\n")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)

    vcf_txt = tmp_path / "in.vcf"
    vcf_txt.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=300>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
        f"chr1\t61\t.\t{del_ref}\t{del_alt}\t.\t.\t.\tGT\t1|0\t0|0\n"
        f"chr1\t111\t.\t{snp_ref}\t{snp_alt}\t.\t.\t.\tGT\t1|0\t0|0\n"
    )
    vcf_gz = tmp_path / "in.vcf.gz"
    with vcf_gz.open("wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf_txt)], stdout=fh, check=True)
    subprocess.run(["tabix", "-p", "vcf", str(vcf_gz)], check=True)

    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [50, 100],
            "chromEnd": [100, 150],
        }
    )
    samples = ["s0", "s1"]
    track = Table(
        "cov",
        pl.DataFrame(
            {
                "sample_id": samples,
                "chrom": ["chr1"] * len(samples),
                "start": [0] * len(samples),
                "end": [len(ref)] * len(samples),
                "value": [1.0] * len(samples),
            }
        ),
    )

    gvl.write(
        tmp_path / "ds", regions, variants=str(vcf_gz), tracks=track, overwrite=True
    )
    return tmp_path / "ds", fasta


def test_tracks_active_variants_are_clipped_to_window(tmp_path):
    """#314: the tracks-active dispatch must clip exactly like the direct one."""
    ds_path, fasta = _write_extend_to_length_ds(tmp_path)
    base = gvl.Dataset.open(ds_path, reference=fasta).with_seqs("variants")
    plain = base.with_tracks(False)
    tracked = base.with_tracks("cov")

    n_regions, n_samples = base.shape
    windows = [(50, 100), (100, 150)]

    for r in range(n_regions):
        r_start, r_end = windows[r]
        for s in range(n_samples):
            p = plain[r, s]
            t = tracked[r, s]
            t = t[0] if isinstance(t, tuple) else t
            for h in range(base.ploidy):
                p_starts = np.asarray(p.start[h]).astype(np.int64).tolist()
                t_starts = np.asarray(t.start[h]).astype(np.int64).tolist()
                assert t_starts == p_starts, (
                    f"cell ({r},{s}) hap {h}: tracks-active variants diverge from "
                    f"tracks-off; tracks-off={p_starts} tracks-on={t_starts}"
                )
                assert all(r_start <= v < r_end for v in t_starts), (
                    f"cell ({r},{s}) hap {h}: variant outside window "
                    f"[{r_start},{r_end}) with tracks active; starts={t_starts}"
                )

    # Non-vacuity: the pos-110 SNP really is in the dataset and really is
    # reachable -- region 1 returns it. Its absence from region 0 above is
    # therefore a clip, not a write-time drop.
    r1 = tracked[1, 0]
    r1 = r1[0] if isinstance(r1, tuple) else r1
    assert np.asarray(r1.start[0]).astype(np.int64).tolist() == [110], (
        "fixture no longer places the out-of-window SNP at pos 110; the clip "
        "assertions above are vacuous"
    )
