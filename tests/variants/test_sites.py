from typing import cast

import genvarloader as gvl
import numpy as np
import polars as pl


def test_sites():
    APPLIED = 0
    DELETED = 1
    EXISTED = 2
    ds = gvl.get_dummy_dataset().with_len(4).subset_to(regions=0)
    sites = (
        pl.concat([ds.regions, ds.regions])
        .with_columns(POS=pl.col("chromStart") + pl.arange(0, 2), ALT=pl.lit("T"))
        .rename({"chrom": "#CHROM"})
        .drop("chromStart", "chromEnd")
    )
    dss = gvl.DatasetWithSites(ds, sites)
    alt = cast(str, sites.item(0, "ALT")).encode("ascii")
    for s in range(dss.n_samples):
        ann_haps = dss.dataset[0, s].squeeze()
        # (l)
        haps = ann_haps.haps
        # (l)
        coords = ann_haps.ref_coords
        # SNP or insertion
        if coords[0] == coords[1] - 1 or coords[0] == coords[1]:
            # (1 l), (1)
            site_haps, flags = dss[0, s]  # first site
            # (l)
            site_haps = site_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            if haps[0] == alt:
                np.testing.assert_array_equal(site_haps, haps)
                assert flags == EXISTED
            elif haps[0] != alt:
                desired = haps.copy()
                desired[0] = alt
                np.testing.assert_array_equal(site_haps, desired)
                assert flags == APPLIED
        # deletion
        elif coords[0] < coords[1] - 1:
            # (1 l), (1)
            site_haps, flags = dss[1, s]  # second site
            # (l)
            site_haps = site_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            np.testing.assert_array_equal(site_haps, haps)
            assert flags == DELETED
