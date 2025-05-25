from typing import cast

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader._variants._sitesonly import APPLIED, DELETED, EXISTED


def test_sites():
    ds = gvl.get_dummy_dataset().with_len(4).subset_to(regions=0).with_tracks(None)
    sites = (
        pl.concat([ds.regions, ds.regions])
        .with_columns(
            POS=pl.col("chromStart") + pl.arange(1, 3), REF=pl.lit("A"), ALT=pl.lit("T")
        )
        .rename({"chrom": "CHROM"})
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
            _, mut_haps, flags = dss[0, s]  # first site
            # (l)
            mut_haps = mut_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            if haps[0] == alt:
                np.testing.assert_array_equal(mut_haps, haps)
                assert flags == EXISTED
            elif haps[0] != alt:
                desired = haps.copy()
                desired[0] = alt
                np.testing.assert_array_equal(mut_haps, desired)
                assert flags == APPLIED
        # deletion
        elif coords[0] < coords[1] - 1:
            # (1 l), (1)
            _, mut_haps, flags = dss[1, s]  # second site
            # (l)
            mut_haps = mut_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            np.testing.assert_array_equal(mut_haps, haps)
            assert flags == DELETED
