from typing import cast
import pytest

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
    dss = gvl.DatasetWithSites(ds, sites, max_variants_per_region=2)
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
            site_haps, flags = dss[0, s]
            # (l)
            site_haps = site_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            if haps[0] == alt:
                np.testing.assert_array_equal(site_haps, haps)
                assert flags[0] == EXISTED
            else:
                desired = haps.copy()
                desired[0] = alt
                np.testing.assert_array_equal(site_haps, desired)
                assert flags[0] == APPLIED
        # deletion
        elif coords[0] < coords[1] - 1:
            # (1 l), (1)
            site_haps, flags = dss[0, s]
            # (l)
            site_haps = site_haps.haps.squeeze()
            # ()
            flags = flags.squeeze()
            np.testing.assert_array_equal(site_haps, haps)
            assert flags[1] == DELETED


def test_sites_max_variants_error():
    ds = gvl.get_dummy_dataset().with_len(4).subset_to(regions=0).with_tracks(None)
    sites = (
        pl.concat([ds.regions, ds.regions])
        .with_columns(
            POS=pl.col("chromStart") + pl.arange(1, 3), REF=pl.lit("A"), ALT=pl.lit("T")
        )
        .rename({"chrom": "CHROM"})
        .drop("chromStart", "chromEnd")
    )
    with pytest.raises(ValueError):
        gvl.DatasetWithSites(ds, sites, max_variants_per_region=1)
