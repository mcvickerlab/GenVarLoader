from pathlib import Path
from typing import (
    Callable,
    Sequence,
    Tuple,
    cast,
)

import cyvcf2
import numba as nb
import numpy as np
import pandera.polars as pa
import polars as pl
import seqpro as sp
from loguru import logger
from numpy.typing import NDArray

from .._dataset._impl import ArrayDataset
from .._dataset._indexing import DatasetIndexer
from .._types import AnnotatedHaps, Idx
from ._records import VLenAlleles


def return_true(variant: cyvcf2.Variant) -> bool:
    return True


def sites_vcf_to_table(
    vcf: str | Path | cyvcf2.VCF,
    filter: Callable[[cyvcf2.Variant], bool] = return_true,
    attributes: Sequence[str] | None = None,
    info_fields: Sequence[str] | None = None,
) -> pl.DataFrame:
    if isinstance(vcf, (str, Path)):
        _vcf = cast(cyvcf2.VCF, cyvcf2.VCF(str(vcf)))
    else:
        _vcf = vcf

    if attributes is None:
        attributes = []
    if info_fields is None:
        info_fields = []

    cols = ["keep", "chrom", "chromStart", "chromEnd", "REF", "ALT"]
    if attributes:
        cols.extend(attributes)
    if info_fields:
        cols.extend(info_fields)

    df = pl.DataFrame(
        dict(
            zip(
                cols,
                # rearrange from tuples of variants to tuples of attributes
                zip(
                    *(
                        (
                            v.is_snp & filter(v),
                            v.CHROM,
                            v.start,
                            v.end,
                            v.REF,
                            v.ALT,
                        )
                        + tuple(getattr(v, attr) for attr in attributes)
                        + tuple(v.INFO[f] for f in info_fields)
                        for v in _vcf
                    )
                ),
            )
        )
    )

    if (df["ALT"].list.len() > 1).any():
        logger.warning(
            "Some sites are multi-allelic; only the first will be used. To avoid this,"
            " preprocess the VCF with `bcftools norm -a -m -` to atomize and split them."
        )

    logger.info(f"Filter removed {(~df['keep']).sum()} of {len(df)} sites.")

    df = df.filter("keep").with_columns(pl.col("ALT").list.get(0)).drop("keep")

    return df


class SitesSchema(pa.DataFrameModel):
    CHROM: str = pa.Field(alias="#CHROM")
    POS: int
    ALT: str


def _sites_table_to_bedlike(sites: pl.DataFrame) -> pl.DataFrame:
    sites = sites.pipe(SitesSchema.validate)
    return sites.with_columns(
        chromEnd=pl.col("POS") + pl.col("ALT").str.len_bytes()
    ).rename({"#CHROM": "chrom", "POS": "chromStart"})


class DatasetWithSites:
    sites: pl.DataFrame
    dataset: "ArrayDataset[AnnotatedHaps, None, None, None]"
    rows: pl.DataFrame
    _row_map: NDArray[np.uint32]
    """Map from row index to dataset row index and site row index."""
    _idxer: DatasetIndexer

    @property
    def n_rows(self) -> int:
        return self._idxer.n_regions

    @property
    def n_samples(self) -> int:
        return self._idxer.n_samples

    @property
    def shape(self) -> Tuple[int, int]:
        return self._idxer.shape

    def __len__(self) -> int:
        return self.n_rows * self.n_samples

    def __init__(
        self,
        dataset: "ArrayDataset",
        sites: pl.DataFrame,
        max_variants_per_region: int = 1,
    ):
        if max_variants_per_region > 1:
            raise NotImplementedError("max_variants_per_region > 1 not yet supported")

        if not isinstance(dataset, ArrayDataset):
            raise ValueError(
                'Dataset output_length must either be "variable" or a fixed length integer.'
            )

        sites = _sites_table_to_bedlike(sites)

        dataset = (
            dataset.with_seqs("annotated")
            .with_tracks(None)
            .with_indices(False)
            .with_transform(None)
            .with_settings(deterministic=True, jitter=0)
        )

        ds_pyr = sp.bed.to_pyranges(dataset.regions.with_row_index("ds_row"))
        sites_pyr = sp.bed.to_pyranges(sites.with_row_index("site_row"))
        rows = pl.from_pandas(ds_pyr.join(sites_pyr, suffix="_site").df)
        if rows.height == 0:
            raise RuntimeError("No overlap between dataset regions and sites.")

        rows = (
            rows.rename(
                {
                    "Chromosome": "chrom",
                    "Start": "chromStart",
                    "End": "chromEnd",
                    "Strand": "strand",
                    "Start_site": "POS",
                },
                strict=False,
            )
            .drop("End_site")
            .sort("site_row")
        )
        self.sites = sites
        self.dataset = dataset
        self.rows = rows.drop("ds_row", "site_row")
        self._row_map = rows.select("ds_row", "site_row").to_numpy()
        self._idxer = DatasetIndexer.from_region_and_sample_idxs(
            np.arange(self.rows.height), np.arange(dataset.n_samples), dataset.samples
        )

    def __getitem__(
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> tuple["AnnotatedHaps", NDArray[np.uint8]]:
        idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, s_idx = np.unravel_index(idx, self.shape)

        ds_rows = self._row_map[r_idx, 0]
        haps = self.dataset[ds_rows, s_idx]

        sites = self.rows[r_idx]
        starts = sites["POS"].to_numpy()  # 0-based
        alts = VLenAlleles.from_polars(sites["ALT"])

        haps, v_idxs, ref_coords, flags = apply_site_only_variants(
            haps=haps.haps.view(np.uint8),  # (b p l)
            v_idxs=haps.var_idxs,  # (b p l)
            ref_coords=haps.ref_coords,  # (b p l)
            site_starts=starts,
            alt_alleles=alts.alleles.view(np.uint8),
            alt_offsets=alts.offsets,
        )

        haps = AnnotatedHaps(
            haps=haps.view("S1"),
            var_idxs=v_idxs,
            ref_coords=ref_coords,
        )

        if squeeze:
            haps = haps.squeeze(0)
            flags = flags.squeeze(0)

        if out_reshape is not None:
            haps = haps.reshape(*out_reshape, 2, -1)
            flags = flags.reshape(*out_reshape, 2)

        return haps, flags


APPLIED = np.uint8(0)
DELETED = np.uint8(1)
EXISTED = np.uint8(2)


# * fixed length, SNPs only
@nb.njit(parallel=True, nogil=True, cache=True)
def apply_site_only_variants(
    haps: NDArray[np.uint8],  # (b p l)
    v_idxs: NDArray[np.int32],  # (b p l)
    ref_coords: NDArray[np.int32],  # (b p l)
    site_starts: NDArray[np.int32],  # (b)
    alt_alleles: NDArray[np.uint8],  # ragged (b)
    alt_offsets: NDArray[np.int64],  # (b+1)
) -> Tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]]:
    batch_size, ploidy, _ = haps.shape
    flags = np.empty((batch_size, ploidy), dtype=np.uint8)

    for b in nb.prange(batch_size):
        for p in nb.prange(ploidy):
            bp_hap = haps[b, p]
            bp_idx = v_idxs[b, p]
            bp_ref_coord = ref_coords[b, p]
            pos = site_starts[b]
            alt = alt_alleles[alt_offsets[b] : alt_offsets[b + 1]]
            rel_start = np.searchsorted(bp_ref_coord, pos)
            rel_end = rel_start + len(alt)

            if bp_ref_coord[rel_start] != pos:
                flags[b, p] = DELETED
                continue

            if np.all(bp_hap[rel_start:rel_end] == alt):
                flags[b, p] = EXISTED
                continue

            flags[b, p] = APPLIED
            bp_hap[rel_start:rel_end] = alt
            bp_idx[rel_start:rel_end] = -2
    return haps, v_idxs, ref_coords, flags
