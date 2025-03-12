from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import cyvcf2
import numba as nb
import numpy as np
import pandera.polars as pa
import polars as pl
from attrs import define, field
from loguru import logger
from numpy.typing import NDArray

from python.genvarloader._types import Idx

from .._utils import bedlike_to_pyranges, idx_like_to_array
from ._records import VLenAlleles

if TYPE_CHECKING:
    from .._dataset import Dataset


FilterFn = Callable[[cyvcf2.Variant], bool]


class SitesOnly(Protocol):
    def to_bedlike(self) -> pl.DataFrame: ...


@define
class SitesOnlyVCF(SitesOnly):
    path: Union[str, Path]
    filter: Callable[[cyvcf2.Variant], bool] = field(default=lambda _: True)
    attributes: Sequence[str] = field(factory=list)
    info_fields: Sequence[str] = field(factory=list)

    def to_bedlike(self) -> pl.DataFrame:
        vcf = cyvcf2.VCF(str(self.path))

        cols = ["keep", "chrom", "chromStart", "chromEnd", "REF", "ALT", "n_alt"]
        if self.attributes:
            cols.extend(self.attributes)
        if self.info_fields:
            cols.extend(self.info_fields)

        df = pl.DataFrame(
            dict(
                zip(
                    cols,
                    # rearrange from tuples of variants to tuples of attributes
                    zip(
                        *(
                            (
                                v.is_snp & self.filter(v),
                                v.CHROM,
                                v.start,
                                v.end,
                                v.REF,
                                v.ALT[0],
                                len(v.ALT),
                            )
                            + tuple(getattr(v, attr) for attr in self.attributes)
                            + tuple(v.INFO[f] for f in self.info_fields)
                            for v in vcf
                        )
                    ),
                )
            )
        )

        if (df["n_alt"] > 1).any():
            logger.warning(
                "Some sites are multi-allelic; only the first will be used. To avoid this,"
                " preprocess the VCF with `bcftools norm -a -m -` to atomize and split them."
            )

        logger.info(f"Filter removed {(~df['keep']).sum()} of {len(df)} sites.")

        df = df.filter("keep").drop("keep")

        vcf.close()

        return df


class SitesOnlySchema(pa.DataFrameModel):
    CHROM: str = pa.Field(alias="#CHROM")
    POS: int
    REF: str
    ALT: str


class SitesOnlyTable(SitesOnly):
    table: pl.LazyFrame

    def __init__(self, table: pl.LazyFrame):
        self.table = table.pipe(SitesOnlySchema.validate)

    def to_bedlike(self) -> pl.DataFrame:
        df = (
            self.table.with_columns(
                chromEnd=pl.col("POS") + pl.col("ALT").str.len_bytes()
            )
            .rename({"#CHROM": "chrom", "POS": "chromStart"})
            .collect()
        )
        return df


class DatasetWithSites:
    sites: SitesOnly
    dataset: "Dataset"
    rows: pl.DataFrame
    _row_map: NDArray[np.uint32]
    """Map from row index to dataset row index and site row index."""

    @property
    def n_rows(self) -> int:
        return self.rows.height

    @property
    def n_samples(self) -> int:
        return self.dataset.n_samples

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_rows, self.n_samples

    def __init__(self, sites: SitesOnly, dataset: "Dataset", max_variants_per_region=1):
        if max_variants_per_region > 1:
            raise NotImplementedError("max_variants_per_region > 1 not yet supported")

        self.sites = sites
        self.dataset = dataset.with_settings(
            return_sequences="haplotypes",
            return_annotations=True,
            return_tracks=False,
            return_indices=False,
            transform=False,
            deterministic=True,
            jitter=0,
        )
        ds_pyr = bedlike_to_pyranges(dataset.regions.with_row_index("ds_row"))
        sites_pyr = bedlike_to_pyranges(sites.to_bedlike().with_row_index("site_row"))
        rows = (
            pl.from_pandas(ds_pyr.join(sites_pyr, suffix="_site").df)
            .rename(
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
        )
        self.rows = rows.drop("ds_row", "site_row")
        self._row_map = rows.select("ds_row", "site_row").to_numpy()

    def __getitem__(self, idx: Union[Idx, Tuple[Idx, Idx]]) -> Any:
        if not isinstance(idx, tuple):
            rows = idx
            samples = slice(None)
        elif len(idx) == 1:
            rows = idx[0]
            samples = slice(None)
        else:
            rows, samples = idx

        ds_rows = self._row_map[rows, 0]
        haps = self.dataset[ds_rows, samples]
        # (b p l), (b p l), (b p l), could be ragged, variable, or fixed length
        haps, v_idxs, ref_coords = haps.values()

        rows = idx_like_to_array(rows, self.n_rows)
        sites = self.rows[rows]
        starts = sites["POS"].to_numpy()  # 0-based
        alts = VLenAlleles.from_polars(sites["ALT"])

        haps, v_idxs, ref_coords, flags = apply_site_only_variants(
            haps=haps,
            v_idxs=v_idxs,
            ref_coords=ref_coords,
            site_starts=starts,
            alt_alleles=alts.alleles.view(np.uint8),
            alt_offsets=alts.offsets,
        )

        return haps, v_idxs, ref_coords


APPLIED = np.uint8(0)
DELETED = np.uint8(1)
EXISTED = np.uint8(2)


# * fixed length
def apply_site_only_variants(
    haps: NDArray[np.uint8],  # (b p l)
    v_idxs: NDArray[np.int32],  # (b p l)
    ref_coords: NDArray[np.int32],  # (b p l)
    site_starts: NDArray[np.int32],  # (b)
    alt_alleles: NDArray[np.uint8],  # ragged (b)
    alt_offsets: NDArray[np.int64],  # (b+1)
) -> Tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]]:
    batch_size, ploidy, length = haps.shape
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
