from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, Sequence, Tuple, overload

import cyvcf2
import numba as nb
import numpy as np
import pandera.polars as pa
import polars as pl
import seqpro as sp
from genoray import VCF
from numpy.typing import NDArray

from .._dataset._impl import ArrayDataset, MaybeTRK
from .._dataset._indexing import DatasetIndexer
from .._types import AnnotatedHaps, Idx
from ._records import RaggedAlleles


def sites_vcf_to_table(
    vcf: str | Path | VCF,
    filter: Callable[[cyvcf2.Variant], bool] | None = None,
    attributes: list[str] | None = None,
    info_fields: list[str] | None = None,
) -> pl.DataFrame:
    """Extract a table of variant site info from a VCF.
    
    Parameters
    ----------
    vcf
        Path to a VCF.
    filter
        A callable that takes a cyvcf2.Variant and returns True if the variant should be included.
    """
    if not isinstance(vcf, VCF):
        vcf = VCF(vcf)
    if filter is not None:
        vcf.filter = filter

    min_attrs = ["CHROM", "POS", "ALT"]
    if attributes is None:
        attributes = min_attrs
    else:
        attributes = min_attrs + [attr for attr in attributes if attr not in min_attrs]

    df = vcf.get_record_info(attrs=attributes, info=info_fields, progress=True)

    return df


class SitesSchema(pa.DataFrameModel):
    """Schema to validate a table of variant sites."""
    CHROM: str
    POS: int
    ALT: str


def _sites_table_to_bedlike(sites: pl.DataFrame) -> pl.DataFrame:
    sites = sites.pipe(SitesSchema.validate)
    return (
        sites.with_columns(
            chromStart=pl.col("POS") - 1,
            chromEnd=pl.col("POS") + pl.col("ALT").str.len_bytes() - 1,
        )
        .drop("POS")
        .rename({"CHROM": "chrom"})
    )


class DatasetWithSites(Generic[MaybeTRK]):
    sites: pl.DataFrame
    dataset: ArrayDataset[AnnotatedHaps, MaybeTRK, None, None]
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
        dataset: ArrayDataset[Any, MaybeTRK, Any, Any],
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
                    "Start_site": "POS0",
                },
                strict=False,
            )
            .drop("End_site")
            .sort("site_row")
        )

        _dataset = (
            dataset.with_seqs("annotated")
            .with_indices(False)
            .with_transform(None)
            .with_settings(deterministic=True, jitter=0)
        )

        self.sites = sites
        self.dataset = _dataset
        self.rows = rows.drop("ds_row", "site_row")
        self._row_map = rows.select("ds_row", "site_row").to_numpy()
        self._idxer = DatasetIndexer.from_region_and_sample_idxs(
            np.arange(self.rows.height), np.arange(dataset.n_samples), dataset.samples
        )

    @overload
    def __getitem__(
        self: DatasetWithSites[None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> tuple[AnnotatedHaps, NDArray[np.uint8]]: ...
    @overload
    def __getitem__(
        self: DatasetWithSites[NDArray[np.float32]],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> tuple[AnnotatedHaps, NDArray[np.uint8], NDArray[np.float32]]: ...
    def __getitem__(
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> (
        tuple[AnnotatedHaps, NDArray[np.uint8]]
        | tuple[AnnotatedHaps, NDArray[np.uint8], NDArray[np.float32]]
    ):
        idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, s_idx = np.unravel_index(idx, self.shape)

        ds_rows = self._row_map[r_idx, 0]
        out = self.dataset[ds_rows, s_idx]
        if isinstance(out, tuple):
            haps, tracks = out
        else:
            haps = out

        ploidy = haps.shape[-2]
        length = haps.shape[-1]

        sites = self.rows[r_idx]
        starts = sites["POS0"].to_numpy()  # 0-based
        alts = RaggedAlleles.from_polars(sites["ALT"])

        # (b p)
        haps = haps.reshape((-1, ploidy, length))
        # flags: (b p)
        haps, v_idxs, ref_coords, flags = apply_site_only_variants(
            haps=haps.haps.view(np.uint8),  # (b p l)
            v_idxs=haps.var_idxs,  # (b p l)
            ref_coords=haps.ref_coords,  # (b p l)
            site_starts=starts,
            alt_alleles=alts.data.view(np.uint8),
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
            haps = haps.reshape((*out_reshape, ploidy, length))
            flags = flags.reshape(*out_reshape, ploidy)

        if isinstance(out, tuple):
            return (
                haps,
                flags,
                tracks,  # type: ignore | guaranteed bound
            )
        else:
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
