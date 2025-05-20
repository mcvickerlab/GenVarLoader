from __future__ import annotations

from pathlib import Path
from typing import Generic, Sequence, Tuple, overload

import numba as nb
import numpy as np
import pandera.polars as pa
import polars as pl
import seqpro as sp
from genoray import VCF
from genoray._utils import ContigNormalizer
from numpy.typing import NDArray

from .._dataset._impl import SEQ, ArrayDataset, MaybeTRK
from .._dataset._indexing import DatasetIndexer
from .._types import AnnotatedHaps, Idx
from .._utils import _lengths_to_offsets
from ._records import RaggedAlleles


def sites_vcf_to_table(
    vcf: str | Path | VCF,
    attributes: list[str] = ["CHROM", "POS", "REF", "ALT"],
    info_fields: list[str] | None = None,
) -> pl.DataFrame:
    """Extract a table of variant site info from a VCF. All sites must be bi-allelic.

    Parameters
    ----------
    vcf
        Path to a VCF or a :class:`genoray.VCF` instance. Note that :class:`genoray.VCF` can accept a filter function.
    attributes
        A list of attributes to include in the output table. Note that "CHROM", "POS", "REF", and "ALT" are always included
        even if not in this list.
    info_fields
        A list of INFO fields to include in the output table.
    """
    if not isinstance(vcf, VCF):
        vcf = VCF(vcf)

    min_attrs = ["CHROM", "POS", "REF", "ALT"]
    attrs = min_attrs + [attr for attr in attributes if attr not in min_attrs]

    df = vcf.get_record_info(attrs=attrs, info=info_fields, progress=True)

    if df.select((pl.col("ALT").list.len() > 1).any()).item():
        raise ValueError("All sites must be bi-allelic.")

    df = df.with_columns(pl.col("ALT").list.first())

    return df


class SitesSchema(pa.DataFrameModel):
    """Schema to validate a table of variant sites."""

    CHROM: str
    POS: int
    REF: str
    ALT: str


def _sites_table_to_bedlike(sites: pl.DataFrame) -> pl.DataFrame:
    sites = sites.pipe(SitesSchema.validate)
    return (
        sites.with_columns(
            chromStart=pl.col("POS") - 1,
            chromEnd=pl.col("POS") + pl.col("REF").str.len_bytes() - 1,
        )
        .drop("POS")
        .rename({"CHROM": "chrom"})
    )


class DatasetWithSites(Generic[MaybeTRK]):
    dataset: ArrayDataset[AnnotatedHaps, MaybeTRK]
    """Dataset of haplotypes and potentially tracks."""
    sites: pl.DataFrame
    """Table of variant site information."""
    rows: pl.DataFrame
    """Rows of this object. Each row corresponds to a dataset region and all
    site-only variants that fall within that region."""
    _row_map: NDArray[np.uint32]
    """Map from row index to dataset row index."""
    _site_offsets: NDArray[np.int64]
    """Offsets into ``_site_starts``/``_alt_data`` for each row."""
    _site_starts: NDArray[np.int32]
    """Flattened start positions of site-only variants."""
    _alt_data: NDArray[np.uint8]
    """Flattened alternative alleles for all site-only variants."""
    _alt_offsets: NDArray[np.int64]
    """Offsets into ``_alt_data`` for each site-only variant."""
    max_variants_per_region: int
    _n_variants_per_row: NDArray[np.int32]
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
        dataset: ArrayDataset[SEQ, MaybeTRK],
        sites: pl.DataFrame,
        max_variants_per_region: int = 1,
    ):
        """Dataset with variant sites, used to apply site-only variants e.g. from ClinVar to a Dataset of haplotypes.
        Currently only supports bi-allelic SNPs. Takes the intersection of the dataset regions and the sites, and
        applies the site-only variants to the Dataset's haplotypes.

        Accessed just like a Dataset, but where the rows are combinations of dataset regions and sites. Will return
        :class:`AnnotatedHaps` with variants applied and flags indicating whether the variant was applied, deleted, or existed.
        The flags are 0 for applied, 1 for deleted, and 2 for existed. If the dataset has tracks, they will be
        returned as well and reflect any site-only variants.

        Parameters
        ----------
        dataset
            Dataset of haplotypes and potentially tracks.
        sites
            Table of variant site information.
        max_variants_per_region
            Maximum number of variants per region. Currently only 1 is supported.

        Examples
        --------
        .. code-block:: python

            import genvarloader as gvl
            sites = gvl.sites_vcf_to_table("path/to/variants.vcf")

            ds = gvl.Dataset.open("path/to/dataset.gvl", "path/to/reference.fasta")
            ds_sites = gvl.DatasetWithSites(ds, sites)
            haps, flags = ds_sites[0, 0]
            # flags is a np.uint8 (or an array of np.uint8 when accessing multiple rows/samples)

            ds_sites.dataset = ds_sites.dataset.with_tracks("read-depth")
            haps, flags, tracks = ds_sites[0, 0]
        """
        self.max_variants_per_region = max_variants_per_region

        if not isinstance(dataset, ArrayDataset):
            raise ValueError(
                'Dataset output_length must either be "variable" or a fixed length integer.'
            )

        sites = _sites_table_to_bedlike(sites)

        if sites.select(
            (
                (pl.col("REF").str.len_bytes() != 1) | pl.col("ALT").str.len_bytes()
                != 1
            ).any()
        ).item():
            raise ValueError(
                "All sites must be SNPs. Consider filtering the VCF as either a preprocessing step or via the sites_vcf_to_table function."
            )

        c_norm = ContigNormalizer(dataset.contigs)
        chroms: list[str] = sites["chrom"].to_list()
        norm_chroms = c_norm.norm(chroms)
        norm_chroms = [
            c if norm is None else norm for c, norm in zip(chroms, norm_chroms)
        ]
        sites = sites.with_columns(chrom=pl.Series(norm_chroms))

        ds_bed = dataset.regions.with_row_index("region_idx")
        if isinstance(dataset.output_length, int):
            ds_bed = ds_bed.with_columns(
                chromEnd=pl.col("chromStart") + dataset.output_length
            )
        ds_pyr = sp.bed.to_pyranges(ds_bed)
        sites_pyr = sp.bed.to_pyranges(sites.with_row_index("site_idx"))
        rows = pl.from_pandas(ds_pyr.join(sites_pyr, suffix="_site").df)
        if rows.height == 0:
            raise RuntimeError("No overlap between dataset regions and sites.")

        rows = rows.rename(
            {
                "Chromosome": "chrom",
                "Start": "chromStart",
                "End": "chromEnd",
                "Strand": "strand",
                "Start_site": "POS0",
            },
            strict=False,
        ).drop("End_site")

        rows = (
            rows.sort(["region_idx", "POS0"])
            .group_by("region_idx", maintain_order=True)
            .agg(
                site_idx=pl.col("site_idx").list(),
                POS0=pl.col("POS0").list(),
                ALT=pl.col("ALT").list(),
            )
            .join(ds_bed, on="region_idx")
        )

        counts = rows["site_idx"].list.len()
        if counts.max() > max_variants_per_region:
            raise ValueError(
                "Found a region with more than allowed site-only variants"
            )
        self._n_variants_per_row = counts.to_numpy().astype(np.int32)

        site_lists = rows["POS0"].to_list()
        flat_sites = [s for sub in site_lists for s in sub]
        self._site_starts = np.array(flat_sites, dtype=np.int32)
        self._site_offsets = _lengths_to_offsets([len(s) for s in site_lists])

        alt_lists = rows["ALT"].to_list()
        flat_alts = [a for sub in alt_lists for a in sub]
        alts = RaggedAlleles.from_polars(pl.Series(flat_alts))
        self._alt_data = alts.data.view(np.uint8)
        self._alt_offsets = alts.offsets

        self.sites = sites
        self.dataset = _dataset
        self.rows = rows
        self._row_map = rows.select("region_idx").to_numpy().ravel()
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

        ds_rows = self._row_map[r_idx]
        out = self.dataset[ds_rows, s_idx]
        if isinstance(out, tuple):
            haps, tracks = out
        else:
            haps = out

        ploidy = haps.shape[-2]
        length = haps.shape[-1]

        haps = haps.reshape((-1, ploidy, length))

        flags = np.full(
            (len(r_idx), ploidy, self.max_variants_per_region),
            255,
            dtype=np.uint8,
        )

        for b, row in enumerate(r_idx):
            start = self._site_offsets[row]
            end = self._site_offsets[row + 1]
            var_i = 0
            for v in range(start, end):
                pos = np.array([self._site_starts[v]], dtype=np.int32)
                alt = self._alt_data[self._alt_offsets[v] : self._alt_offsets[v + 1]]
                alt_off = np.array([0, len(alt)], dtype=np.int64)
                _, _, _, flg = apply_site_only_variants(
                    haps=haps.haps.view(np.uint8)[b : b + 1],
                    v_idxs=haps.var_idxs[b : b + 1],
                    ref_coords=haps.ref_coords[b : b + 1],
                    site_starts=pos,
                    alt_alleles=alt,
                    alt_offsets=alt_off,
                )
                flags[b, :, var_i] = flg[0]
                var_i += 1

        haps = AnnotatedHaps(
            haps=haps.haps.view("S1"),
            var_idxs=haps.var_idxs,
            ref_coords=haps.ref_coords,
        )

        if self.max_variants_per_region == 1:
            flags = flags[..., 0]

        if squeeze:
            haps = haps.squeeze(0)
            flags = flags.squeeze(0)

        if out_reshape is not None:
            haps = haps.reshape((*out_reshape, ploidy, length))
            if flags.ndim == 3:
                flags = flags.reshape(*out_reshape, ploidy, self.max_variants_per_region)
            else:
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
