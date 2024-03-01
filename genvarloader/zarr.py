from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from .types import Reader
from .util import normalize_contig_name

try:
    import tensorstore as ts
    import zarr

    ZARR_TENSORSTORE_INSTALLED = True
except ImportError:
    ZARR_TENSORSTORE_INSTALLED = False


"""
Implementation note regarding using tensorstore with num_workers > 0
https://github.com/google/tensorstore/issues/61
TL;DR TensorStore is not Send.
"""


class ZarrTracks(Reader):
    chunked = True

    def __init__(self, name: str, path: Union[str, Path]):
        if not ZARR_TENSORSTORE_INSTALLED:
            raise ImportError(
                "ZarrTracks support requires zarr and tensorstore to be installed."
            )

        self.name = name
        self.path = Path(path)
        z = zarr.open_group(self.path, mode="r")
        self.contigs = cast(Dict[str, int], z.attrs["contigs"])
        self.sizes = cast(Dict[str, int], z.attrs["sizes"])
        self.coords = {d: np.asarray(z.attrs[d]) for d in self.sizes}
        self.samples = cast(Optional[List[str]], z.attrs.get("samples", None))
        self.ploidy = cast(Optional[int], z.attrs.get("ploidy", None))
        self.dtype = z[next(iter(self.contigs))].dtype
        # each is (s? p? l)
        self.tstores: Optional[Dict[str, Any]] = None

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        path: Union[str, Path],
        chunk_shape: Optional[Tuple[int, ...]] = None,
    ):
        if isinstance(path, str):
            path = Path(path)
        z = zarr.open_group(path)
        z.attrs["contigs"] = reader.contigs
        z.attrs["sizes"] = reader.sizes
        if "sample" in reader.coords:
            z.attrs["samples"] = reader.coords["sample"].tolist()
        if "ploidy" in reader.sizes:
            z.attrs["ploidy"] = reader.sizes["ploidy"]
        with tqdm(total=len(reader.contigs)) as pbar:
            for contig, e in reader.contigs.items():
                pbar.set_description(f"Writing {contig}")
                data = reader.read(contig, 0, e)
                if chunk_shape is None:
                    chunk_layout = (
                        ts.ChunkLayout(  # pyright: ignore[reportAttributeAccessIssue]
                            chunk_shape=(20,) * len(reader.sizes) + (int(5e4),)
                        )
                    )
                else:
                    chunk_layout = (
                        ts.ChunkLayout(  # pyright: ignore[reportAttributeAccessIssue]
                            chunk_shape=chunk_shape
                        )
                    )
                tstore = ts.open(  # pyright: ignore[reportAttributeAccessIssue]
                    {
                        "driver": "zarr",
                        "kvstore": {"driver": "file", "path": str(path / contig)},
                        "metadata": {
                            "compressor": {
                                "id": "blosc",
                                "shuffle": -1,
                                "clevel": 5,
                                "cname": "lz4",
                            }
                        },
                    },
                    read=True,
                    write=True,
                    create=True,
                    delete_existing=True,
                    dtype=data.dtype,
                    chunk_layout=chunk_layout,
                    shape=data.shape,
                ).result()
                tstore.write(data).result()
                pbar.update()
        zarr.consolidate_metadata(str(path))  # pyright: ignore[reportArgumentType]

        return cls(reader.name, path)

    def _tstore(self, contig: str):
        tstore = ts.open(  # pyright: ignore[reportAttributeAccessIssue]
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(self.path / contig)},
            },
            read=True,
        ).result()
        return tstore

    def read(
        self, contig: str, starts: ArrayLike, ends: ArrayLike, **kwargs
    ) -> NDArray:
        if self.tstores is None:
            self.tstores = {contig: self._tstore(contig) for contig in self.contigs}

        contig = normalize_contig_name(
            contig, self.contigs
        )  # pyright: ignore[reportAssignmentType]
        if contig is None:
            raise ValueError(f"Contig {contig} not found")

        tstore = self.tstores[contig]
        samples = cast(Optional[List[str]], kwargs.get("samples", self.samples))
        if samples is not None:
            if self.samples is None:
                raise ValueError("No sample information available")
            elif missing := set(samples).difference(self.samples):
                raise ValueError(f"Samples {missing} were not found")
            key_idx, query_idx = np.intersect1d(
                self.samples, samples, return_indices=True
            )[1:]
            sample_idx = key_idx[query_idx]
            # (s p? l)
            tstore = tstore[sample_idx]

        ploidy = cast(Optional[ArrayLike], kwargs.get("ploidy", None))
        if ploidy is not None:
            if self.ploidy is None:
                raise ValueError("No ploidy information available")
            else:
                haplotype_idx = np.asarray(ploidy, dtype=int)
                if (haplotype_idx >= self.ploidy).any():
                    raise ValueError("Ploidies requested exceed maximum ploidy")
                # (s? p l)
                if self.samples is None:
                    # (p l)
                    tstore = tstore[haplotype_idx]
                else:
                    # (s p l)
                    tstore = tstore[:, haplotype_idx]

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        sub_values = [None] * len(starts)
        for i, (s, e) in enumerate(zip(starts, ends)):
            # no variants in query regions
            if s == e:
                continue
            # (s? p? l)
            sub_values[i] = tstore[..., s:e]

        values = ts.concat(  # pyright: ignore[reportAttributeAccessIssue]
            sub_values, axis=-1
        )[
            ts.d[0].translate_to[0]  # pyright: ignore[reportAttributeAccessIssue]
        ]

        values = cast(NDArray, values.read().result())

        return values
