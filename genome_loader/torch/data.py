from functools import partial, reduce
from importlib import import_module
from itertools import product
from textwrap import dedent
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import einops as ein
import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import nn
from torch.utils.data import BatchSampler, Dataset, Sampler
from torchvision.transforms import Compose
from tqdm import tqdm

from genome_loader.gloader import GenomeLoader
from genome_loader.gloader.consensus import ConsensusGenomeLoader
from genome_loader.utils import IndexType, PathType, read_bed


class ConsensusGLDataset(Dataset):
    def __init__(
        self,
        gl: ConsensusGenomeLoader,
        bed_file: PathType,
        length: int,
        region_idx: Optional[IndexType] = None,
        samples: Optional[
            Union[List[str], NDArray[np.str_], NDArray[np.uint32]]
        ] = None,
        ploid_idx: Optional[NDArray[np.uint32]] = None,
        pad_val: Union[bytes, str] = "N",
        transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """A GenomeLoader dataset of consensus sequences i.e. after putting variants
        into a reference sequence. Returns a tensor flattened over regions, samples, and ploidy.

        This class should be used exclusively with `NDimSampler` or other samplers
        that can yield 3-tuples. This does not implement the typical interface for
        `Dataset.__getitem__()` (takes 3-tuples) so is not compatible with e.g.
        `Subset` or any PyTorch samplers.
        """
        super().__init__()
        self.gl = gl
        self.bed = read_bed(bed_file, region_idx).collect()
        self.length = length
        if samples is None:
            self.samples: Union[NDArray[np.str_], NDArray[np.uint32]] = self.gl._vcf[
                "samples"
            ].to_numpy()
        elif isinstance(samples, list):
            self.samples = np.asarray(samples)
        else:
            self.samples = samples
        self.ploid_idx: NDArray[np.uint32] = (
            np.arange(self.gl._vcf.dims["ploidy"], dtype="u4")
            if ploid_idx is None
            else ploid_idx
        )
        self.n_regions = len(self.bed)
        self.n_samples = len(self.samples)
        self.n_ploidy = len(self.ploid_idx)
        self.pad_val = pad_val
        self.transforms = Compose(transforms if transforms is not None else [])
        self._dtype = dtype if dtype is not None else torch.float32

    def __len__(self):
        return self.n_regions * self.n_samples * self.n_ploidy

    def __getitem__(
        self, index: Tuple[List[int], List[int], List[int]]
    ) -> torch.Tensor:
        region_idx, _sample_idx, _ploid_idx = index
        sample_idx: Union[NDArray[np.str_], NDArray[np.uint32]] = self.samples[
            _sample_idx
        ]
        ploid_idx: NDArray[np.uint32] = self.ploid_idx[_ploid_idx]
        sequences = torch.from_numpy(
            self.gl.sel_from_bed(
                self.bed, self.length, region_idx, sample_idx, ploid_idx, self.pad_val
            )
        )
        sequences = ein.rearrange(sequences, "r s p l a -> (r s p) l a")
        return self.transforms(sequences).to(dtype=self.dtype)

    def __repr__(self) -> str:
        msg = dedent(
            f"""
        ConsensusGLDataset with {len(self)} consensus sequences.
        Regions: {self.n_regions}
        Samples: {self.n_samples}
        Ploidy: {self.n_ploidy}
        """
        ).strip()
        return msg

    @property
    def shape(self) -> Tuple[int, ...]:
        shape = [self.n_regions, self.n_samples, self.n_ploidy, self.length]
        if self.gl.spec is not None:
            shape.append(len(self.gl.spec))
        return tuple(shape)

    @property
    def dtype(self) -> torch.dtype:
        """Returns the PyTorch datatype that this dataset outputs."""
        return self._dtype


class GLDropN(nn.Module):
    def __init__(self, gl: GenomeLoader) -> None:
        super().__init__()
        assert hasattr(gl, "spec"), "Genome is not one-hot encoded."
        if b"N" not in gl.spec:  # type: ignore
            raise ValueError("N is not part of the genome loader's alphabet.")

    def forward(self, x: torch.Tensor):
        return x[..., :-1]


class NDimSampler(Sampler):
    def __init__(
        self,
        dataset_shape: Tuple[int, ...],
        batch_shape: Tuple[Union[int, None], ...],
        samplers: Union[str, Sampler, List[Union[str, Sampler]]] = "RandomSampler",
        drop_last: bool = False,
    ) -> None:
        """A PyTorch `Sampler` for orthogonal indexing over multiple dimensions.

        Parameters
        ----------
        dataset_shape : tuple[int]
            Shape of the dataset.
        batch_shape : tuple[int | None]
            Shape of the batch dimension (which can be multi-dimensional).
        samplers : str | Sampler | list[str | Sampler]
            Samplers to use for each dimension (if multiple are provided).
            If one is provided, uses it for all dimensions.
        drop_last : bool
            Drop the last batch if it's not full. Default False

        Notes
        -----
        If you're using a slice-able dataset, make sure to set `batch_size=None` for `DataLoader`
        since collating is inefficient with a dataset that can be sliced. In addition, you may
        want to set `num_workers=1` if slicing the dataset is multithreaded.
        ```python
        DataLoader(
            SliceableDataset(...),
            sampler=NDimSampler(...),
            batch_size=None,
            num_workers=1,
            ...
        )
        ```
        See [this forum post](https://discuss.pytorch.org/t/dataloader-sample-by-slices-from-dataset/113005/5) for more info.

        Examples
        --------
        To get indices corresponding to contiguous blocks from a tensor, use `SequentialSampler`:
        ```python
        sampler = NDimSampler(..., samplers = 'SequentialSampler')
        ```
        To get indices corresponding to random blocks from a tensor i.e. for orthogonal indexing:
        ```
        sampler = NDimSampler(..., samplers = 'RandomSampler')
        ```
        """

        self.dataset_shape = dataset_shape

        def normalize_batch_shape(batch_shape: Tuple[Union[int, None], ...]):
            _batch_shape: List[int] = []
            if len(batch_shape) > len(self.dataset_shape):
                raise ValueError("Got more batch dimensions than dataset dimensions.")
            for i, (length, batch_size) in enumerate(
                zip(self.dataset_shape, batch_shape)
            ):
                if batch_size is None:
                    _batch_shape.append(length)
                elif batch_size > length:
                    raise ValueError(
                        f"Dimension {i} of batch shape is larger than the same dimension in the dataset."
                    )
                else:
                    _batch_shape.append(batch_size)
            return _batch_shape

        self.batch_shape = normalize_batch_shape(batch_shape)

        def normalize_samplers(
            samplers: Union[str, Sampler, List[Union[str, Sampler]]]
        ):
            if not isinstance(samplers, list):
                if isinstance(samplers, str):
                    _samplers: List[Sampler] = [
                        getattr(import_module("torch.utils.data"), samplers)
                    ] * len(self.batch_shape)
                else:
                    _samplers = [samplers] * len(self.batch_shape)
            else:
                _samplers = []
                for sampler in samplers:
                    if isinstance(sampler, str):
                        _samplers.append(
                            getattr(import_module("torch.utils.data"), sampler)
                        )
                    else:
                        _samplers.append(sampler)
            return _samplers

        self.inner_samplers = normalize_samplers(samplers)
        if len(self.inner_samplers) != len(self.batch_shape):
            raise ValueError(
                "Must have as many samplers as there are dimensions in the batch."
            )

        self.batch_samplers: List[BatchSampler] = []
        for length, batch_size, sampler in zip(
            self.dataset_shape, self.batch_shape, self.inner_samplers
        ):
            self.batch_samplers.append(
                BatchSampler(sampler(torch.arange(length)), batch_size, drop_last)
            )
        self.batch_size: int = np.prod(self.batch_shape)

    def __iter__(self):
        return product(*self.batch_samplers)

    def __len__(self) -> int:
        return reduce(int.__mul__, map(len, self.batch_samplers))


class PredictToTensorStore(BasePredictionWriter):
    def __init__(
        self,
        ts_store,
        dataset_sample_dims: Optional[Tuple[int, ...]] = None,
        batch_reshape: Optional[Tuple[int, ...]] = None,
        progress: bool = False,
        write_interval: str = "batch",
    ) -> None:
        """Save model outputs to a TensorStore.

        Specify `batch_reshape` to reshape the batch dimension.

        Parameters
        ----------
        ts_store : tensorstore.TensorStore
            A synchronous TensorStore.
        dataset_sample_dims : Tuple[int, ...]
            Axes of the TensorStore that correspond to sample dimensions. i.e. the dimensions
            that are part of the batch dimension for the model.
            Suppose you have a dataset where the dimensions are (individual timepoint feature).
            This is fed to a model that takes a single (feature) vector such that one sample
            is a single individual, at a single timepoint. Then, dataset_sample_dims should be
            (0, 1) since dimensions 0 and 1 are sampled over.
        batch_reshape : Tuple[int, ...]
            Batch (i.e. first) dimension of model output will be reshaped to this shape.
        progress : bool
            Whether to provide a progress bar.
        write_interval : str
            How often to write predictions. Default "batch".
        """
        super().__init__(write_interval)
        self.ts_store = ts_store
        self.dataset_shape: Tuple[int, ...] = self.ts_store.shape
        self.dataset_sample_dims = dataset_sample_dims
        self.progress = progress
        self.batch_reshape = batch_reshape
        if self.batch_reshape is not None:
            if len(self.batch_reshape) > len(self.dataset_shape):
                raise ValueError(
                    "Reshaped batch dimension cannot have more dimensions than the dataset."
                )
            start_idxs = (
                range(0, d_len, b_len)
                for d_len, b_len in zip(self.dataset_shape, self.batch_reshape)
            )
            self.batch_slices = (
                [
                    slice(start, start + self.batch_reshape[i])
                    for i, start in enumerate(starts)
                ]
                for starts in product(*start_idxs)
            )
            b_dims = [f"b{i}" for i in range(len(self.batch_reshape))]
            b_dim_str = " ".join(b_dims)  # "b0 ... bn"
            self._reshape_str = f"({b_dim_str}) ... -> {b_dim_str} ..."
            self._reshape_dict = dict(zip(b_dims, self.batch_reshape))

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.progress:
            if self.dataset_sample_dims is not None:
                n_samples = np.prod(
                    [self.dataset_shape[dim] for dim in self.dataset_sample_dims]
                )
                self.pbar = tqdm(total=n_samples)
            else:
                self.pbar = tqdm()

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.batch_reshape is not None:
            slices = next(iter(self.batch_slices))
            self.ts_store[tuple(slices)] = ein.rearrange(
                prediction, self._reshape_str, **self._reshape_dict
            )
        else:
            start = batch_idx * len(prediction)
            self.ts_store[start : start + len(prediction)] = prediction

        if hasattr(self, "pbar"):
            self.pbar.update(len(prediction))

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if hasattr(self, "pbar"):
            self.pbar.close()

    def on_keyboard_interrupt(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if hasattr(self, "pbar"):
            self.pbar.close()

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        if hasattr(self, "pbar"):
            self.pbar.close()
