from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..types import Idx

if TYPE_CHECKING:
    from ..dataset import Dataset

try:
    import torch
    import torch.utils.data as td
except ImportError:
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


class TorchDataset(td.Dataset):  # type: ignore
    def __init__(self, dataset: "Dataset"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: Idx
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", ...], Any]:
        batch = self.dataset[idx]
        if isinstance(batch, np.ndarray):
            batch = _tensor_from_maybe_bytes(batch)
        elif isinstance(batch, tuple):
            batch = tuple(_tensor_from_maybe_bytes(b) for b in batch)
        return batch


def _tensor_from_maybe_bytes(array: NDArray) -> "torch.Tensor":
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Could not import PyTorch. Please install PyTorch to use torch features."
        )

    if array.dtype.type == np.bytes_:
        array = array.view(np.uint8)
    return torch.from_numpy(array)  # type: ignore
