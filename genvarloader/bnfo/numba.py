from typing import Optional, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize("(n),(),()->(n)", target="parallel", cache=True)
def gufunc_multi_slice(
    arr: NDArray,
    start: Union[int, NDArray[np.integer]],
    length: int,
    res: Optional[NDArray] = None,
) -> NDArray:  # type: ignore
    res[:length] = arr[start : start + length]  # type: ignore
