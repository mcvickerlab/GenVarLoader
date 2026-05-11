import math

import numpy as np
import pytest

from genvarloader._dataset._insertion_fill import (
    CONSTANT,
    FLANK_SAMPLE,
    INTERPOLATE,
    REPEAT_5P,
    REPEAT_5P_NORM,
    Constant,
    FlankSample,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
    lower,
)


def test_lower_all_strategies():
    strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        Constant(value=0.5),
        FlankSample(flank_width=3),
        Interpolate(order=2),
    ]
    ids, params = lower(strategies)
    assert ids.dtype == np.int8
    assert params.dtype == np.float64
    assert ids.tolist() == [
        REPEAT_5P,
        REPEAT_5P_NORM,
        CONSTANT,
        FLANK_SAMPLE,
        INTERPOLATE,
    ]
    assert params[2, 0] == 0.5
    assert params[3, 0] == 3.0
    assert params[4, 0] == 2.0
    # Repeat5p (index 0) and Repeat5pNormalized (index 1) have no params: assert zeros.
    assert np.all(params[0] == 0)
    assert np.all(params[1] == 0)


def test_lower_empty():
    ids, params = lower([])
    assert ids.shape == (0,)
    assert params.shape == (0, 1)


def test_constant_default_is_nan():
    assert math.isnan(Constant().value)


def test_flank_sample_negative_width_rejected():
    with pytest.raises(ValueError, match="flank_width must be >= 0"):
        FlankSample(flank_width=-1)


def test_interpolate_order_capped():
    Interpolate(order=1)
    Interpolate(order=3)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=4)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=0)


def test_lower_unknown_class_raises():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="Unknown InsertionFill subclass"):
        lower([Bogus()])  # type: ignore[list-item]


def test_insertion_fill_base_not_instantiable():
    from genvarloader._dataset._insertion_fill import InsertionFill

    with pytest.raises(TypeError, match="abstract"):
        InsertionFill()
