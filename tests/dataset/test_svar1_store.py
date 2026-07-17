import pytest
from genvarloader.genvarloader import Svar1Store  # the compiled extension module


def test_missing_store_errors():
    with pytest.raises(Exception):
        Svar1Store("/no/such/svar", ["chr1"], 2, 2)
