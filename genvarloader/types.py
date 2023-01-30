from enum import Enum
from pathlib import Path
from typing import Union, cast

import numpy as np
from numpy.typing import NDArray

PathType = Union[str, Path]


class SequenceAlphabet:
    def __init__(self, alphabet: str, complement: str) -> None:
        """Parse and validate sequence alphabets.

        All alphabets must:
        1. Include N at the end.
        2. Be complemented by being reversed (without N).
            For example, `reverse(ACGT) = complement(ACGT) = TGCA`

        Parameters
        ----------
        alphabet : str
            For example, DNA could be 'ACGTN'.
        complement : str
            Complement of the alphabet, to continue the example this would be 'TGCAN'.
        """
        self.string = alphabet
        self.array = cast(
            NDArray[np.bytes_], np.frombuffer(alphabet.encode("ascii"), "|S1")
        )
        self.complement_map = dict(zip(list(alphabet), list(complement)))
        self.complement_map_bytes = {
            k.encode(): v.encode() for k, v in self.complement_map.items()
        }
        self.validate()

    def validate(self):
        if len(set(self.string)) != len(self.string):
            raise ValueError("Alphabet has repeated characters.")

        try:
            n_index = self.string.index("N")
        except ValueError:
            raise ValueError("N is not in the alphabet.")

        if n_index != (len(self.string) - 1):
            raise ValueError("N is not at the end of the alphabet.")

        without_N = self.string[:-1] if "N" in self.string else self.string
        for character, maybe_complement in zip(without_N, without_N[::-1]):
            if self.complement_map[character] != maybe_complement:
                raise ValueError("Reverse of alphabet does not yield the complement.")


ALPHABETS = {
    "DNA": SequenceAlphabet("ACGTN", "TGCAN"),
    "RNA": SequenceAlphabet("ACGUN", "UGCAN"),
}


class SequenceEncoding(str, Enum):
    BYTES = "bytes"
    ONEHOT = "onehot"


class Tn5CountMethod(str, Enum):
    CUTSITE = "cutsite"
    MIDPOINT = "midpoint"
    FRAGMENT = "fragment"
