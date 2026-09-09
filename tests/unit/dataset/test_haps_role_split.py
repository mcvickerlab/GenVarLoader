"""The ``Haps`` role holds no storage, and the two implementations stay replaceable.

``Svar2Haps`` used to inherit ``Haps``' SVAR1 fields and fabricate empty values
for them at open time. Every caller that read one off the role got a silently
wrong answer on the SVAR2 backend rather than a type error -- #361 and #363 are
two instances. The split moved those fields down into ``Svar1Haps``; these tests
pin that they cannot come back, and that the settings block on the role still
round-trips through ``dataclasses.replace`` on both sides of the hierarchy.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC

import pytest

from genvarloader._dataset._haps import Haps, Svar1Haps
from genvarloader._dataset._svar2_haps import Svar2Haps

# The SVAR1 on-disk layout. Naming them here rather than deriving them keeps the
# test honest: adding one back to the role has to be a deliberate edit of this list.
SVAR1_STORAGE_FIELDS = ("variants", "genotypes", "dosages", "var_field_data")


def _field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_role_is_abstract():
    """``Haps`` is a role, not something you can open a dataset as."""
    assert issubclass(Haps, ABC)
    assert inspect.isabstract(Haps)
    with pytest.raises(TypeError):
        Haps()  # type: ignore[abstract]


@pytest.mark.parametrize("name", SVAR1_STORAGE_FIELDS)
def test_role_holds_no_svar1_storage(name):
    """Storage belongs to the implementation, so the role must not declare it."""
    assert name not in _field_names(Haps)
    assert name in _field_names(Svar1Haps)


@pytest.mark.parametrize("name", SVAR1_STORAGE_FIELDS + ("_ffi_static",))
def test_svar2_has_no_svar1_storage(name):
    """``Svar2Haps`` must not carry an SVAR1 field, fabricated or otherwise.

    ``slots=True`` makes this an attribute-level guarantee, not just a
    dataclass-field one: there is nowhere for a placeholder to live.
    """
    assert name not in _field_names(Svar2Haps)
    assert not hasattr(Svar2Haps, name)


def test_both_implementations_satisfy_the_role():
    """Neither implementation may be left with an unimplemented role member."""
    for cls in (Svar1Haps, Svar2Haps):
        assert issubclass(cls, Haps)
        assert not inspect.isabstract(cls), (
            f"{cls.__name__} does not implement: "
            f"{sorted(getattr(cls, '__abstractmethods__', ()))}"
        )


@pytest.mark.parametrize("cls", [Svar1Haps, Svar2Haps])
def test_settings_fields_are_replaceable(cls):
    """``with_settings`` and friends reach both classes through the same call.

    ``Dataset`` mutates the reconstructor exclusively via
    ``dataclasses.replace``, so a settings field that stopped being an
    ``init=True`` field on either side would break the builder chain at runtime
    rather than at import.
    """
    init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
    for name in (
        "kind",
        "min_af",
        "max_af",
        "var_fields",
        "dummy_variant",
        "flank_length",
        "token_lut",
        "token_dtype",
        "unknown_token",
        "token_alphabet",
        "window_opt",
        "unphased_union",
    ):
        assert name in init_fields, f"{cls.__name__}.{name} is not replace()-able"
