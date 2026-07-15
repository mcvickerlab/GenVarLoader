"""Safely publish a directory artifact via build-to-temp-then-atomic-rename.

The single primitive both the `.gvlfa` FASTA cache and `gvl.write` dataset
directories need: build into a private sibling temp dir, then publish it to the
destination. New destinations use one atomic `os.replace`. Portable overwrite of
a non-empty directory requires move-aside-then-rename, with rollback if publishing
the new directory fails. A best-effort `filelock` avoids redundant builds but is
not required for data integrity.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from filelock import FileLock, Timeout
from loguru import logger

__all__ = ["atomic_dir", "SkipPublish", "DEFAULT_LOCK_TIMEOUT"]

DEFAULT_LOCK_TIMEOUT = 60.0
"""Seconds to wait for the build lock before giving up and building anyway."""


class SkipPublish(Exception):  # noqa: N818 - control-flow sentinel, not an error
    """Raise inside an `atomic_dir` block to abort publishing.

    Use when a double-check inside the lock finds `dest` already valid: the temp
    dir is removed and `dest` is left untouched, with no exception surfacing to
    the caller.
    """


def _publish(
    tmp: Path,
    dest: Path,
    *,
    overwrite: bool,
    require_publish: bool,
) -> None:
    """Move `tmp` into place at `dest` as atomically as the filesystem allows."""
    if dest.exists():
        if not overwrite:
            # A racing builder published `dest` while we built. Discard ours and
            # Reuse callers can let the other publisher win when content is
            # interchangeable; strict callers raise below.
            shutil.rmtree(tmp, ignore_errors=True)
            if require_publish:
                raise FileExistsError(
                    f"{dest} was published by another writer before this build "
                    "completed."
                )
            return
        aside = dest.with_name(f"{dest.name}.old.{uuid4().hex[:8]}")
        try:
            os.replace(dest, aside)
        except FileNotFoundError:
            aside = None  # another writer already moved/removed dest
        try:
            os.replace(tmp, dest)
        except BaseException as publish_error:
            if aside is not None:
                try:
                    os.replace(aside, dest)
                except BaseException as rollback_error:
                    raise publish_error from rollback_error
            raise
        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
    else:
        os.replace(tmp, dest)


@contextmanager
def atomic_dir(
    dest: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    lock: bool = True,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    require_publish: bool = False,
) -> Iterator[Path]:
    """Yield a private temp dir to build into; atomically publish it to `dest`.

    On clean exit the temp dir is `os.replace`-d into `dest`. On `SkipPublish`
    the temp dir is removed and `dest` is left untouched. On any other exception
    the temp dir is removed and the exception propagates; `dest` is never
    partially written. During overwrite, readers can briefly observe ``dest`` as
    absent between the move-aside and publish renames. A publish failure restores
    the old directory; if rollback also fails, its sibling backup is preserved.

    Parameters
    ----------
    dest
        Final destination directory.
    overwrite
        If False and `dest` already exists, raise `FileExistsError` up front. If
        True, an existing `dest` is replaced via move-aside-then-rename.
    lock
        Acquire a best-effort `<dest>.lock` to avoid redundant concurrent builds.
    timeout
        Seconds to wait for the lock before logging and proceeding anyway.
    require_publish
        Raise :class:`FileExistsError` if another writer publishes ``dest``
        before this build. By default, discard this build and reuse the winner.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists; pass overwrite=True to replace it."
        )

    flock: FileLock | None = None
    if lock:
        flock = FileLock(str(dest) + ".lock")
        try:
            flock.acquire(timeout=timeout)
        except Timeout:
            logger.info(
                f"Timed out after {timeout}s waiting for {dest}.lock; "
                "building anyway (atomic rename keeps this correct)."
            )
            flock = None

    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}-{uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except SkipPublish:
        shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    else:
        try:
            _publish(
                tmp,
                dest,
                overwrite=overwrite,
                require_publish=require_publish,
            )
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    finally:
        if flock is not None:
            lock_path = Path(str(dest) + ".lock")
            flock.release()
            # Ensure the lock file persists as a sibling so future callers can
            # reuse it; filelock may remove it on release.
            lock_path.touch(exist_ok=True)
