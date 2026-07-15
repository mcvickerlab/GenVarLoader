"""Structured progress events and atomic JSON snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

from loguru import logger

ProgressState: TypeAlias = Literal["running", "complete", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One immutable update for a long-running artifact operation."""

    phase: str
    completed: int
    total: int | None
    unit: str
    state: ProgressState = "running"
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("phase must not be empty")
        if not self.unit:
            raise ValueError("unit must not be empty")
        if self.completed < 0:
            raise ValueError("completed must be >= 0")
        if self.total is not None:
            if self.total <= 0:
                raise ValueError("total must be > 0 when provided")
            if self.completed > self.total:
                raise ValueError("completed must not exceed total")

    @property
    def percent(self) -> float | None:
        """Return exact completion percent, or ``None`` when indeterminate."""
        if self.total is None:
            return None
        return 100.0 * self.completed / self.total


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]

_SNAPSHOT_NAME = ".nf-seqlab-progress.json"
_IDENTITY_ENV = {
    "run_id": "NF_SEQLAB_PROGRESS_RUN_ID",
    "stage_id": "NF_SEQLAB_PROGRESS_STAGE_ID",
    "process": "NF_SEQLAB_PROGRESS_PROCESS",
    "file_id": "NF_SEQLAB_PROGRESS_FILE_ID",
    "parent_file_id": "NF_SEQLAB_PROGRESS_PARENT_FILE_ID",
    "task_id": "NF_SEQLAB_PROGRESS_TASK_ID",
}


def _snapshot_path(
    progress_path: str | Path | None,
    environ: Mapping[str, str],
) -> Path | None:
    if progress_path is not None:
        return Path(progress_path)
    configured = (
        environ.get("NF_SEQLAB_PROGRESS_PATH")
        or environ.get("NF_SEQLAB_PROGRESS_FILE")
        or environ.get("NF_SEQLAB_PROGRESS_SNAPSHOT_PATH")
    )
    if configured:
        return Path(configured)
    if directory := environ.get("NF_SEQLAB_PROGRESS_DIR"):
        return Path(directory) / _SNAPSHOT_NAME
    shorthand = environ.get("NF_SEQLAB_PROGRESS")
    if shorthand and shorthand.lower() not in {"0", "false", "no", "off"}:
        if shorthand.lower() in {"1", "true", "yes", "on"}:
            return Path.cwd() / _SNAPSHOT_NAME
        return Path(shorthand)
    return None


class JsonProgressSink:
    """Atomically replace one ``nf-seqlab.progress/v1`` JSON snapshot."""

    def __init__(
        self,
        path: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        env = os.environ if environ is None else environ
        self._identity = {field: env.get(name) for field, name in _IDENTITY_ENV.items()}
        if self._identity["parent_file_id"] is None:
            self._identity["parent_file_id"] = self._identity["file_id"]
        attempt = env.get("NF_SEQLAB_PROGRESS_ATTEMPT")
        try:
            self._attempt = int(attempt) if attempt is not None else None
        except ValueError:
            logger.warning(f"Ignoring invalid NF_SEQLAB_PROGRESS_ATTEMPT={attempt!r}")
            self._attempt = None

    def __call__(self, event: ProgressEvent) -> None:
        payload = {
            "schema": "nf-seqlab.progress/v1",
            **self._identity,
            "attempt": self._attempt,
            "state": event.state,
            "phase": event.phase,
            "completed": event.completed,
            "total": event.total,
            "unit": event.unit,
            "percent": event.percent,
            "message": event.message,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


class _ProgressFanout:
    """Deliver to independent observers, disabling each one after a failure."""

    def __init__(self, callbacks: list[ProgressCallback]) -> None:
        self._callbacks = callbacks

    def __call__(self, event: ProgressEvent) -> None:
        active: list[ProgressCallback] = []
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.warning(f"Progress callback failed; disabling it: {exc!r}")
            else:
                active.append(callback)
        self._callbacks = active


def _resolve_progress(
    progress_callback: ProgressCallback | None = None,
    progress_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProgressCallback | None:
    """Combine an explicit callback with an explicit or environment JSON sink."""
    env = os.environ if environ is None else environ
    callbacks: list[ProgressCallback] = []
    if progress_callback is not None:
        callbacks.append(progress_callback)
    if path := _snapshot_path(progress_path, env):
        callbacks.append(JsonProgressSink(path, environ=env))
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return _ProgressFanout(callbacks)


__all__ = ["JsonProgressSink", "ProgressCallback", "ProgressEvent"]
