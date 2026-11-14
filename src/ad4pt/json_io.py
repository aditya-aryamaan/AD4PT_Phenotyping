"""Standard-library JSON and local-file helpers."""

import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def open_regular(path: Path, flags: int) -> int:
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"must be a regular file: {path}")
    return fd


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def decode(data: bytes) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    return value


def save_json(
    path: Path, record: dict[str, Any], *, overwrite: bool = True
) -> None:
    """Replace only after the complete JSON has been written successfully."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".ad4pt-", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic creation keeps simultaneous runs from replacing each other.
            os.link(temporary, path)
            os.unlink(temporary)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def save_run_json(
    directory: Path, started: datetime, record: dict[str, Any]
) -> Path:
    """Readable New Zealand start time, with a suffix for same-second runs."""
    local = started.astimezone(ZoneInfo("Pacific/Auckland"))
    name = f"strawberry-counts_{local.strftime('%Y-%m-%d_%H-%M-%S_%Z')}"
    number = 1
    while True:
        suffix = "" if number == 1 else f"_{number:02d}"
        path = directory / f"{name}{suffix}.json"
        try:
            save_json(path, record, overwrite=False)
            return path
        except FileExistsError:
            number += 1
