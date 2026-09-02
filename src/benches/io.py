from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from domain.errors import HarnessError


def load_json(path: str | Path) -> Any:
    source = Path(path).expanduser()
    try:
        if source.suffix == ".gz":
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                return json.load(stream)
        with source.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"failed to load dataset {source}: {error}") from error


def require_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    if missing:
        raise HarnessError(f"{label} is missing fields: {', '.join(missing)}")
    return value
