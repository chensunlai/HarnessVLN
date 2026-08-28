from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

from schemas import JsonObject


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    media_type: str


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    name: str
    metadata: JsonObject
    artifacts: tuple[Artifact, ...]
    output_errors: tuple[str, ...] = ()

    def as_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "output_errors": list(self.output_errors),
        }


class ComponentOutput:
    """A component-owned output scope. A missing root is a valid null sink."""

    def __init__(self, name: str, root: Path | None) -> None:
        self.name = name
        self.root = root
        self._metadata: JsonObject = {}
        self._artifacts: dict[str, Artifact] = {}
        self._lock = Lock()
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def set_metadata(self, **values: Any) -> None:
        with self._lock:
            self._metadata.update(values)

    def write_json(self, relative_path: str, value: Any) -> Path | None:
        path = self.path(relative_path)
        if path is not None:
            _write_json(path, value)
        return path

    def append_jsonl(self, relative_path: str, value: Any) -> Path | None:
        path = self.path(relative_path)
        if path is not None:
            line = json.dumps(_json_value(value), ensure_ascii=True, sort_keys=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.write("\n")
        return path

    def path(self, relative_path: str) -> Path | None:
        relative = _relative_path(relative_path)
        if self.root is None:
            return None
        path = self.root.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def add_artifact(self, relative_path: str, media_type: str) -> None:
        relative = _relative_path(relative_path).as_posix()
        if not media_type.strip():
            raise ValueError("artifact media_type must not be empty")
        with self._lock:
            self._artifacts[relative] = Artifact(relative, media_type)

    def manifest(self) -> ComponentManifest:
        with self._lock:
            artifacts: list[Artifact] = []
            errors: list[str] = []
            if self.root is not None:
                for relative, artifact in sorted(self._artifacts.items()):
                    if (self.root / relative).is_file():
                        artifacts.append(artifact)
                    else:
                        errors.append(f"missing declared artifact: {relative}")
            return ComponentManifest(
                name=self.name,
                metadata=dict(self._metadata),
                artifacts=tuple(artifacts),
                output_errors=tuple(errors),
            )


class DomainOutput:
    def __init__(self, root: str | Path | None, domain_id: str) -> None:
        self.root = Path(root).expanduser().resolve() / domain_id if root else None
        self.domain_id = domain_id
        self._components: dict[str, ComponentOutput] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def component(self, name: str) -> ComponentOutput:
        if name not in self._components:
            root = self.root / "components" / name if self.root is not None else None
            self._components[name] = ComponentOutput(name, root)
        return self._components[name]

    def finish(self, result: Any, calls: list[JsonObject]) -> None:
        if self.root is None:
            return
        manifests = {
            name: output.manifest().as_dict()
            for name, output in sorted(self._components.items())
        }
        _write_json(self.root / "domain.json", result)
        _write_json(self.root / "components.json", manifests)
        calls_path = self.root / "calls.jsonl"
        with calls_path.open("w", encoding="utf-8") as stream:
            for call in calls:
                stream.write(json.dumps(_json_value(call), ensure_ascii=True, sort_keys=True))
                stream.write("\n")


def write_json(path: str | Path, value: Any) -> None:
    _write_json(Path(path), value)


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"output path must be a safe relative path: {value!r}")
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_value(value),
                stream,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_value(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_value(value.as_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
