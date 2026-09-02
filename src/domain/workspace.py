from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain.contracts import WorkspaceSpec
from domain.errors import HarnessError
from domain.register import DomainRegister


class Workspace:
    """Episode-local filesystem and bounded command surface."""

    def __init__(self, root: str | Path, spec: WorkspaceSpec) -> None:
        self.root = Path(root).expanduser().resolve()
        self.spec = spec
        self.root.mkdir(parents=True, exist_ok=False)
        self._file_lock = threading.RLock()

    def mount(self, register: DomainRegister) -> None:
        register.register_reference(
            "workspace", "workspace.root", lambda: str(self.root), description="Workspace root"
        )
        register.register_function(
            "workspace",
            "workspace.read_text",
            self.read_text,
            description="Read a UTF-8 text file from the episode workspace.",
            parameters=_path_schema(),
        )
        register.register_function(
            "workspace",
            "workspace.write_text",
            self.write_text,
            description="Write a UTF-8 text file inside the episode workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                "required": ["path", "text"],
                "additionalProperties": False,
            },
            mutates=True,
            serial_key="workspace.files",
        )
        register.register_function(
            "workspace",
            "workspace.list",
            self.list,
            description="List files under a workspace directory.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        register.register_function(
            "workspace",
            "workspace.run",
            self.run,
            description="Run one argv command with the workspace as its filesystem scope.",
            parameters=_command_schema(),
            mutates=True,
            serial_key="workspace.process",
        )
        register.register_function(
            "workspace",
            "workspace.python",
            self.python,
            description="Run the configured Python interpreter in the episode workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["arguments"],
                "additionalProperties": False,
            },
            mutates=True,
            serial_key="workspace.process",
        )

    def module(self, name: str) -> "ModuleWorkspace":
        return ModuleWorkspace(self, f"modules/{name}")

    def read_text(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8")

    def write_text(self, path: str, text: str) -> dict[str, Any]:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            target.write_text(text, encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "bytes": len(text.encode())}

    def write_json(self, path: str, value: Any) -> Path:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def list(self, path: str = ".") -> list[str]:
        directory = self._path(path)
        if not directory.is_dir():
            raise HarnessError(f"workspace path is not a directory: {path}")
        return sorted(str(item.relative_to(self.root)) for item in directory.iterdir())

    def run(
        self,
        command: Sequence[str],
        cwd: str = ".",
        timeout_s: float | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if isinstance(command, (str, bytes)) or not command:
            raise HarnessError("workspace command must be a non-empty argv list")
        argv = [str(item) for item in command]
        timeout = self.spec.command_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise HarnessError("workspace command timeout must be positive")
        process_environment = os.environ.copy()
        process_environment.update(self.spec.environment)
        process_environment.update({str(k): str(v) for k, v in (environment or {}).items()})
        try:
            completed = subprocess.run(
                argv,
                cwd=self._path(cwd),
                env=process_environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise HarnessError(f"workspace command failed: {type(error).__name__}: {error}") from error
        limit = self.spec.max_output_chars
        return {
            "command": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout[:limit],
            "stderr": completed.stderr[:limit],
            "stdout_truncated": len(completed.stdout) > limit,
            "stderr_truncated": len(completed.stderr) > limit,
        }

    def python(
        self,
        arguments: Sequence[str],
        cwd: str = ".",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.run([self.spec.python, *(str(item) for item in arguments)], cwd, timeout_s)

    def _path(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise HarnessError(f"workspace path escapes episode root: {value!r}")
        return candidate


class ModuleWorkspace:
    def __init__(self, workspace: Workspace, prefix: str) -> None:
        self.workspace = workspace
        self.prefix = prefix
        self.root = workspace._path(prefix)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        return self.workspace._path(f"{self.prefix}/{relative}")

    def read_text(self, relative: str) -> str:
        return self.workspace.read_text(f"{self.prefix}/{relative}")

    def write_text(self, relative: str, text: str) -> dict[str, Any]:
        return self.workspace.write_text(f"{self.prefix}/{relative}", text)

    def write_json(self, relative: str, value: Any) -> Path:
        return self.workspace.write_json(f"{self.prefix}/{relative}", value)


def _path_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }


def _command_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "cwd": {"type": "string"},
            "timeout_s": {"type": "number", "exclusiveMinimum": 0},
            "environment": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["command"],
        "additionalProperties": False,
    }
