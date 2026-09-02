from __future__ import annotations

import sys

import pytest

from domain.contracts import WorkspaceSpec
from domain.errors import HarnessError
from domain.workspace import Workspace


def test_workspace_file_and_python_boundaries(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace", WorkspaceSpec(python=sys.executable))
    workspace.write_text("notes/value.txt", "ok")
    assert workspace.read_text("notes/value.txt") == "ok"
    result = workspace.python(["-c", "print('ready')"])
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ready"
    with pytest.raises(HarnessError):
        workspace.read_text("../outside.txt")
