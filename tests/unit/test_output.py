import json

import pytest

from harness.output import ComponentOutput


def test_component_output_is_scoped_and_reports_only_declared_artifacts(tmp_path):
    output = ComponentOutput("agent", tmp_path / "agent")
    output.set_metadata(model="dummy")
    output.write_json("trace/result.json", {"ok": True})
    output.add_artifact("trace/result.json", "application/json")

    assert json.loads((tmp_path / "agent" / "trace" / "result.json").read_text()) == {
        "ok": True
    }
    manifest = output.manifest()
    assert manifest.metadata == {"model": "dummy"}
    assert manifest.artifacts[0].path == "trace/result.json"


@pytest.mark.parametrize("path", ["", "/absolute", "../outside", "a/../../outside"])
def test_component_output_rejects_paths_outside_its_scope(tmp_path, path):
    output = ComponentOutput("agent", tmp_path / "agent")
    with pytest.raises(ValueError):
        output.path(path)


def test_component_manifest_does_not_claim_a_missing_artifact(tmp_path):
    output = ComponentOutput("agent", tmp_path / "agent")
    output.add_artifact("missing.json", "application/json")

    manifest = output.manifest()
    assert manifest.artifacts == ()
    assert manifest.output_errors == ("missing declared artifact: missing.json",)
