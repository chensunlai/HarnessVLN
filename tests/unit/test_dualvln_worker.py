from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vln.dualvln.worker import (
    DualVLNBackend,
    NativeDualPolicy,
    _require_action,
    _require_rgbd,
    _restore_diffusers_gradient_checkpointing,
    build_model_settings,
)


class Policy:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.loaded = None
        self.reset_count = 0
        self.observations = []
        self.closed = False

    def load(self, upstream_root, checkpoint, options):
        self.loaded = (upstream_root, checkpoint, dict(options))
        return {"width": 640, "height": 480}

    def reset(self):
        self.reset_count += 1

    def step(self, observations):
        self.observations.append(observations)
        return next(self.outputs)

    def close(self):
        self.closed = True


class Tools:
    def __init__(self, observations=None):
        self.observations = iter(observations or [rgbd()] * 20)
        self.moves = []

    def observe(self):
        return next(self.observations)

    def move_discrete(self, action):
        self.moves.append(action)
        return {"action": action}


def rgbd():
    return {
        "channels": {
            "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "depth": np.zeros((480, 640, 1), dtype=np.float32),
        }
    }


def output(action, *, ideal=True):
    return [{"action": [action], "ideal_flag": ideal}]


def load(backend, *, max_steps=10, model_settings=None):
    backend.load(
        {
            "upstream_root": "/upstream",
            "checkpoint": "/checkpoint",
            "options": {
                "max_steps": max_steps,
                "model_settings": model_settings or {},
            },
        }
    )


def test_dual_backend_preserves_agent_state_and_maps_discrete_actions() -> None:
    policy = Policy([output(-1), output(1), output(2), output(3), output(0), output(0)])
    backend = DualVLNBackend(policy)
    tools = Tools()
    load(backend)

    assert backend.navigate("go to the chair", {}, tools, threading.Event()) == {
        "state": "succeeded",
        "steps": 4,
        "reason": "model emitted STOP",
    }
    assert backend.navigate("next goal", {}, tools, threading.Event()) == {
        "state": "succeeded",
        "steps": 0,
        "reason": "model emitted STOP",
    }
    backend.close()

    assert tools.moves == ["stand_still", "forward", "turn_left", "turn_right"]
    assert policy.reset_count == 2
    assert [item[0]["instruction"] for item in policy.observations] == [
        "go to the chair",
        "go to the chair",
        "go to the chair",
        "go to the chair",
        "go to the chair",
        "next goal",
    ]
    assert all(len(item) == 1 for item in policy.observations)
    assert policy.closed


def test_dual_cancel_after_inference_fences_motion() -> None:
    cancelled = threading.Event()

    class CancellingPolicy(Policy):
        def step(self, observations):
            value = super().step(observations)
            cancelled.set()
            return value

    backend = DualVLNBackend(CancellingPolicy([output(1)]))
    tools = Tools()
    load(backend)

    assert backend.navigate("go", {}, tools, cancelled) == {
        "state": "cancelled",
        "steps": 0,
        "reason": "job cancelled",
    }
    assert tools.moves == []


def test_dual_step_limits_are_bounded_and_report_normal_completion() -> None:
    backend = DualVLNBackend(Policy([output(1), output(1)]))
    load(backend, max_steps=2)

    assert backend.navigate("go", {"max_steps": 1}, Tools(), threading.Event()) == {
        "state": "limit_reached",
        "steps": 1,
        "reason": "step limit reached: 1",
    }
    with pytest.raises(ValueError, match="exceeds worker limit"):
        backend.navigate("go", {"max_steps": 3}, Tools(), threading.Event())


@pytest.mark.parametrize(
    "bad_output, message",
    [
        ([], "one-item list"),
        ([{"action": [1], "ideal_flag": False}], "velocity output"),
        ([{"action": [True], "ideal_flag": True}], "integer list"),
        ([{"action": [1, 2], "ideal_flag": True}], "integer list"),
        ([{"ideal_flag": True}], "integer list"),
    ],
)
def test_dual_output_schema_is_strict(bad_output, message) -> None:
    with pytest.raises(ValueError, match=message):
        _require_action(bad_output)


@pytest.mark.parametrize(
    "channels, message",
    [
        ({"depth": np.zeros((480, 640, 1), np.float32)}, "requires rgb"),
        (
            {
                "rgb": np.zeros((480, 640, 3), np.float32),
                "depth": np.zeros((480, 640, 1), np.float32),
            },
            "uint8 RGB",
        ),
        (
            {
                "rgb": np.zeros((480, 640, 3), np.uint8),
                "depth": np.zeros((480, 640), np.float32),
            },
            "depth shape",
        ),
        (
            {
                "rgb": np.zeros((480, 640, 3), np.uint8),
                "depth": np.full((480, 640, 1), 2.0, np.float32),
            },
            "normalized",
        ),
    ],
)
def test_dual_rejects_incompatible_observations(channels, message) -> None:
    backend = DualVLNBackend(Policy([output(0)]))
    load(backend)
    with pytest.raises(ValueError, match=message):
        backend.navigate(
            "go", {}, Tools([{"channels": channels}]), threading.Event()
        )


def test_dual_converts_environment_depth_range_to_model_normalization() -> None:
    depth = np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32)
    observation = {
        "channels": {
            "rgb": np.zeros((1, 3, 3), dtype=np.uint8),
            "depth": depth,
            "depth_metadata": {
                "encoding": "linear_normalized",
                "minimum_m": 0.5,
                "maximum_m": 5.0,
            },
        }
    }

    _, converted = _require_rgbd(observation, 1, 3)

    assert converted.dtype == np.float32
    assert converted is not depth
    np.testing.assert_allclose(converted[:, :, 0], [[0.05, 0.275, 0.5]])


@pytest.mark.parametrize(
    "metadata",
    [
        {"encoding": "metric", "minimum_m": 0.5, "maximum_m": 5.0},
        {"encoding": "linear_normalized", "minimum_m": 5.0, "maximum_m": 0.5},
        {"encoding": "linear_normalized", "minimum_m": 0.0, "maximum_m": 12.0},
    ],
)
def test_dual_rejects_unsupported_depth_metadata(metadata) -> None:
    observation = rgbd()
    observation["channels"]["depth_metadata"] = metadata

    with pytest.raises(ValueError, match="depth"):
        _require_rgbd(observation, 480, 640)


def test_dual_model_settings_keep_checkpoint_and_policy_authoritative() -> None:
    settings = build_model_settings(
        {"state_encoder": None, "policy_name": "base"},
        Path("/models/dual"),
        {
            "device": "cuda:4",
            "model_settings": {
                "model_path": "/wrong",
                "policy_name": "wrong",
                "num_history": 12,
            },
        },
    )

    assert settings["model_path"] == "/models/dual"
    assert settings["policy_name"] == "InternVLAN1_Policy"
    assert settings["device"] == "cuda:4"
    assert settings["num_history"] == 12
    assert settings["infer_mode"] == "partial_async"


def test_native_dual_loader_uses_internnav_agent_contract(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "InternNav"
    package_root = source_root / "internnav"
    package_root.mkdir(parents=True)
    checkpoint = tmp_path / "DualVLN"
    checkpoint.mkdir()
    depth_checkpoint = checkpoint / "depth_anything_v2_metric_hypersim_vits.pth"
    depth_checkpoint.touch()
    initialized = {}

    class AgentCfg:
        def __init__(self, **values):
            self.values = values

    class Agent:
        @classmethod
        def init(cls, config):
            initialized["config"] = config
            return SimpleNamespace(reset=lambda _: None, step=lambda _: output(0))

    class ModelConfig:
        @staticmethod
        def model_dump():
            return {"state_encoder": None, "policy_name": "upstream-default"}

    class CurrentModelMixin:
        def _set_gradient_checkpointing(
            self, enable=True, gradient_checkpointing_func=None
        ):
            return (enable, gradient_checkpointing_func)

    class LegacyLumina(CurrentModelMixin):
        def _set_gradient_checkpointing(self, module, value=False):
            return (module, value)

    architecture = SimpleNamespace(MODEL_PATH_TO="checkpoints")

    modules = {
        "internnav.agent": SimpleNamespace(
            __file__=str(package_root / "agent" / "__init__.py"), Agent=Agent
        ),
        "internnav.configs.agent": SimpleNamespace(AgentCfg=AgentCfg),
        "internnav.configs.model": SimpleNamespace(internvla_n1_cfg=ModelConfig()),
        "internnav.model.basemodel.internvla_n1.nextdit_traj": SimpleNamespace(
            LuminaNextDiT2DModel=LegacyLumina
        ),
        "internnav.model.basemodel.internvla_n1.internvla_n1_arch": architecture,
    }
    monkeypatch.setattr(
        "vln.dualvln.worker.importlib.import_module", lambda name: modules[name]
    )

    settings = NativeDualPolicy().load(
        source_root,
        checkpoint,
        {"device": "cuda:3", "model_settings": {"model_path": "/wrong"}},
    )

    config = initialized["config"].values
    assert config["model_name"] == "internvla_n1"
    assert config["ckpt_path"] == str(checkpoint)
    assert config["model_settings"] == settings
    assert settings["model_path"] == str(checkpoint)
    assert settings["policy_name"] == "InternVLAN1_Policy"
    assert settings["device"] == "cuda:3"
    assert settings["continuous_traj"] is True
    assert "_set_gradient_checkpointing" not in LegacyLumina.__dict__
    assert architecture.MODEL_PATH_TO == str(checkpoint)


def test_dual_diffusers_patch_keeps_hook_required_by_old_base() -> None:
    class LegacyLumina:
        def _set_gradient_checkpointing(self, module, value=False):
            return (module, value)

    original = LegacyLumina.__dict__["_set_gradient_checkpointing"]
    _restore_diffusers_gradient_checkpointing(LegacyLumina)
    assert LegacyLumina.__dict__["_set_gradient_checkpointing"] is original
