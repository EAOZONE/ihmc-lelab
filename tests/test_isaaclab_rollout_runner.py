import argparse
import io
import json
import sys
import types

import numpy as np
import torch
import pytest

import lelab.isaaclab_rollout_runner as runner
from lelab.isaaclab_rollout_runner import (
    _flag_any,
    _lever_angle,
    _reward_mean,
    observation_to_policy_features,
)


class _FakeData:
    joint_pos = torch.tensor([[0.1, -0.7]])


class _FakeLever:
    data = _FakeData()


class _FakeScene:
    def __getitem__(self, name: str):
        if name != "lever":
            raise KeyError(name)
        return _FakeLever()


class _FakeEnv:
    scene = _FakeScene()
    unwrapped = None


def test_lever_angle_reads_max_abs_joint_position() -> None:
    env = _FakeEnv()
    env.unwrapped = env
    assert _lever_angle(env) == pytest.approx(0.7)


def test_observation_mapping_keeps_cameras_and_lerobot_keys() -> None:
    obs = {
        "policy": np.zeros((1, 48), dtype=np.float32),
        "images": {
            "cam_zed_left": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "cam_zed_right": np.full((1, 480, 640, 3), 255, dtype=np.uint8),
        },
    }
    manifest = {
        "camera_prefix": "observation.images.",
        "policy_schema": {
            "input_features": {
                "observation.images.cam_zed_left": {"shape": [3, 224, 224], "type": "VISUAL"},
                "observation.images.cam_zed_right": {"shape": [3, 224, 224], "type": "VISUAL"},
                "observation.state": {"shape": [48], "type": "STATE"},
            }
        },
    }
    features = observation_to_policy_features(obs, manifest)
    assert features["observation.state"].shape == (48,)
    assert features["observation.images.cam_zed_left"].shape == (480, 640, 3)
    assert features["observation.images.cam_zed_right"].dtype == np.uint8
    assert set(features) == {
        "observation.state",
        "observation.images.cam_zed_left",
        "observation.images.cam_zed_right",
    }


def test_flag_any_and_reward_mean_move_tensors_to_numpy() -> None:
    assert _flag_any(torch.tensor([False, True])) is True
    assert _flag_any(torch.tensor([False, False])) is False
    assert _reward_mean(torch.tensor([1.0, 3.0])) == pytest.approx(2.0)


def test_main_resets_policy_server_after_each_env_reset(monkeypatch) -> None:
    events: list[str] = []

    class _Env:
        unwrapped = types.SimpleNamespace(device=None)

        def reset(self):
            events.append("env-reset")
            return {"policy": np.zeros((1, 48), dtype=np.float32)}

        def step(self, _action):
            events.append("env-step")
            return {"policy": np.zeros((1, 48), dtype=np.float32)}, 0.0, True, False, {}

        def close(self):
            events.append("env-close")

    def _post(url: str, _body: bytes, _content_type: str) -> bytes:
        if url.endswith("/reset"):
            events.append("server-reset")
            return b'{"status":"ok"}'
        events.append("predict")
        output = io.BytesIO()
        np.save(output, np.zeros((1, 1, 46), dtype=np.float32), allow_pickle=False)
        return output.getvalue()

    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        types.SimpleNamespace(make=lambda _environment, cfg: _Env()),
    )
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        types.SimpleNamespace(parse_env_cfg=lambda _environment, device, num_envs: {"device": device}),
    )
    monkeypatch.setattr(runner, "_post", _post)

    runner.main(
        argparse.Namespace(
            environment="Isaac-Alex-Lever-Play-v0",
            remote_url="http://127.0.0.1:8766",
            rollout_manifest=json.dumps({"version": 1}),
            num_episodes=2,
            fps=30,
            language_instruction="",
            embodiment="",
            metrics_output=None,
            video_dir=None,
            video=False,
            camera_video=False,
        ),
        types.SimpleNamespace(device_id=0),
    )

    assert events == [
        "env-reset",
        "server-reset",
        "predict",
        "env-step",
        "env-reset",
        "server-reset",
        "predict",
        "env-step",
        "env-close",
    ]
