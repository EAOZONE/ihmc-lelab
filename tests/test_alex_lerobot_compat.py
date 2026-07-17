import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_module():
    module_path = Path(__file__).parents[1] / "docker" / "alex_lerobot_compat.py"
    spec = importlib.util.spec_from_file_location("alex_lerobot_compat_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stats_compat_repairs_json_nan_std_from_range() -> None:
    module = _load_module()

    stats = module._repair_nonfinite_json_stats(
        {
            "observation.state": {
                "min": [1.0, 2.0],
                "max": [1.0, 4.0],
                "mean": [1.0, 3.0],
                "std": ["NaN", 0.25],
                "count": [10],
            }
        }
    )
    std = stats["observation.state"]["std"]
    assert std.dtype.kind == "f"
    assert np.allclose(std, [0.0, 0.25])


def test_policy_feature_filter_uses_state_only_for_tdmpc() -> None:
    module = _load_module()
    state = SimpleNamespace(type=SimpleNamespace(name="STATE"))
    image = SimpleNamespace(type=SimpleNamespace(name="VISUAL"))

    filtered = module._alex_filtered_policy_input_features(
        "tdmpc",
        {
            "observation.state": state,
            "observation.images.cam_zed_left": image,
            "observation.images.cam_zed_right": image,
        },
    )

    assert filtered == {"observation.state": state}


def test_policy_feature_filter_uses_one_camera_for_vqbet() -> None:
    module = _load_module()
    state = SimpleNamespace(type=SimpleNamespace(name="STATE"))
    left = SimpleNamespace(type=SimpleNamespace(name="VISUAL"))
    right = SimpleNamespace(type=SimpleNamespace(name="VISUAL"))

    filtered = module._alex_filtered_policy_input_features(
        "vqbet",
        {
            "observation.state": state,
            "observation.images.cam_zed_right": right,
            "observation.images.cam_zed_left": left,
        },
    )

    assert filtered == {
        "observation.state": state,
        "observation.images.cam_zed_left": left,
    }


def test_policy_factory_patch_rebinds_public_package_make_policy(monkeypatch) -> None:
    module = _load_module()

    def original_make_policy(**kwargs):
        return kwargs

    factory = SimpleNamespace(make_policy=original_make_policy, dataset_to_policy_features=lambda features: features)
    policies = SimpleNamespace(factory=factory, make_policy=original_make_policy)
    train = SimpleNamespace(make_policy=original_make_policy)
    monkeypatch.setitem(sys.modules, "lerobot", SimpleNamespace(policies=policies))
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setitem(sys.modules, "lerobot.scripts.lerobot_train", train)

    module._patch_policy_factory()

    assert factory.make_policy is not original_make_policy
    assert policies.make_policy is factory.make_policy
    assert train.make_policy is factory.make_policy


def test_video_timestamp_patch_relaxes_pyav_tolerance(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def decode_video_frames(video_path, timestamps, tolerance_s, backend=None, return_uint8=False, is_depth=False):
        calls.append((tolerance_s, backend, return_uint8, is_depth))
        return "frames"

    def decode_video_frames_pyav(
        video_path,
        timestamps,
        tolerance_s,
        log_loaded_timestamps=False,
        return_uint8=False,
        is_depth=False,
    ):
        calls.append((tolerance_s, "pyav-direct", log_loaded_timestamps, return_uint8, is_depth))
        return "pyav-frames"

    video_utils = SimpleNamespace(
        decode_video_frames=decode_video_frames,
        decode_video_frames_pyav=decode_video_frames_pyav,
    )
    datasets = SimpleNamespace(video_utils=video_utils)
    monkeypatch.setitem(sys.modules, "lerobot", SimpleNamespace(datasets=datasets))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", video_utils)

    module._patch_video_timestamp_tolerance()

    assert video_utils.decode_video_frames("video.mp4", [1.0], 0.0001, "pyav") == "frames"
    assert video_utils.decode_video_frames_pyav("video.mp4", [1.0], 0.0001) == "pyav-frames"
    assert calls[0][0] == module.ALEX_VIDEO_TIMESTAMP_TOLERANCE_S
    assert calls[1][0] == module.ALEX_VIDEO_TIMESTAMP_TOLERANCE_S


def test_video_timestamp_patch_keeps_larger_tolerance(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def decode_video_frames(video_path, timestamps, tolerance_s, backend=None, return_uint8=False, is_depth=False):
        calls.append((tolerance_s, backend))
        return "frames"

    video_utils = SimpleNamespace(
        decode_video_frames=decode_video_frames,
        decode_video_frames_pyav=lambda *args, **kwargs: None,
    )
    datasets = SimpleNamespace(video_utils=video_utils)
    monkeypatch.setitem(sys.modules, "lerobot", SimpleNamespace(datasets=datasets))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", video_utils)

    module._patch_video_timestamp_tolerance()

    assert video_utils.decode_video_frames("video.mp4", [1.0], 0.1, "pyav") == "frames"
    assert calls == [(0.1, "pyav")]
