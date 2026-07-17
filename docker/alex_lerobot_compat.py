"""Small runtime compatibility fixes for the pinned Alex LeRobot image."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

ALEX_STATS_SANITIZER = True
ALEX_POLICY_FEATURE_FILTER = True
ALEX_VIDEO_TIMESTAMP_TOLERANCE_PATCH = True
ALEX_VIDEO_TIMESTAMP_TOLERANCE_ENV = "ALEX_VIDEO_TIMESTAMP_TOLERANCE_S"
ALEX_VIDEO_TIMESTAMP_TOLERANCE_S = 0.08
ALEX_TEST_OBS_NEW_STATE_DIM = 48
ALEX_TEST_OBS_NEW_ACTION_DIM = 46
ALEX_TEST_OBS_NEW_PRIMARY_CAMERA = "observation.images.cam_zed_left"


def _repair_nonfinite_json_stats(converted: dict) -> dict:
    repaired: list[str] = []

    for feature_name, feature_stats in converted.items():
        if not isinstance(feature_stats, dict):
            continue

        # JSON has no portable NaN value. LeRobot's serializer writes "NaN",
        # which otherwise creates a numpy string array and later crashes
        # torch.as_tensor before a policy can decide which stats it needs.
        for stat_name, value in tuple(feature_stats.items()):
            array = np.asarray(value)
            if array.dtype.kind in {"O", "S", "U"}:
                try:
                    feature_stats[stat_name] = array.astype(np.float32)
                except (TypeError, ValueError):
                    continue

        std = feature_stats.get("std")
        minimum = feature_stats.get("min")
        maximum = feature_stats.get("max")
        if std is None or minimum is None or maximum is None:
            continue

        std_array = np.asarray(std, dtype=np.float32)
        invalid = ~np.isfinite(std_array)
        if not invalid.any():
            continue

        fallback = np.abs(
            np.asarray(maximum, dtype=np.float32) - np.asarray(minimum, dtype=np.float32)
        ) * 0.5
        if fallback.shape != std_array.shape:
            continue
        feature_stats["std"] = np.where(invalid, fallback, std_array)
        repaired.append(f"{feature_name}.std ({int(invalid.sum())} values)")

    if repaired:
        logging.warning(
            "Alex training image repaired non-finite dataset statistics in memory: %s. "
            "Recompute the dataset stats on Hugging Face for a permanent fix.",
            ", ".join(repaired),
        )
    return converted


def _is_alex_test_obs_new_features(features: Mapping[str, Any]) -> bool:
    state = features.get("observation.state")
    action = features.get("action")
    cameras = [
        name
        for name, spec in features.items()
        if name.startswith("observation.images.")
        or (isinstance(spec, Mapping) and spec.get("dtype") in {"image", "video"})
    ]
    return (
        isinstance(state, Mapping)
        and tuple(state.get("shape") or ()) == (ALEX_TEST_OBS_NEW_STATE_DIM,)
        and isinstance(action, Mapping)
        and tuple(action.get("shape") or ()) == (ALEX_TEST_OBS_NEW_ACTION_DIM,)
        and len(cameras) >= 2
    )


def _feature_type_name(feature: Any) -> str:
    feature_type = getattr(feature, "type", None)
    return str(getattr(feature_type, "name", feature_type))


def _alex_filtered_policy_input_features(policy_type: str, policy_features: Mapping[str, Any]) -> dict[str, Any] | None:
    state = policy_features.get("observation.state")
    if state is None:
        return None

    if policy_type == "tdmpc":
        return {"observation.state": state}

    if policy_type == "vqbet":
        visual_keys = [
            key
            for key, feature in policy_features.items()
            if _feature_type_name(feature) == "VISUAL" or key.startswith("observation.images.")
        ]
        if not visual_keys:
            return None
        camera_key = (
            ALEX_TEST_OBS_NEW_PRIMARY_CAMERA
            if ALEX_TEST_OBS_NEW_PRIMARY_CAMERA in visual_keys
            else sorted(visual_keys)[0]
        )
        return {
            "observation.state": state,
            camera_key: policy_features[camera_key],
        }

    return None


def _patch_stats_loader() -> None:
    try:
        from lerobot.datasets import io_utils
    except ModuleNotFoundError:
        return

    original_cast_stats_to_numpy = io_utils.cast_stats_to_numpy

    def _cast_stats_to_numpy_with_nonfinite_json_support(stats: dict) -> dict:
        return _repair_nonfinite_json_stats(original_cast_stats_to_numpy(stats))

    io_utils.cast_stats_to_numpy = _cast_stats_to_numpy_with_nonfinite_json_support


def _patch_policy_factory() -> None:
    try:
        import lerobot.policies as policies
        from lerobot.policies import factory
    except ModuleNotFoundError:
        return

    original_make_policy = factory.make_policy

    def _make_policy_with_alex_feature_filter(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if ds_meta is not None and _is_alex_test_obs_new_features(getattr(ds_meta, "features", {}) or {}):
            policy_type = str(getattr(cfg, "type", ""))
            filtered = _alex_filtered_policy_input_features(
                policy_type,
                factory.dataset_to_policy_features(ds_meta.features),
            )
            if filtered is not None:
                cfg.input_features = filtered
                logging.warning(
                    "Alex training image filtered %s input_features for H2Ozone/test_obs_new compatibility: %s",
                    policy_type,
                    ", ".join(filtered),
                )
        return original_make_policy(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

    factory.make_policy = _make_policy_with_alex_feature_filter
    policies.make_policy = _make_policy_with_alex_feature_filter

    # Defensive only: normal sitecustomize/.pth import order runs this before
    # lerobot_train imports make_policy, but patch an already-imported script too.
    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is not None:
        train_module.make_policy = _make_policy_with_alex_feature_filter


def _alex_video_timestamp_tolerance_s() -> float:
    raw = os.environ.get(ALEX_VIDEO_TIMESTAMP_TOLERANCE_ENV)
    if raw is None:
        return ALEX_VIDEO_TIMESTAMP_TOLERANCE_S
    try:
        return float(raw)
    except ValueError:
        logging.warning(
            "Ignoring invalid %s=%r; using %.4fs",
            ALEX_VIDEO_TIMESTAMP_TOLERANCE_ENV,
            raw,
            ALEX_VIDEO_TIMESTAMP_TOLERANCE_S,
        )
        return ALEX_VIDEO_TIMESTAMP_TOLERANCE_S


def _relaxed_video_tolerance(tolerance_s: float) -> float:
    return max(float(tolerance_s), _alex_video_timestamp_tolerance_s())


def _patch_video_timestamp_tolerance() -> None:
    try:
        from lerobot.datasets import video_utils
    except ModuleNotFoundError:
        return

    original_decode_video_frames = video_utils.decode_video_frames
    original_decode_video_frames_pyav = video_utils.decode_video_frames_pyav

    def _decode_video_frames_with_alex_tolerance(
        video_path,
        timestamps,
        tolerance_s,
        backend=None,
        return_uint8=False,
        is_depth=False,
    ):
        if backend in {None, "pyav", "video_reader"}:
            tolerance_s = _relaxed_video_tolerance(tolerance_s)
        return original_decode_video_frames(
            video_path,
            timestamps,
            tolerance_s,
            backend,
            return_uint8=return_uint8,
            is_depth=is_depth,
        )

    def _decode_video_frames_pyav_with_alex_tolerance(
        video_path,
        timestamps,
        tolerance_s,
        log_loaded_timestamps=False,
        return_uint8=False,
        is_depth=False,
    ):
        return original_decode_video_frames_pyav(
            video_path,
            timestamps,
            _relaxed_video_tolerance(tolerance_s),
            log_loaded_timestamps=log_loaded_timestamps,
            return_uint8=return_uint8,
            is_depth=is_depth,
        )

    video_utils.decode_video_frames = _decode_video_frames_with_alex_tolerance
    video_utils.decode_video_frames_pyav = _decode_video_frames_pyav_with_alex_tolerance

    # Defensive only: patch modules that imported the function before this hook.
    dataset_reader = sys.modules.get("lerobot.datasets.dataset_reader")
    if dataset_reader is not None:
        dataset_reader.decode_video_frames = _decode_video_frames_with_alex_tolerance


_patch_stats_loader()
_patch_policy_factory()
_patch_video_timestamp_tolerance()
