#!/usr/bin/env python3
"""Direct Isaac Lab rollout runner for LeLab remote policies."""

from __future__ import annotations

import argparse
import io
import json
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    # Import here so helper-unit tests can import this module without Isaac Sim.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--remote_url", required=True)
    parser.add_argument("--rollout_manifest", required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--language_instruction", default="")
    parser.add_argument("--embodiment", default="")
    parser.add_argument("--metrics_output")
    parser.add_argument("--video_dir")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--camera_video", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _post(url: str, body: bytes, content_type: str) -> bytes:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - loopback tunnel
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"policy server {exc.code} for {url}: {detail}") from exc


def _as_numpy(value: Any) -> np.ndarray | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, (int, float, bool)) for item in value):
        return np.asarray(value)
    return None


def _flag_any(value: Any) -> bool:
    array = _as_numpy(value)
    if array is None:
        return bool(value)
    return bool(np.asarray(array).any())


def _reward_mean(value: Any) -> float:
    array = _as_numpy(value)
    if array is None:
        return float(value)
    return float(np.asarray(array).mean())


def _first_env(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array is None:
        raise TypeError(f"expected tensor/ndarray observation, got {type(value)!r}")
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _camera_key_map(manifest: dict[str, Any]) -> dict[str, str]:
    """Map Isaac Lab image term names -> LeRobot observation.images.* keys."""
    prefix = str(manifest.get("camera_prefix") or "observation.images.")
    schema = manifest.get("policy_schema") or {}
    inputs = schema.get("input_features") or {}
    visual = [
        name
        for name, spec in inputs.items()
        if str((spec or {}).get("type", "")).upper() == "VISUAL" or name.startswith(prefix)
    ]
    if visual:
        return {name.split(".")[-1]: name for name in visual}
    return {
        "cam_zed_left": f"{prefix}cam_zed_left",
        "cam_zed_right": f"{prefix}cam_zed_right",
    }


def observation_to_policy_features(obs: Any, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert Isaac Lab ManagerBasedRLEnv observations into LeRobot feature names.

    Lever Play returns ``{"policy": state(48,), "images": {"cam_zed_left": ..., ...}}``.
    An older flatten that recursed only into ``policy`` dropped the cameras and caused
    FastWAM ``/predict`` to 400 on missing image keys.
    """
    if isinstance(obs, tuple) and obs:
        obs = obs[0]

    features: dict[str, np.ndarray] = {}
    if isinstance(obs, dict) and ("policy" in obs or "images" in obs):
        policy = obs.get("policy", obs)
        if isinstance(policy, dict):
            state_value = policy.get("state", next(iter(policy.values()), None))
        else:
            state_value = policy
        if state_value is not None:
            features["observation.state"] = np.asarray(_first_env(state_value), dtype=np.float32).reshape(-1)

        images = obs.get("images", {})
        if isinstance(images, dict):
            for term_name, dataset_key in _camera_key_map(manifest).items():
                if term_name not in images:
                    continue
                frame = np.asarray(_first_env(images[term_name]))
                if frame.dtype != np.uint8:
                    if np.nanmax(frame) <= 1.0:
                        frame = frame * 255.0
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                if frame.ndim == 3 and frame.shape[-1] > 3:
                    frame = frame[..., :3]
                features[dataset_key] = frame
    else:
        array = _as_numpy(obs)
        if array is not None:
            features["observation.state"] = np.asarray(_first_env(array), dtype=np.float32).reshape(-1)

    if "observation.state" not in features:
        raise RuntimeError("Isaac Lab environment did not expose a numeric observation.state-compatible feature")
    return features


def _policy_payload(features: dict[str, np.ndarray], manifest: dict[str, Any], task: str) -> bytes:
    meta = {
        "task": task,
        "robot_type": manifest.get("target_profile", "alex"),
        "features": {},
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, array) in enumerate(features.items()):
            filename = f"feature_{index}.npy"
            meta["features"][name] = filename
            item = io.BytesIO()
            np.save(item, array, allow_pickle=False)
            archive.writestr(filename, item.getvalue())
        archive.writestr("meta.json", json.dumps(meta, separators=(",", ":")))
    return output.getvalue()


def _decode_action(body: bytes, device: Any | None = None) -> Any:
    actions = np.load(io.BytesIO(body), allow_pickle=False)
    if actions.ndim >= 3:
        actions = actions[:, 0]
    if actions.ndim == 1:
        actions = actions[None, :]
    try:
        import torch

        tensor = torch.from_numpy(actions.astype(np.float32, copy=False))
        return tensor.to(device) if device is not None else tensor
    except Exception:
        return actions

def _lever_angle(env: Any) -> float | None:
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    if scene is None:
        return None
    try:
        lever = scene["lever"]
        joint_pos = lever.data.joint_pos
        if joint_pos.numel() == 0:
            return None
        return float(joint_pos.detach().abs().max().cpu().item())
    except Exception:
        return None


def _write_failure(metrics_output: str | None, exc: BaseException) -> None:
    """Persist the failure before SimulationApp.close(), which may hard-exit 0."""
    text = traceback.format_exc()
    print(f"[isaaclab-rollout] failed: {exc}", flush=True)
    print(text, flush=True)
    if not metrics_output:
        return
    path = Path(metrics_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name("error.txt").write_text(text)


def main(args: argparse.Namespace, app_launcher: Any) -> None:
    manifest = json.loads(args.rollout_manifest)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Isaac Lab registers ManagerBasedRLEnv creators that require an explicit cfg=
    # constructed from env_cfg_entry_point. Bare gym.make(id) raises TypeError and
    # SimulationApp.close() can still hard-exit 0, which Lab then reports as done.
    device = f"cuda:{getattr(app_launcher, 'device_id', 0)}"
    env_cfg = parse_env_cfg(args.environment, device=device, num_envs=1)
    env = gym.make(args.environment, cfg=env_cfg)
    returns: list[float] = []
    lengths: list[int] = []
    final_lever_angles: list[float] = []
    max_lever_angle = 0.0
    started = time.time()

    for _episode in range(args.num_episodes):
        reset_result = env.reset()
        _post(f"{args.remote_url.rstrip('/')}/reset", b"{}", "application/json")
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            features = observation_to_policy_features(obs, manifest)
            payload = _policy_payload(features, manifest, args.language_instruction)
            action = _decode_action(
                _post(f"{args.remote_url.rstrip('/')}/predict", payload, "application/zip"),
                device=getattr(env.unwrapped, "device", None),
            )
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _info = step_result
                done = _flag_any(terminated) or _flag_any(truncated)
            else:
                obs, reward, done, _info = step_result
                done = _flag_any(done)
            total_reward += _reward_mean(reward)
            angle = _lever_angle(env)
            if angle is not None:
                max_lever_angle = max(max_lever_angle, angle)
            steps += 1
        returns.append(total_reward)
        lengths.append(steps)
        angle = _lever_angle(env)
        if angle is not None:
            final_lever_angles.append(angle)

    if args.metrics_output:
        metrics = {
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "elapsed_s": time.time() - started,
        }
        if final_lever_angles:
            metrics["final_lever_angle"] = final_lever_angles[-1]
            metrics["mean_final_lever_angle"] = float(np.mean(final_lever_angles))
            metrics["max_lever_angle"] = max_lever_angle
            metrics["lever_success"] = max_lever_angle >= 0.45
        path = Path(args.metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    cli_args = _parse_args()
    launcher = AppLauncher(vars(cli_args))
    simulation_app = launcher.app
    try:
        main(cli_args, launcher)
    except BaseException as exc:
        _write_failure(cli_args.metrics_output, exc)
        raise
    finally:
        simulation_app.close()
