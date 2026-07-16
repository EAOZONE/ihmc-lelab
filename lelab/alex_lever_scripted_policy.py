"""Scripted Alex lever-push policy in the H2Ozone/test_obs_new action layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 48
ACTION_DIM = 46

LEFT_POSE = slice(0, 7)
RIGHT_POSE = slice(7, 14)
LEFT_FOREARM_QUAT = slice(14, 18)
RIGHT_FOREARM_QUAT = slice(18, 22)
HEAD_QUAT = slice(22, 26)
HAND_JOINTS = slice(26, 46)
RIGHT_HAND_JOINTS_IN_HAND_BLOCK = slice(10, 20)

DEFAULT_LEFT_FOREARM_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_RIGHT_FOREARM_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_HEAD_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

OPEN_HAND = np.array(
    [
        0.0,
        0.77,
        0.0,
        0.77,
        0.0,
        0.77,
        0.0,
        0.77,
        -1.73,
        0.0,
        0.0,
        0.77,
        0.0,
        0.77,
        0.0,
        0.77,
        0.0,
        0.77,
        -1.73,
        0.0,
    ],
    dtype=np.float32,
)
CLOSED_HAND = np.array(
    [
        0.7,
        1.35,
        0.7,
        1.35,
        0.7,
        1.35,
        0.55,
        1.15,
        -0.65,
        0.04,
        1.25,
        2.0,
        1.25,
        2.0,
        1.25,
        2.0,
        1.2,
        2.0,
        -0.35,
        0.04,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class LeverScriptConfig:
    """Timing and target offsets for the scripted lever interaction."""

    episode_steps: int = 180
    hover_height: float = 0.28
    push_depth: float = 0.34
    hand_x_offset: float = 0.0
    hand_y_offset: float = 0.0
    approach_fraction: float = 0.35
    close_fraction: float = 0.20
    push_fraction: float = 0.30


@dataclass(frozen=True)
class NeckLookConfig:
    """Gains/limits mapping a look-at target to NECK_Z (yaw) / NECK_Y (pitch) targets.

    Sign conventions are derived from the Alex URDF: NECK_Z rotates about the
    torso's +Z (world-up) axis with +Y being the robot's left side, so yaw =
    atan2(dy, dx) already turns negative (rightward) when the target is on the
    robot's right, matching the right hand / lever workspace. NECK_Y rotates
    about +Y, which points the local +X (forward) axis toward -Z for positive
    angles, so positive pitch already means "look down".
    """

    pitch_down_bias: float = 0.35
    yaw_gain: float = 1.0
    pitch_gain: float = 1.0
    yaw_limit: tuple[float, float] = (-1.4, 1.4)
    pitch_limit: tuple[float, float] = (-0.45, 0.45)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Return a normalized xyzw quaternion, falling back to identity."""

    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def state_quat_or_default(state: np.ndarray, key: slice, default: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(state[key])
    if not np.isfinite(quat).all():
        return default.copy()
    return quat


def smoothstep(value: float) -> float:
    """Cubic interpolation with zero slope at both ends."""

    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation for numpy vectors."""

    return (1.0 - alpha) * start + alpha * end


def phase_alphas(step: int, cfg: LeverScriptConfig) -> tuple[float, float, float]:
    """Return approach, close, and push blend amounts for a step."""

    denom = max(cfg.episode_steps - 1, 1)
    progress = float(np.clip(step / denom, 0.0, 1.0))
    approach_end = cfg.approach_fraction
    close_end = approach_end + cfg.close_fraction
    push_end = close_end + cfg.push_fraction
    approach = smoothstep(progress / max(approach_end, 1.0e-6))
    close = smoothstep((progress - approach_end) / max(cfg.close_fraction, 1.0e-6))
    push = smoothstep((progress - close_end) / max(cfg.push_fraction, 1.0e-6))
    if progress >= push_end:
        push = 1.0
    return approach, close, push


def right_hand_target_position(
    state: np.ndarray,
    lever_position: np.ndarray,
    step: int,
    cfg: LeverScriptConfig = LeverScriptConfig(),
    *,
    initial_right_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 3-D right-hand target position for the given step.

    Shared by :func:`build_lever_push_action` and by head/neck look-at logic so
    the camera can track the same point the hand is reaching for.
    """

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"expected state shape ({STATE_DIM},), got {state.shape}")
    lever_position = np.asarray(lever_position, dtype=np.float32).reshape(3)

    start_pose = state[RIGHT_POSE] if initial_right_pose is None else np.asarray(initial_right_pose, dtype=np.float32)
    start_pos = start_pose.reshape(7)[:3]

    above_pos = lever_position + np.array([cfg.hand_x_offset, cfg.hand_y_offset, cfg.hover_height], dtype=np.float32)
    push_pos = lever_position + np.array([cfg.hand_x_offset, cfg.hand_y_offset, cfg.hover_height - cfg.push_depth], dtype=np.float32)
    approach, _close, push = phase_alphas(step, cfg)
    target_pos = lerp(start_pos, above_pos, approach)
    target_pos = lerp(target_pos, push_pos, push)
    return target_pos.astype(np.float32)


def compute_neck_joint_targets(
    head_position: np.ndarray,
    look_at_position: np.ndarray,
    cfg: NeckLookConfig = NeckLookConfig(),
) -> tuple[float, float]:
    """Return (neck_z, neck_y) joint targets that aim the head at ``look_at_position``.

    Both positions must be in the same frame (env-origin), matching the frame
    used for gripper/lever positions elsewhere in this module.
    """

    head_position = np.asarray(head_position, dtype=np.float32).reshape(3)
    look_at_position = np.asarray(look_at_position, dtype=np.float32).reshape(3)
    delta = look_at_position - head_position

    horizontal_dist = float(np.hypot(delta[0], delta[1]))
    yaw = cfg.yaw_gain * float(np.arctan2(delta[1], delta[0]))
    pitch = cfg.pitch_gain * float(np.arctan2(-delta[2], max(horizontal_dist, 1.0e-6))) + cfg.pitch_down_bias

    yaw = float(np.clip(yaw, cfg.yaw_limit[0], cfg.yaw_limit[1]))
    pitch = float(np.clip(pitch, cfg.pitch_limit[0], cfg.pitch_limit[1]))
    return yaw, pitch


def build_lever_push_action(
    state: np.ndarray,
    lever_position: np.ndarray,
    step: int,
    cfg: LeverScriptConfig = LeverScriptConfig(),
    *,
    initial_right_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 46-D action that reaches above the lever, closes, then pushes down.

    Args:
        state: Current 48-D H2Ozone/test_obs_new state vector.
        lever_position: Lever position in the same env-origin frame as the gripper positions.
        step: Current episode frame index.
        cfg: Script timing and target offset configuration.
        initial_right_pose: Optional fixed 7-D starting right gripper pose. Passing this
            prevents the approach target from chasing the current controller state.
    """

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"expected state shape ({STATE_DIM},), got {state.shape}")
    lever_position = np.asarray(lever_position, dtype=np.float32).reshape(3)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[LEFT_POSE] = state[LEFT_POSE]
    action[LEFT_POSE.stop - 4 : LEFT_POSE.stop] = normalize_quat_xyzw(action[LEFT_POSE.stop - 4 : LEFT_POSE.stop])

    start_pose = state[RIGHT_POSE] if initial_right_pose is None else np.asarray(initial_right_pose, dtype=np.float32)
    right_quat = normalize_quat_xyzw(start_pose.reshape(7)[3:7])

    target_pos = right_hand_target_position(
        state, lever_position, step, cfg, initial_right_pose=initial_right_pose
    )
    _approach, close, _push = phase_alphas(step, cfg)

    action[RIGHT_POSE] = np.concatenate((target_pos.astype(np.float32), right_quat))
    action[LEFT_FOREARM_QUAT] = state_quat_or_default(state, LEFT_FOREARM_QUAT, DEFAULT_LEFT_FOREARM_QUAT)
    action[RIGHT_FOREARM_QUAT] = state_quat_or_default(state, RIGHT_FOREARM_QUAT, DEFAULT_RIGHT_FOREARM_QUAT)
    action[HEAD_QUAT] = state_quat_or_default(state, HEAD_QUAT, DEFAULT_HEAD_QUAT)

    hand = lerp(OPEN_HAND, CLOSED_HAND, close).astype(np.float32)
    hand[: RIGHT_HAND_JOINTS_IN_HAND_BLOCK.start] = OPEN_HAND[: RIGHT_HAND_JOINTS_IN_HAND_BLOCK.start]
    action[HAND_JOINTS] = hand
    return action
