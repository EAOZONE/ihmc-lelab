import numpy as np

from lelab.alex_lever_scripted_policy import (
    CLOSED_HAND,
    OPEN_HAND,
    LeverScriptConfig,
    NeckLookConfig,
    build_lever_push_action,
    compute_neck_joint_targets,
)


def _state() -> np.ndarray:
    state = np.zeros(48, dtype=np.float32)
    state[3:7] = [0.0, 0.0, 0.0, 1.0]
    state[7:10] = [0.1, 0.2, 0.3]
    state[10:14] = [0.0, 0.0, 0.0, 1.0]
    state[14:18] = [0.0, 0.0, 0.0, 1.0]
    state[18:22] = [0.0, 0.0, 0.0, 1.0]
    state[22:26] = [0.0, 0.0, 0.0, 1.0]
    state[26:46] = OPEN_HAND
    return state


def test_lever_push_action_has_dataset_shape() -> None:
    action = build_lever_push_action(_state(), np.array([1.0, 2.0, 3.0], dtype=np.float32), 0)
    assert action.shape == (46,)
    assert action.dtype == np.float32


def test_lever_push_starts_at_initial_right_pose_and_open_hand() -> None:
    state = _state()
    action = build_lever_push_action(state, np.array([1.0, 2.0, 3.0], dtype=np.float32), 0)
    np.testing.assert_allclose(action[7:14], state[7:14])
    np.testing.assert_allclose(action[26:46], OPEN_HAND)


def test_lever_push_ends_below_lever_with_right_hand_closed() -> None:
    cfg = LeverScriptConfig(episode_steps=10, hover_height=0.28, push_depth=0.34)
    lever = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    action = build_lever_push_action(_state(), lever, 9, cfg)
    np.testing.assert_allclose(action[7:10], [1.0, 2.0, 2.94], atol=1e-6)
    np.testing.assert_allclose(action[26:36], OPEN_HAND[:10])
    np.testing.assert_allclose(action[36:46], CLOSED_HAND[10:20])


def test_neck_look_at_target_to_the_right_and_below_yields_negative_yaw_and_positive_pitch() -> None:
    head_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    look_at_position = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    yaw, pitch = compute_neck_joint_targets(head_position, look_at_position, NeckLookConfig(pitch_down_bias=0.0))
    assert yaw < 0.0
    assert pitch > 0.0


def test_neck_look_at_directly_ahead_and_level_still_applies_pitch_down_bias() -> None:
    head_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    look_at_position = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    yaw, pitch = compute_neck_joint_targets(head_position, look_at_position, NeckLookConfig(pitch_down_bias=0.3))
    np.testing.assert_allclose(yaw, 0.0, atol=1e-6)
    np.testing.assert_allclose(pitch, 0.3, atol=1e-6)


def test_neck_look_at_clips_to_configured_limits() -> None:
    head_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    look_at_position = np.array([0.0, 1.0, -5.0], dtype=np.float32)
    cfg = NeckLookConfig(pitch_down_bias=0.0, yaw_limit=(-1.0, 1.0), pitch_limit=(-0.4, 0.4))
    yaw, pitch = compute_neck_joint_targets(head_position, look_at_position, cfg)
    np.testing.assert_allclose(yaw, 1.0, atol=1e-6)
    np.testing.assert_allclose(pitch, 0.4, atol=1e-6)
