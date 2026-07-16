from pathlib import Path

import pytest

from lelab.alex_models import (
    CCILTrainingConfig,
    DatasetConversionConfig,
    EvaluationConfig,
    GR00TTrainingConfig,
    LeRobotTrainingConfig,
    RolloutConfig,
    TeleopConfig,
    build_arena_rollout_command,
    build_dataset_conversion_command,
    build_evaluation_command,
    build_isaaclab_rollout_command,
    build_isaaclab_teleop_command,
    build_lerobot_training_command,
)


def test_gr00t_conversion_command_uses_arena_script() -> None:
    config = DatasetConversionConfig(
        format="gr00t",
        input_path="/datasets/alex",
        output_path="/datasets/alex_gr00t",
        modality_template="/configs/modality.json",
    )
    command = build_dataset_conversion_command(config)
    assert command[1].endswith("convert_lerobot_v3_to_gr00t.py")
    assert command[-2:] == ["--modality_template", "/configs/modality.json"]


def test_gr00t_conversion_rejects_repo_only_source() -> None:
    with pytest.raises(ValueError, match="requires input_path"):
        DatasetConversionConfig(
            format="gr00t",
            repo_id="user/alex",
            output_path="/tmp/out",
            modality_template="/tmp/modality.json",
        )


def test_typed_training_configs_validate_limits() -> None:
    gr00t = GR00TTrainingConfig(dataset_id="user/alex", max_steps=10, use_lora=True)
    assert gr00t.kind == "gr00t"
    assert gr00t.lora_rank == 64
    ccil = CCILTrainingConfig(pickle_path="/datasets/alex.pkl", naive=True)
    assert ccil.kind == "ccil"
    with pytest.raises(ValueError):
        GR00TTrainingConfig(dataset_id="user/alex", max_steps=0)


def test_lerobot_training_config_is_hub_only_and_policy_safe() -> None:
    config = LeRobotTrainingConfig(
        dataset_repo_id="user/alex-data",
        model_repo_id="user/alex-act",
        policy_type="act",
    )
    assert config.kind == "lerobot"
    with pytest.raises(ValueError, match="owner/name"):
        LeRobotTrainingConfig(dataset_repo_id="/local/data", model_repo_id="user/model")
    with pytest.raises(ValueError, match="GR00T"):
        LeRobotTrainingConfig(
            dataset_repo_id="user/data",
            model_repo_id="user/model",
            policy_type="act",
            policy_chunk_size=16,
        )


def test_lerobot_multi_gpu_command_uses_accelerate_and_hub_checkpoints() -> None:
    config = LeRobotTrainingConfig(
        dataset_repo_id="user/alex-data",
        model_repo_id="user/alex-groot",
        policy_type="groot",
        policy_base_model_path="nvidia/GR00T-N1.7-3B",
        policy_chunk_size=16,
        policy_n_action_steps=8,
        policy_use_bf16=True,
    )
    command = build_lerobot_training_command(config, "/outputs/run", 2)
    assert command[:6] == [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        "2",
        "--mixed_precision",
    ]
    assert "--policy.type" in command
    assert command[command.index("--policy.type") + 1] == "groot"
    assert command[command.index("--policy.repo_id") + 1] == "user/alex-groot"
    assert command[command.index("--dataset.video_backend") + 1] == "pyav"
    assert command[command.index("--save_checkpoint_to_hub") + 1] == "true"


def test_test_obs_new_adds_policy_dimension_overrides() -> None:
    for policy_type in ["eo1", "evo1", "pi0", "pi05", "pi0_fast", "smolvla", "wall_x", "xvla"]:
        command = build_lerobot_training_command(
            LeRobotTrainingConfig(
                dataset_repo_id="H2Ozone/test_obs_new",
                model_repo_id=f"user/alex-{policy_type}",
                policy_type=policy_type,
            ),
            "/outputs/run",
            1,
        )
        assert command[command.index("--policy.max_state_dim") + 1] == "48"
        assert command[command.index("--policy.max_action_dim") + 1] == "46"

    fastwam = build_lerobot_training_command(
        LeRobotTrainingConfig(
            dataset_repo_id="H2Ozone/test_obs_new",
            model_repo_id="user/alex-fastwam",
            policy_type="fastwam",
        ),
        "/outputs/run",
        1,
    )
    assert fastwam[fastwam.index("--policy.action_dim") + 1] == "46"
    assert fastwam[fastwam.index("--policy.proprio_dim") + 1] == "48"

    lingbot = build_lerobot_training_command(
        LeRobotTrainingConfig(
            dataset_repo_id="H2Ozone/test_obs_new",
            model_repo_id="user/alex-lingbot",
            policy_type="lingbot_va",
        ),
        "/outputs/run",
        1,
    )
    assert lingbot[lingbot.index("--policy.action_dim") + 1] == "46"
    assert lingbot[lingbot.index("--policy.used_action_channel_ids") + 1].endswith("45]")


def test_isaaclab_rollout_command_uses_direct_launcher() -> None:
    config = RolloutConfig(
        target="sim",
        policy_ref="owner/model",
        dataset_repo_id="owner/data",
        environment="Isaac-Alex-Lever-Play-v0",
        isaaclab_root="/opt/IsaacLab",
        task="Stand ready",
    )
    command = build_isaaclab_rollout_command(
        config, "http://127.0.0.1:5000", {"version": 1}, "/tmp/metrics.json"
    )
    assert command[:2] == ["/opt/IsaacLab/isaaclab.sh", "-p"]
    assert Path(command[2]).name == "isaaclab_rollout_runner.py"
    assert "--environment" in command
    assert command[command.index("--environment") + 1] == "Isaac-Alex-Lever-Play-v0"
    assert "--remote_url" in command
    assert command[command.index("--remote_url") + 1] == "http://127.0.0.1:5000"
    assert "--metrics_output" in command
    assert "isaaclab_arena/evaluation/policy_runner.py" not in " ".join(command)


def test_arena_rollout_command_uses_docker_policy_runner() -> None:
    config = RolloutConfig(
        target="arena",
        policy_ref="owner/model",
        dataset_repo_id="owner/data",
        task="Turn the lever",
    )
    command = build_arena_rollout_command(
        config, "http://127.0.0.1:5000", {"version": 1, "policy_type": "fastwam"}, None
    )
    assert command[:3] == ["docker", "exec", "isaaclab_arena-latest"]
    assert Path(command[4]).name == "policy_runner.py"
    assert command[command.index("--policy_type") + 1].endswith("LeRobotRemotePolicy")
    assert command[command.index("--remote_url") + 1] == "http://127.0.0.1:5000"
    assert "--rollout_manifest" in command
    environment_index = command.index("alex_empty")
    assert command[environment_index + 1 :] == [
        "--embodiment",
        "alex_v2_ability_hands",
        "--usd",
        "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
    ]
    assert command.index("--policy_type") < environment_index


def test_isaaclab_teleop_command_uses_direct_launcher() -> None:
    config = TeleopConfig(isaaclab_root="/opt/IsaacLab")
    command = build_isaaclab_teleop_command(config)
    assert command[:2] == ["/opt/IsaacLab/isaaclab.sh", "-p"]
    assert Path(command[2]).name == "teleop_se3_agent.py"
    assert command[command.index("--task") + 1] == "Isaac-Alex-Lever-Play-v0"
    assert command[command.index("--teleop_device") + 1] == "keyboard"
    assert "--headless" not in command


def test_teleop_command_respects_device_and_sensitivity() -> None:
    config = TeleopConfig(
        teleop_device="spacemouse", sensitivity=2.5, num_envs=1, isaaclab_root="/opt/IsaacLab"
    )
    command = build_isaaclab_teleop_command(config)
    assert command[command.index("--teleop_device") + 1] == "spacemouse"
    assert command[command.index("--sensitivity") + 1] == "2.5"


def test_evaluation_argument_order_places_subparser_last() -> None:
    config = EvaluationConfig(policy_type="zero_action", model_path="/tmp/policy.pt")
    command = build_evaluation_command(config)
    environment_index = command.index("alex_open_microwave")
    assert command[environment_index + 1 :] == ["--embodiment", "alex_v2_ability_hands"]
    assert command.index("--policy_type") < environment_index
    assert command[:3] == ["docker", "exec", "isaaclab_arena-latest"]
    assert Path(command[4]).name == "policy_runner.py"
