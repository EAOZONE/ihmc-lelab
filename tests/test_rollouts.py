from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lelab.alex_models import RolloutConfig
from lelab.rollouts import (
    RolloutManager,
    build_inference_container_command,
    build_local_inference_container_command,
    resolve_rollout_source,
)


def test_rollout_config_requires_exactly_one_source() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one"):
        RolloutConfig(target="arena")
    with pytest.raises(ValueError, match="exactly one"):
        RolloutConfig(job_id="job", policy_ref="owner/model")


def test_job_checkpoint_resolves_to_zero_padded_hub_ref() -> None:
    jobs = MagicMock()
    jobs.get.return_value.config = {
        "model_repo_id": "owner/alex-act",
        "dataset_repo_id": "owner/alex-data",
        "policy_type": "act",
    }
    ref, manifest = resolve_rollout_source(RolloutConfig(job_id="job-1", checkpoint="500"), jobs)
    assert ref == "owner/alex-act@checkpoints/000500"
    assert manifest["dataset_repo_id"] == "owner/alex-data"
    assert manifest["policy_type"] == "act"


def test_direct_hub_latest_resolves_to_latest_ref() -> None:
    jobs = MagicMock()
    ref, manifest = resolve_rollout_source(
        RolloutConfig(policy_ref="owner/alex-groot", checkpoint="latest", dataset_repo_id="owner/alex-data"),
        jobs,
    )
    assert ref == "owner/alex-groot@latest"
    assert manifest["dataset_repo_id"] == "owner/alex-data"


def test_arena_target_is_preserved() -> None:
    config = RolloutConfig(target="arena", policy_ref="owner/model", dataset_repo_id="owner/data")
    assert config.target == "arena"
    assert config.environment == "alex_empty"
    assert config.usd.endswith("LEVER_AGAIN.usd")


def test_remote_inference_command_is_loopback_only_and_has_no_token() -> None:
    command = build_inference_container_command("run-1", "owner/model@latest", "GPU-a", 24567)
    assert "--host 127.0.0.1" in command
    assert "--network host" in command
    assert "GPU-a" in command
    assert "owner/model@latest" in command
    assert "auth token" in command
    assert "hf_x" not in command


def test_local_inference_command_downloads_policy_on_host_network() -> None:
    command = build_local_inference_container_command("run-1", "owner/model@latest", "0", 24567)
    assert command[:5] == ["docker", "run", "--detach", "--name", "alex-rollout-run-1"]
    assert command[command.index("--gpus") + 1] == "device=0"
    assert "--network" in command
    assert "host" in command
    assert "owner/model@latest" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_local_inference_command_mounts_local_policy_path(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()

    command = build_local_inference_container_command("run-1", str(policy), "0", 24567)

    volume = command[command.index("--volume", command.index("alex_hf_cache:/cache/huggingface")) + 1]
    assert volume == f"{policy.resolve()}:/policy:ro"
    assert command[command.index("--policy") + 1] == "/policy"


def test_remote_inference_rejects_local_policy_path(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    manager = RolloutManager(tmp_path / "rollouts", cluster=MagicMock(), jobs=MagicMock())

    with pytest.raises(ValueError, match="local policy_ref paths require local inference"):
        manager.start(
            RolloutConfig(target="arena", policy_ref=str(policy), dataset_repo_id="owner/dataset")
        )


def test_physical_alex_rollout_returns_structured_safety_blockers(tmp_path: Path) -> None:
    manager = RolloutManager(tmp_path, cluster=MagicMock(), jobs=MagicMock())
    record = manager.start(
        RolloutConfig(target="robot", policy_ref="owner/model", dataset_repo_id="owner/dataset")
    )
    assert record.state == "blocked"
    assert len(record.blockers) == 4
    assert "state readback" in " ".join(record.blockers)
    assert (tmp_path / record.id / "rollout.json").is_file()


def test_sim_rollout_requires_isaaclab_launcher(tmp_path: Path) -> None:
    manager = RolloutManager(tmp_path, cluster=MagicMock(), jobs=MagicMock())
    with pytest.raises(FileNotFoundError, match="Isaac Lab launcher"):
        manager.start(
            RolloutConfig(
                target="sim",
                policy_ref="owner/model",
                dataset_repo_id="owner/dataset",
                environment="Isaac-Alex-Lever-Play-v0",
                isaaclab_root=str(tmp_path / "missing"),
            )
        )


def test_sim_rollout_launches_local_isaaclab_process(tmp_path: Path, monkeypatch) -> None:
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    launcher = isaaclab / "isaaclab.sh"
    launcher.write_text("#!/usr/bin/env bash\n")

    cluster = MagicMock()
    cluster.gpus.return_value = [{"index": 0, "uuid": "GPU-a"}]
    cluster.execute.return_value = (0, "container", "")
    tunnel = MagicMock(local_port=45678)
    cluster.forward_remote_port.return_value = tunnel
    jobs = MagicMock()
    manager = RolloutManager(tmp_path / "rollouts", cluster=cluster, jobs=jobs)
    monkeypatch.setattr(manager, "_wait_policy_server_ready", lambda _record, _port: {"protocol_version": 1})

    popen = MagicMock()
    popen.return_value.pid = 1234
    monkeypatch.setattr("lelab.rollouts.subprocess.Popen", popen)

    record = manager.start(
        RolloutConfig(
            target="sim",
            policy_ref="owner/model",
            dataset_repo_id="owner/dataset",
            environment="Isaac-Alex-Lever-Play-v0",
            isaaclab_root=str(isaaclab),
        )
    )

    assert record.state == "running"
    assert record.pid == 1234
    command = popen.call_args.args[0]
    assert command[:2] == [str(launcher), "-p"]
    assert Path(command[2]).name == "isaaclab_rollout_runner.py"
    assert command[command.index("--environment") + 1] == "Isaac-Alex-Lever-Play-v0"
    assert popen.call_args.kwargs["cwd"] == str(isaaclab.resolve())
    assert "CONDA_PREFIX" not in popen.call_args.kwargs["env"]


def test_arena_rollout_launches_docker_policy_runner(tmp_path: Path, monkeypatch) -> None:
    cluster = MagicMock()
    cluster.gpus.return_value = [{"index": 0, "uuid": "GPU-a"}]
    cluster.execute.return_value = (0, "container", "")
    tunnel = MagicMock(local_port=45678)
    cluster.forward_remote_port.return_value = tunnel
    manager = RolloutManager(tmp_path / "rollouts", cluster=cluster, jobs=MagicMock())
    monkeypatch.setattr(manager, "_wait_policy_server_ready", lambda _record, _port: {"protocol_version": 1})
    monkeypatch.setattr(manager, "_require_arena_container", lambda _name: None)

    popen = MagicMock()
    popen.return_value.pid = 4321
    monkeypatch.setattr("lelab.rollouts.subprocess.Popen", popen)

    record = manager.start(
        RolloutConfig(target="arena", policy_ref="owner/model", dataset_repo_id="owner/dataset")
    )

    assert record.state == "running"
    assert record.pid == 4321
    command = popen.call_args.args[0]
    assert command[:3] == ["docker", "exec", "isaaclab_arena-latest"]
    assert "alex_empty" in command
    assert command[command.index("--usd") + 1].endswith("LEVER_AGAIN.usd")
    assert command[command.index("--remote_url") + 1] == "http://127.0.0.1:45678"
    assert popen.call_args.kwargs["cwd"] is None


def test_arena_rollout_can_use_local_inference_container(tmp_path: Path, monkeypatch) -> None:
    cluster = MagicMock()
    manager = RolloutManager(tmp_path / "rollouts", cluster=cluster, jobs=MagicMock())
    monkeypatch.setattr(manager, "_wait_ready", lambda _port, record=None: {"protocol_version": 1})
    monkeypatch.setattr(manager, "_require_arena_container", lambda _name: None)

    run = MagicMock()
    run.return_value.returncode = 0
    run.return_value.stdout = "container"
    run.return_value.stderr = ""
    monkeypatch.setattr("lelab.rollouts.subprocess.run", run)

    popen = MagicMock()
    popen.return_value.pid = 4321
    monkeypatch.setattr("lelab.rollouts.subprocess.Popen", popen)

    record = manager.start(
        RolloutConfig(
            target="arena",
            inference_location="local",
            policy_ref="owner/model",
            dataset_repo_id="owner/dataset",
        )
    )

    assert record.state == "running"
    cluster.gpus.assert_not_called()
    server_command = run.call_args_list[0].args[0]
    assert server_command[:3] == ["docker", "run", "--detach"]
    assert server_command[server_command.index("--gpus") + 1] == "device=0"
    runner_command = popen.call_args.args[0]
    assert runner_command[runner_command.index("--remote_url") + 1].startswith("http://127.0.0.1:")


def test_exit_zero_without_metrics_is_failed_for_sim(tmp_path: Path) -> None:
    """SimulationApp.close() can hard-exit 0 after gym.make fails without cfg."""
    manager = RolloutManager(tmp_path, cluster=MagicMock(), jobs=MagicMock())
    record = manager.start(
        RolloutConfig(target="robot", policy_ref="owner/model", dataset_repo_id="owner/dataset")
    )
    record.state = "running"
    record.config = record.config.model_copy(update={"target": "sim"})
    record.pid = 1
    manager._persist(record)

    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    manager._processes[record.id] = proc

    updated = manager.get(record.id)
    assert updated.state == "failed"
    assert "metrics" in (updated.error_message or "")


def test_arena_exit_zero_without_metrics_is_done(tmp_path: Path) -> None:
    manager = RolloutManager(tmp_path, cluster=MagicMock(), jobs=MagicMock())
    record = manager.start(
        RolloutConfig(target="robot", policy_ref="owner/model", dataset_repo_id="owner/dataset")
    )
    record.state = "running"
    record.config = record.config.model_copy(update={"target": "arena"})
    record.pid = 1
    manager._persist(record)

    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    manager._processes[record.id] = proc

    updated = manager.get(record.id)
    assert updated.state == "done"


def test_reattach_removes_interrupted_inference_container(tmp_path: Path) -> None:
    cluster = MagicMock()
    cluster.status.return_value = {"connected": True}
    manager = RolloutManager(tmp_path, cluster=cluster, jobs=MagicMock())
    record = manager.start(
        RolloutConfig(target="robot", policy_ref="owner/model", dataset_repo_id="owner/dataset")
    )
    record.state = "interrupted"
    record.inference_container = "alex-rollout-old"

    manager.reattach()

    cluster.execute.assert_called_once_with("docker rm --force alex-rollout-old", timeout=20)


def test_sim_rollout_failure_includes_container_diagnostics(tmp_path: Path, monkeypatch) -> None:
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    (isaaclab / "isaaclab.sh").write_text("#!/usr/bin/env bash\n")

    cluster = MagicMock()
    cluster.gpus.return_value = [{"index": 0, "uuid": "GPU-a"}]
    cluster.status.return_value = {"connected": True}
    cluster.forward_remote_port.return_value = MagicMock(local_port=45678)

    def execute(command, timeout=30):
        if "docker run" in command:
            return 0, "container", ""
        if "docker ps" in command:
            return 0, "alex-rollout-x Exited (1) 2 seconds ago", ""
        if "docker logs" in command:
            return 0, "RuntimeError: policy load failed", ""
        return 0, "", ""

    cluster.execute.side_effect = execute
    manager = RolloutManager(tmp_path / "rollouts", cluster=cluster, jobs=MagicMock())
    monkeypatch.setattr(manager, "_wait_policy_server_ready", lambda _record, _port: (_ for _ in ()).throw(TimeoutError("not ready")))

    with pytest.raises(RuntimeError, match="policy load failed"):
        manager.start(
            RolloutConfig(
                target="sim",
                policy_ref="owner/model",
                dataset_repo_id="owner/dataset",
                environment="Isaac-Alex-Lever-Play-v0",
                isaaclab_root=str(isaaclab),
            )
        )

    record = next(iter(manager._records.values()))
    assert record.state == "failed"
    assert "container status" in record.error_message
    assert "policy load failed" in record.error_message
