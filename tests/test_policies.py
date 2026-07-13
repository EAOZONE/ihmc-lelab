from unittest.mock import MagicMock


def test_capabilities_mark_dataset_incompatibility(monkeypatch) -> None:
    from lelab import policies

    policies._cache.clear()
    monkeypatch.setattr(
        policies,
        "_remote_probe",
        lambda cluster, image: {
            "lerobot_version": "0.6.0",
            "policies": [
                {"type": "act", "available": True, "unavailable_reason": None},
                {"type": "groot", "available": True, "unavailable_reason": None},
            ],
        },
    )
    cluster = MagicMock()
    cluster.status.return_value = {"connected": True}
    result = policies.get_training_capabilities(
        dataset={
            "repo_id": "user/alex",
            "revision": "abc",
            "valid": True,
            "features": ["observation.state", "action"],
            "cameras": [],
            "has_tasks": True,
        },
        cluster=cluster,
    )
    by_type = {item["type"]: item for item in result["policies"]}
    assert by_type["act"]["compatible"] is True
    assert by_type["groot"]["compatible"] is False
    assert "camera" in by_type["groot"]["compatibility_reason"]


def test_capabilities_preserve_missing_extra_reason(monkeypatch) -> None:
    from lelab import policies

    policies._cache.clear()
    monkeypatch.setattr(
        policies,
        "_remote_probe",
        lambda cluster, image: {
            "lerobot_version": "0.6.0",
            "policies": [
                {"type": "smolvla", "available": False, "unavailable_reason": "missing transformers"},
            ],
        },
    )
    cluster = MagicMock()
    cluster.status.return_value = {"connected": True}
    result = policies.get_training_capabilities(cluster=cluster)
    assert result["policies"][0]["available"] is False
    assert result["policies"][0]["unavailable_reason"] == "missing transformers"


def test_capabilities_report_non_finite_stats_repair(monkeypatch) -> None:
    from lelab import policies

    policies._cache.clear()
    monkeypatch.setattr(
        policies,
        "_remote_probe",
        lambda cluster, image: {
            "lerobot_version": "0.6.0",
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "cuda_device_count": 3,
            "stats_sanitizer": True,
            "policies": [{"type": "groot", "available": True, "unavailable_reason": None}],
        },
    )
    cluster = MagicMock()
    cluster.status.return_value = {"connected": True}
    result = policies.get_training_capabilities(
        dataset={
            "repo_id": "user/alex",
            "valid": True,
            "features": ["observation.state", "action", "observation.images.left"],
            "cameras": ["observation.images.left"],
            "has_tasks": True,
            "non_finite_stats": [
                {"feature": "observation.state", "stat": "std", "count": 5}
            ],
            "groot_relative_actions_ready": False,
            "groot_relative_actions_reason": "grouped action names",
        },
        cluster=cluster,
    )
    assert result["torch_cuda_version"] == "12.8"
    assert "repair them in memory" in result["dataset_warnings"][0]
    assert result["groot_relative_actions_ready"] is False
    assert "grouped action names" in result["dataset_warnings"][1]
    groot = result["policies"][0]
    relative_field = next(field for field in groot["fields"] if field["name"] == "policy_use_relative_actions")
    assert relative_field["default"] is False


def test_remote_probe_checks_real_gpu_access() -> None:
    import pytest

    from lelab.policies import _remote_probe

    cluster = MagicMock()
    cluster.execute.return_value = (
        0,
        'ALEX_CAPABILITIES={"lerobot_version":"0.6.0","torch_version":"2.11.0+cu130",'
        '"torch_cuda_version":"13.0","cuda_available":false,"policies":[]}\n',
        "",
    )
    with pytest.raises(RuntimeError, match="cannot initialize a gpu2 GPU"):
        _remote_probe(cluster, "alex-lerobot-train:0.6.0")
    command = cluster.execute.call_args.args[0]
    assert "--gpus all" in command


def test_capabilities_require_cluster_connection() -> None:
    import pytest

    from lelab.policies import get_training_capabilities

    cluster = MagicMock()
    cluster.status.return_value = {"connected": False}
    with pytest.raises(ConnectionError, match="not connected"):
        get_training_capabilities(cluster=cluster)


def test_remote_dataset_probe_uses_remote_hf_token_without_exposing_it() -> None:
    from lelab.policies import _remote_dataset_probe

    cluster = MagicMock()
    cluster.execute.return_value = (
        0,
        'ALEX_DATASET={"repo_id":"user/private","valid":true,"features":["action"],"cameras":[],"has_tasks":true}\n',
        "",
    )
    result = _remote_dataset_probe(cluster, "alex-lerobot-train:0.6.0", "user/private")
    assert result["repo_id"] == "user/private"
    command = cluster.execute.call_args.args[0]
    assert "auth token" in command
    assert "export HF_TOKEN" in command
    assert "--env HF_TOKEN" in command
    assert "python3" in command
    assert "secret" not in command
