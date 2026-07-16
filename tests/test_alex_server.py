from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from lelab.alex_server import app

client = TestClient(app)


def test_health_and_spa_are_served() -> None:
    assert client.get("/health").json() == {"status": "ok", "app": "alex-lab"}
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_cluster_status_has_frontend_user_alias(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server.cluster_manager,
        "status",
        MagicMock(
            return_value={
                "connected": True,
                "host": "gpu2",
                "username": "bpratt",
                "port": 22,
                "fingerprint": "SHA256:test",
            }
        ),
    )
    body = client.get("/alex/cluster/status").json()
    assert body["connected"] is True
    assert body["user"] == "bpratt"


def test_gpu_response_maps_telemetry_and_reservations(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server.cluster_manager,
        "gpus",
        MagicMock(
            return_value=[
                {
                    "index": 0,
                    "uuid": "GPU-a",
                    "name": "H100",
                    "utilization_percent": 75,
                    "memory_used_mb": 1000,
                    "memory_total_mb": 81559,
                    "temperature_c": 45,
                    "power_draw_w": 300.0,
                    "power_limit_w": 700.0,
                    "processes": [{"pid": 1, "name": "python", "memory_used_mb": 1000}],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        alex_server.remote_job_manager,
        "reservations",
        MagicMock(return_value={"GPU-a": "job-1"}),
    )
    gpu = client.get("/alex/cluster/gpus").json()[0]
    assert gpu["utilization"] == 75
    assert gpu["power_w"] == 300.0
    assert gpu["occupied"] is True
    assert gpu["reserved_by"] == "job-1"
    assert gpu["processes"][0]["memory_mb"] == 1000


def test_training_capabilities_use_remote_dataset_probe(monkeypatch) -> None:
    from lelab import alex_server

    capability_mock = MagicMock(
        return_value={"image": "alex-lerobot-train:0.6.0", "lerobot_version": "0.6.0", "policies": []}
    )
    monkeypatch.setattr(alex_server, "get_training_capabilities", capability_mock)
    response = client.get("/alex/training/capabilities?dataset_repo_id=user%2Falex")
    assert response.status_code == 200
    capability_mock.assert_called_once_with(dataset_repo_id="user/alex")


def test_training_rejects_legacy_ccil_launch() -> None:
    response = client.post(
        "/alex/training",
        json={"name": "old", "gpus": ["0"], "config": {"kind": "ccil", "pickle_path": "/x.pkl"}},
    )
    assert response.status_code == 422


def test_training_rejects_policy_missing_from_remote_image(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server,
        "get_training_capabilities",
        MagicMock(return_value={"policies": []}),
    )
    response = client.post(
        "/alex/training",
        json={
            "name": "run",
            "gpus": ["0"],
            "config": {
                "kind": "lerobot",
                "dataset_repo_id": "user/alex",
                "model_repo_id": "user/alex-act",
                "policy_type": "act",
            },
        },
    )
    assert response.status_code == 409
    assert "not provided" in response.json()["detail"]


def test_training_rejects_unsupported_groot_relative_actions(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server,
        "get_training_capabilities",
        MagicMock(
            return_value={
                "groot_relative_actions_ready": False,
                "groot_relative_actions_reason": "grouped action names",
                "policies": [
                    {
                        "type": "groot",
                        "available": True,
                        "compatible": True,
                        "unavailable_reason": None,
                        "compatibility_reason": None,
                    }
                ],
            }
        ),
    )
    response = client.post(
        "/alex/training",
        json={
            "name": "run",
            "gpus": ["0"],
            "config": {
                "kind": "lerobot",
                "dataset_repo_id": "user/alex",
                "model_repo_id": "user/alex-groot",
                "policy_type": "groot",
                "policy_use_relative_actions": True,
            },
        },
    )
    assert response.status_code == 409
    assert "Disable 'Use relative actions'" in response.json()["detail"]


def test_teleop_start_returns_running_session(monkeypatch) -> None:
    from lelab import alex_server
    from lelab.alex_teleop import TeleopRecord

    record = TeleopRecord(
        id="teleop-1",
        state="running",
        config={"environment": "Isaac-Alex-Lever-Play-v0"},
        started_at=0.0,
        pid=123,
        log_path="/tmp/teleop-1.log",
    )
    monkeypatch.setattr(alex_server.teleop_manager, "start", MagicMock(return_value=record))
    response = client.post("/alex/teleop", json={})
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "teleop-1"
    assert body["status"] == "running"
    assert body["pid"] == 123


def test_teleop_start_missing_launcher_returns_400(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server.teleop_manager,
        "start",
        MagicMock(side_effect=FileNotFoundError("Isaac Lab launcher not found: /missing/isaaclab.sh")),
    )
    response = client.post("/alex/teleop", json={})
    assert response.status_code == 400


def test_teleop_session_not_found_returns_404(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(alex_server.teleop_manager, "get", MagicMock(side_effect=KeyError("teleop-x")))
    assert client.get("/alex/teleop/teleop-x").status_code == 404
    monkeypatch.setattr(alex_server.teleop_manager, "logs", MagicMock(side_effect=KeyError("teleop-x")))
    assert client.get("/alex/teleop/teleop-x/logs").status_code == 404
    monkeypatch.setattr(alex_server.teleop_manager, "stop", MagicMock(side_effect=KeyError("teleop-x")))
    assert client.post("/alex/teleop/teleop-x/stop").status_code == 404


def test_teleop_stop_not_running_returns_409(monkeypatch) -> None:
    from lelab import alex_server

    monkeypatch.setattr(
        alex_server.teleop_manager,
        "stop",
        MagicMock(side_effect=RuntimeError("teleop session is not running")),
    )
    response = client.post("/alex/teleop/teleop-1/stop")
    assert response.status_code == 409


def test_training_metrics_history_route(monkeypatch) -> None:
    from lelab import alex_server
    from lelab.jobs import MetricsHistoryPoint

    monkeypatch.setattr(
        alex_server.remote_job_manager,
        "metrics_history",
        MagicMock(return_value=[MetricsHistoryPoint(step=100, loss=0.42, lr=1e-4, grad_norm=1.5)]),
    )

    response = client.get("/alex/jobs/job-1/metrics-history")

    assert response.status_code == 200
    assert response.json()["points"] == [{"step": 100, "loss": 0.42, "lr": 1e-4, "grad_norm": 1.5}]
