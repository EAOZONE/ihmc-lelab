from unittest.mock import MagicMock

import paramiko
import pytest

from lelab.cluster import (
    ClusterConnectRequest,
    FingerprintPolicy,
    HostKeyVerificationError,
    parse_nvidia_smi_gpus,
    sha256_fingerprint,
)


def test_nvidia_smi_parser_attaches_processes_by_uuid() -> None:
    gpus = parse_nvidia_smi_gpus(
        "0, GPU-aaa, NVIDIA H100 80GB HBM3, 72, 1000, 81559, 48, 300.5, 700.0\n"
        "1, GPU-bbb, NVIDIA H100 80GB HBM3, 0, 3, 81559, 31, N/A, 700.0\n",
        "GPU-aaa, 1234, python, 900\n",
    )
    assert len(gpus) == 2
    assert gpus[0]["uuid"] == "GPU-aaa"
    assert gpus[0]["processes"][0]["pid"] == 1234
    assert gpus[1]["power_draw_w"] is None


def test_password_is_excluded_from_serialization() -> None:
    request = ClusterConnectRequest(host="gpu2", username="alex", password="secret")
    assert "password" not in request.model_dump()
    assert "secret" not in request.model_dump_json()


def test_fingerprint_policy_accepts_only_exact_sha256() -> None:
    key = paramiko.RSAKey.generate(1024)
    expected = sha256_fingerprint(key)
    policy = FingerprintPolicy(expected)
    policy.missing_host_key(MagicMock(), "gpu2", key)

    other = paramiko.RSAKey.generate(1024)
    with pytest.raises(HostKeyVerificationError, match="host key mismatch"):
        policy.missing_host_key(MagicMock(), "gpu2", other)


def test_ssh_tunnel_bridge_suppresses_remote_connect_refused(monkeypatch) -> None:
    from lelab.cluster import SshTunnel

    tunnel = object.__new__(SshTunnel)
    tunnel._remote = ("127.0.0.1", 24000)
    tunnel._stop = MagicMock()
    tunnel._stop.is_set.return_value = False
    tunnel._transport = MagicMock()
    tunnel._transport.open_channel.side_effect = paramiko.ChannelException(2, "Connect failed")
    client = MagicMock()

    tunnel._bridge(client, ("127.0.0.1", 12345))

    client.close.assert_called_once()
