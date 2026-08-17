import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANAGER = REPO_ROOT / "examples/train/manage_mtp_glm52_ascend_online_4v4t.sh"
WRAPPER = REPO_ROOT / "examples/train/run_mtp_glm52_ascend_online_container.sh"


def test_manager_configure_and_dry_run_topology(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    subprocess.run(
        [
            "bash",
            str(MANAGER),
            "configure",
            "--verifier-ips",
            "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            "--trainer-ips",
            "10.0.1.1,10.0.1.2,10.0.1.3,10.0.1.4",
            "--container-prefix",
            "test-mtp",
            "--config",
            str(config),
        ],
        check=True,
    )

    text = config.read_text(encoding="utf-8")
    assert 'container_name_prefix: "test-mtp"' in text
    assert "nic_name: auto" in text
    assert 'mtp_init_model_path: "/kos_ulan/models/GLM-5.2"' in text
    assert (
        'data_path: "/kos_ulan/lzs/spec_train/dataset/hf/'
        'nuoya-average2k8k-32k"' in text
    )
    assert "verifier_max_model_len: 32769" in text
    assert "verifier_max_batched_tokens: 32768" in text
    assert "total_seq_len: 32768" in text
    assert "request_timeout: 900" in text
    assert "install_speculators_verifier: false" in text
    assert "install_speculators_trainer: true" in text

    environment = os.environ.copy()
    environment["MANAGER_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", str(MANAGER), "train", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.count("[ssh]") == 4
    for rank in range(4):
        assert f"CONTAINER_NAME=test-mtp-trainer{rank}" in result.stdout
        assert f"NODE_RANK={rank}" in result.stdout
        assert f"VERIFIER_HOST=10.0.0.{rank + 1}" in result.stdout
    assert "MASTER_ADDR=10.0.1.1" in result.stdout

    result = subprocess.run(
        ["bash", str(MANAGER), "offline", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.count("TRAINER_DATA_MODE=offline") == 4


def test_manager_rejects_duplicate_node_addresses(tmp_path: Path):
    result = subprocess.run(
        [
            "bash",
            str(MANAGER),
            "configure",
            "--verifier-ips",
            "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            "--trainer-ips",
            "10.0.0.1,10.0.1.2,10.0.1.3,10.0.1.4",
            "--container-prefix",
            "test-mtp",
            "--config",
            str(tmp_path / "cluster.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "duplicate node address" in result.stderr


def test_wrapper_resolves_nic_from_local_ip(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ip = fake_bin / "ip"
    fake_ip.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '2: eth-test    inet 10.0.0.7/24 scope global eth-test'\n",
        encoding="utf-8",
    )
    fake_ip.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "DRY_RUN": "1",
            "ROLE": "preflight",
            "LOCAL_IP": "10.0.0.7",
            "NIC_NAME": "auto",
            "MTP_INIT_MODEL_PATH": "/kos_ulan/models/GLM-5.2",
            "DATA_PATH": "/kos_ulan/datasets/glm52-mtp-online",
        }
    )
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "[network] LOCAL_IP=10.0.0.7 NIC_NAME=eth-test" in result.stdout
    assert "NIC_NAME=eth-test" in result.stdout


def test_yaml_wrapper_skips_install_for_verifier_and_installs_for_trainer(
    tmp_path: Path,
):
    config = tmp_path / "cluster.yaml"
    subprocess.run(
        [
            "bash",
            str(MANAGER),
            "configure",
            "--verifier-ips",
            "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            "--trainer-ips",
            "10.0.1.1,10.0.1.2,10.0.1.3,10.0.1.4",
            "--container-prefix",
            "test-mtp",
            "--config",
            str(config),
        ],
        check=True,
    )
    base_environment = {
        **os.environ,
        "DRY_RUN": "1",
        "NIC_NAME": "eth0",
    }
    verifier = subprocess.run(
        ["bash", str(WRAPPER), str(config)],
        env={**base_environment, "NODE_IP": "10.0.0.1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pip install" not in verifier.stdout

    trainer = subprocess.run(
        ["bash", str(WRAPPER), str(config)],
        env={**base_environment, "NODE_IP": "10.0.1.1", "TRAINER_MODE": "trainer"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pip\\ install\\ --no-deps" in trainer.stdout
    assert "/hs_connectors" in trainer.stdout
