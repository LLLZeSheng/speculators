import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANAGER = REPO_ROOT / "examples/train/manage_mtp_glm52_ascend_online_4v4t.sh"
WRAPPER = REPO_ROOT / "examples/train/run_mtp_glm52_ascend_online_container.sh"
LAUNCHER = REPO_ROOT / "examples/train/mtp_glm52_ascend_online.sh"
TEMPLATE = REPO_ROOT / "examples/train/mtp_glm52_ascend_online_4v4t.example.yaml"


def _write_user_yaml(path: Path) -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "FILL_VERIFIER_0_IP": "10.0.0.1",
        "FILL_VERIFIER_1_IP": "10.0.0.2",
        "FILL_VERIFIER_2_IP": "10.0.0.3",
        "FILL_VERIFIER_3_IP": "10.0.0.4",
        "FILL_TRAINER_0_IP": "10.0.1.1",
        "FILL_TRAINER_1_IP": "10.0.1.2",
        "FILL_TRAINER_2_IP": "10.0.1.3",
        "FILL_TRAINER_3_IP": "10.0.1.4",
        "FILL_UNIQUE_SMOKE_ID": "unit-test",
        "glm52-w4a8-mg13-speculator-training": "test-mtp",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def test_hidden_state_verifier_does_not_enable_pd_mixed_balancing():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '"enable_balance_scheduling":true' not in text
    assert '"enable_dsa_cp":false' in text
    assert '"enable_flashcomm1":false' in text
    assert '"enable_sparse_li_c8":true' in text
    assert "SPECULATORS_LOCAL_WEIGHT_LOAD_LOCK" in text


def test_verifier_maps_local_ip_to_vllm_host_ip(tmp_path: Path):
    environment = {
        **os.environ,
        "ROLE": "verifier",
        "VERIFIER_ID": "2",
        "DRY_RUN": "1",
        "LOCAL_IP": "7.150.9.19",
        "VERIFIER_QUANTIZATION_MODE": "ascend",
        "MTP_INIT_MODEL_PATH": "/model",
        "VERIFIER_MODEL_PATH": "/model",
        "DATA_PATH": "/data",
        "HIDDEN_STATES_PATH": "/hidden",
        "VERIFIER_METADATA_PATH": str(tmp_path / "metadata"),
        "LOG_ROOT": str(tmp_path / "logs"),
        # Simulate a stale value inherited from a reused container.
        "VLLM_HOST_IP": "7.150.8.73",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "VLLM_HOST_IP=7.150.9.19" in result.stdout


def test_manager_validates_user_yaml_and_dry_runs_topology(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)

    text = config.read_text(encoding="utf-8")
    assert "container_name_prefix: test-mtp" in text
    assert "nic_name: auto" in text
    assert (
        "mtp_init_model_path: "
        "/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4"
        in text
    )
    assert (
        "data_path: /kos_ulan/lzs/spec_train/dataset/hf/"
        "nuoya-average2k8k-32k" in text
    )
    assert "verifier_max_model_len: 32769" in text
    assert "verifier_max_batched_tokens: 32784" in text
    assert "total_seq_len: 32768" in text
    assert "request_timeout: 900" in text
    assert "container_mode: create" in text
    assert "- /mnt/xds/sfs:/mnt/xds/sfs" in text
    assert "- /root/.cache:/root/.cache" in text
    assert "install_speculators_verifier: false" in text
    assert "install_speculators_trainer: false" in text

    result = subprocess.run(
        ["bash", str(MANAGER), "validate-config", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CONFIG_STATUS=valid" in result.stdout
    assert config.read_text(encoding="utf-8") == text

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


def test_manager_reuses_existing_container_without_renaming_it(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
    text = config.read_text(encoding="utf-8").replace(
        "container_mode: create",
        "container_mode: existing\nexisting_container_name: existing-test-container",
    )
    config.write_text(text, encoding="utf-8")
    environment = {**os.environ, "MANAGER_DRY_RUN": "1"}

    result = subprocess.run(
        ["bash", str(MANAGER), "start-verifiers", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.count(
        "CONTAINER_NAME=existing-test-container"
    ) == 4
    assert "test-mtp-verifier0.host.log" in result.stdout


def test_manager_stop_restarts_each_existing_container_once(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
    text = config.read_text(encoding="utf-8").replace(
        "container_mode: create",
        "container_mode: existing\nexisting_container_name: existing-test-container",
    )
    config.write_text(text, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(MANAGER), "stop", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MANAGER_DRY_RUN": "1"},
    )

    assert result.stdout.count("[restart]") == 8
    assert result.stdout.count("docker\\ restart") == 8


def test_manager_can_restart_existing_roles_independently(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
    text = config.read_text(encoding="utf-8").replace(
        "container_mode: create",
        "container_mode: existing\nexisting_container_name: existing-test-container",
    )
    config.write_text(text, encoding="utf-8")
    environment = {**os.environ, "MANAGER_DRY_RUN": "1"}

    verifiers = subprocess.run(
        ["bash", str(MANAGER), "restart-verifiers", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert verifiers.stdout.count("[stop]") == 4
    assert verifiers.stdout.count("[restart]") == 4
    assert "host=10.0.0.1" in verifiers.stdout
    assert "host=10.0.1.1" not in verifiers.stdout

    trainers = subprocess.run(
        ["bash", str(MANAGER), "restart-trainers", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert trainers.stdout.count("[stop]") == 8
    assert trainers.stdout.count("[restart]") == 4
    assert "host=10.0.1.1" in trainers.stdout
    assert "host=10.0.0.1" not in trainers.stdout
    assert "180" not in trainers.stdout
    assert "15" in trainers.stdout


def test_manager_rejects_duplicate_node_addresses(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
    text = config.read_text(encoding="utf-8").replace("10.0.1.1", "10.0.0.1")
    config.write_text(text, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(MANAGER), "validate-config", "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "addresses must be unique" in result.stderr


def test_manager_rejects_unedited_yaml_template():
    result = subprocess.run(
        ["bash", str(MANAGER), "validate-config", "--config", str(TEMPLATE)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "FILL_ placeholder" in result.stderr


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
            "MTP_INIT_MODEL_PATH": (
                "/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/"
                "v1-ascend-modelslim-v4"
            ),
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
    assert "--net host" in result.stdout
    assert "--shm-size 1g" in result.stdout


def test_yaml_wrapper_skips_install_for_all_default_roles(
    tmp_path: Path,
):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
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
    assert "INSTALL_SPECULATORS=0" in trainer.stdout
    assert "run_mtp_glm52_ascend_online_job.sh" in trainer.stdout


def test_yaml_wrapper_uses_docker_exec_for_existing_container(tmp_path: Path):
    config = tmp_path / "cluster.yaml"
    _write_user_yaml(config)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "container_mode: create",
            "container_mode: existing\n"
            "existing_container_name: existing-test-container",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(WRAPPER), str(config)],
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NIC_NAME": "eth0",
            "NODE_IP": "10.0.0.1",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[container-existing] docker exec" in result.stdout
    assert "existing-test-container" in result.stdout
    assert "docker run" not in result.stdout
