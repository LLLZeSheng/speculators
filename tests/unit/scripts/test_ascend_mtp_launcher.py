import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "train"
    / "mtp_glm52_ascend_online.sh"
)
CONTAINER_SCRIPT = SCRIPT.with_name("run_mtp_glm52_ascend_online_container.sh")


def _run(role: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ROLE": role,
        "DRY_RUN": "1",
        "VERIFIER_HOST": "10.0.0.10",
        "MASTER_ADDR": "10.0.0.20",
        "NODE_RANK": "0",
        "SMOKE_RUN_ID": "unit-test",
        **overrides,
    }
    return subprocess.run(  # noqa: S603
        [shutil.which("bash") or "/bin/bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_container(role: str = "", **overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ROLE": role,
        "DRY_RUN": "1",
        **overrides,
    }
    return subprocess.run(  # noqa: S603
        [shutil.which("bash") or "/bin/bash", str(CONTAINER_SCRIPT)],
        cwd=CONTAINER_SCRIPT.parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_container_wrapper_uses_requested_a3_image_and_all_devices():
    result = _run_container("verifier")

    assert result.returncode == 0, result.stderr
    assert "quay.io/ascend/vllm-ascend:v0.23.0rc1-a3" in result.stdout
    assert "/kos_ulan:/kos_ulan" in result.stdout
    assert "--device /dev/davinci0" in result.stdout
    assert "--device /dev/davinci15" in result.stdout
    assert "--ipc host" in result.stdout


def test_verifier_dry_run_builds_w4a8_hidden_state_service():
    result = _run("verifier")

    assert result.returncode == 0, result.stderr
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in (
        result.stdout
    )
    assert "v1-ascend-modelslim-v4" in result.stdout
    assert "scripts/launch_vllm.py" in result.stdout
    assert "--target-layer-ids 78" in result.stdout
    assert "--hidden-states-path" in result.stdout
    assert "/kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8" in result.stdout
    assert "--data-parallel-size 1" in result.stdout
    assert "--tensor-parallel-size 16" in result.stdout
    assert "--max-model-len 32769" in result.stdout
    assert "--max-num-batched-tokens 32768" in result.stdout
    assert "scripts/prepare_mixed_quant_model.py" not in result.stdout
    assert "--quantization ascend" in result.stdout
    assert "--enable-expert-parallel" in result.stdout
    assert "--required-devices 16" in result.stdout
    assert "--served-model-name glm52-w4a8c8-verifier" in result.stdout
    assert "/kos_ulan/spec_train/metadata/glm52-w4a8c8" in result.stdout


def test_verifier_can_still_select_compressed_tensors_normalization():
    result = _run(
        "verifier", VERIFIER_QUANTIZATION_MODE="compressed-tensors"
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_mixed_quant_model.py" in result.stdout
    assert "--quantization ascend" not in result.stdout


def test_trainer_dry_run_builds_four_node_bf16_mtp3_job():
    result = _run("trainer", NODE_RANK="3")

    assert result.returncode == 0, result.stderr
    assert "--nnodes 4" in result.stdout
    assert "--nproc-per-node 16" in result.stdout
    assert "--node-rank 3" in result.stdout
    assert "--master-addr 10.0.0.20" in result.stdout
    assert "--verifier-name-or-path /kos_ulan/models/GLM-5.2" in result.stdout
    assert "--generation-model-name-or-path glm52-w4a8c8-verifier" in result.stdout
    assert "--from-pretrained" in result.stdout
    assert "--speculator-type mtp" in result.stdout
    assert "--num-speculative-steps 3" in result.stdout
    assert "--hidden-states-dtype bfloat16" in result.stdout
    assert "--fsdp-shard" in result.stdout
    assert "--fsdp-skip-initial-broadcast" in result.stdout
    assert "--on-missing generate" in result.stdout


def test_trainer_accepts_two_verifier_endpoints_for_local_rank_fanout():
    result = _run(
        "trainer",
        NNODES="2",
        NODE_RANK="1",
        VERIFIER_HOSTS="10.0.0.11,10.0.0.13",
    )

    assert result.returncode == 0, result.stderr
    assert "--nnodes 2" in result.stdout
    assert "--node-rank 1" in result.stdout
    assert "--vllm-endpoint" in result.stdout
    assert "http://10.0.0.11:8077/v1" in result.stdout
    assert "http://10.0.0.13:8077/v1" in result.stdout
    assert "--on-generate cache" in result.stdout
    assert "--force-generate" not in result.stdout
    assert "--on-generation-error raise" in result.stdout
    assert "--checkpoint-steps 1000" in result.stdout
    assert "--total-seq-len 32768" in result.stdout
    assert "--request-timeout 900" in result.stdout
    assert "--max-retries 3" in result.stdout


def test_trainer_offline_mode_never_contacts_verifier():
    result = _run("trainer", TRAINER_DATA_MODE="offline")

    assert result.returncode == 0, result.stderr
    train_command = next(
        line for line in result.stdout.splitlines() if "scripts/train.py" in line
    )
    assert "--on-missing raise" in train_command
    assert "--vllm-endpoint" not in train_command
    assert "--on-generate" not in train_command
    assert "--force-generate" not in train_command


def test_verifier_preflight_uses_dp_times_tp_device_count():
    result = _run(
        "verifier", VERIFIER_TP_SIZE="8", VERIFIER_DP_SIZE="1", NPROC_PER_NODE="16"
    )

    assert result.returncode == 0, result.stderr
    assert "--required-devices 8" in result.stdout


def test_trainer_rejects_context_that_leaves_no_output_token():
    result = _run(
        "trainer",
        TOTAL_SEQ_LEN="8192",
        VERIFIER_MAX_MODEL_LEN="8192",
    )

    assert result.returncode != 0
    assert "VERIFIER_MAX_MODEL_LEN" in result.stderr


def test_trainer_accepts_8192_with_one_output_token_of_headroom():
    result = _run(
        "trainer",
        TOTAL_SEQ_LEN="8192",
        VERIFIER_MAX_MODEL_LEN="8193",
    )

    assert result.returncode == 0, result.stderr
    assert "--total-seq-len 8192" in result.stdout


def test_trainer_rejects_token_budget_smaller_than_context():
    result = _run(
        "trainer",
        TOTAL_SEQ_LEN="32768",
        VERIFIER_MAX_MODEL_LEN="32769",
        VERIFIER_MAX_BATCHED_TOKENS="8192",
    )

    assert result.returncode != 0
    assert "VERIFIER_MAX_BATCHED_TOKENS" in result.stderr


def test_smoke_dry_run_caps_data_context_and_steps():
    result = _run("smoke")

    assert result.returncode == 0, result.stderr
    assert "Prepare smoke dataset (256 samples)" in result.stdout
    assert "--total-seq-len 1024" in result.stdout
    assert "--max-steps 2" in result.stdout
    assert "smoke" in result.stdout
    assert "--on-missing generate" in result.stdout
    assert "-smoke-unit-test" in result.stdout
    assert "--no-resume-from-checkpoint" in result.stdout


def test_smoke_rejects_dataset_too_small_for_world_size():
    result = _run("smoke", SMOKE_SAMPLES="64")

    assert result.returncode != 0
    assert "SMOKE_SAMPLES=64 is too small for 64 training ranks" in result.stderr


def test_smoke_requires_a_shared_run_id():
    result = _run("smoke", SMOKE_RUN_ID="")

    assert result.returncode != 0
    assert "SMOKE_RUN_ID" in result.stderr


@pytest.mark.parametrize("role", ["", "unknown"])
def test_invalid_role_fails(role):
    result = _run(role)

    assert result.returncode != 0
    assert "ROLE must be one of" in result.stderr


@pytest.mark.parametrize("missing", ["VERIFIER_HOST", "MASTER_ADDR", "NODE_RANK"])
def test_trainer_requires_topology(missing):
    result = _run("trainer", **{missing: ""})

    assert result.returncode != 0
    assert missing in result.stderr


def test_node_rank_must_be_in_range():
    result = _run("trainer", NODE_RANK="4")

    assert result.returncode != 0
    assert "NODE_RANK" in result.stderr


def test_shared_config_auto_maps_trainer_to_matching_verifier(tmp_path):
    config = tmp_path / "cluster.env"
    config.write_text(
        "\n".join(
            [
                'CLUSTER_VERIFIER_IPS=("10.0.0.10" "10.0.0.11" "10.0.0.12" "10.0.0.13")',
                'CLUSTER_TRAINER_IPS=("10.0.0.20" "10.0.0.21" "10.0.0.22" "10.0.0.23")',
                "NIC_NAME=eth0",
                "SMOKE_RUN_ID=unit-test",
            ]
        ),
        encoding="utf-8",
    )
    result = _run_container(
        CONFIG_FILE=str(config), NODE_IP="10.0.0.22", TRAINER_MODE="trainer"
    )

    assert result.returncode == 0, result.stderr
    assert "ROLE=trainer" in result.stdout
    assert "NODE_RANK=2" in result.stdout
    assert "NNODES=4" in result.stdout
    assert "MASTER_ADDR=10.0.0.20" in result.stdout
    assert "VERIFIER_HOST=10.0.0.12" in result.stdout
    assert "LOCAL_IP=10.0.0.22" in result.stdout


def test_shared_config_auto_maps_two_trainers_across_four_verifiers(tmp_path):
    config = tmp_path / "cluster-4v2t.env"
    config.write_text(
        "\n".join(
            [
                'CLUSTER_VERIFIER_IPS=("10.0.0.10" "10.0.0.11" "10.0.0.12" "10.0.0.13")',
                'CLUSTER_TRAINER_IPS=("10.0.0.20" "10.0.0.21")',
                "NIC_NAME=eth0",
                "SMOKE_RUN_ID=unit-test",
            ]
        ),
        encoding="utf-8",
    )
    result = _run_container(
        CONFIG_FILE=str(config), NODE_IP="10.0.0.21", TRAINER_MODE="trainer"
    )

    assert result.returncode == 0, result.stderr
    assert "NODE_RANK=1" in result.stdout
    assert "NNODES=2" in result.stdout
    assert "VERIFIER_HOST=10.0.0.11" in result.stdout
    assert "VERIFIER_HOSTS=10.0.0.11" in result.stdout
    assert "10.0.0.13" in result.stdout


def test_shared_config_auto_maps_verifier_and_id(tmp_path):
    config = tmp_path / "cluster.env"
    config.write_text(
        "\n".join(
            [
                'CLUSTER_VERIFIER_IPS=("10.0.0.10" "10.0.0.11" "10.0.0.12" "10.0.0.13")',
                'CLUSTER_TRAINER_IPS=("10.0.0.20" "10.0.0.21" "10.0.0.22" "10.0.0.23")',
                "NIC_NAME=eth0",
            ]
        ),
        encoding="utf-8",
    )
    result = _run_container(CONFIG_FILE=str(config), NODE_IP="10.0.0.11")

    assert result.returncode == 0, result.stderr
    assert "ROLE=verifier" in result.stdout
    assert "VERIFIER_ID=1" in result.stdout
