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


def test_verifier_dry_run_builds_w4a8_hidden_state_service():
    result = _run("verifier")

    assert result.returncode == 0, result.stderr
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in (
        result.stdout
    )
    assert "/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1" in result.stdout
    assert "scripts/launch_vllm.py" in result.stdout
    assert "--target-layer-ids 78" in result.stdout
    assert "--hidden-states-path" in result.stdout
    assert "/mnt/xds/sfs/spec_train/online_hidden_states/glm52-w4a8" in result.stdout
    assert "--tensor-parallel-size 16" in result.stdout
    assert "--max-model-len 8193" in result.stdout
    assert "--required-devices 16" in result.stdout
    assert "--served-model-name glm52-w4a8-verifier" in result.stdout


def test_trainer_dry_run_builds_three_node_bf16_mtp3_job():
    result = _run("trainer", NODE_RANK="2")

    assert result.returncode == 0, result.stderr
    assert "--nnodes 3" in result.stdout
    assert "--nproc-per-node 16" in result.stdout
    assert "--node-rank 2" in result.stdout
    assert "--master-addr 10.0.0.20" in result.stdout
    assert "--verifier-name-or-path /mnt/xds/sfs/GLM-5.2" in result.stdout
    assert "--generation-model-name-or-path glm52-w4a8-verifier" in result.stdout
    assert "--from-pretrained" in result.stdout
    assert "--speculator-type mtp" in result.stdout
    assert "--num-speculative-steps 3" in result.stdout
    assert "--hidden-states-dtype bfloat16" in result.stdout
    assert "--fsdp-shard" in result.stdout
    assert "--on-missing generate" in result.stdout
    assert "--on-generate delete" in result.stdout
    assert "--force-generate" in result.stdout
    assert "--on-generation-error raise" in result.stdout
    assert "--checkpoint-steps 1000" in result.stdout


def test_verifier_preflight_uses_tensor_parallel_device_count():
    result = _run("verifier", VERIFIER_TP_SIZE="8", NPROC_PER_NODE="16")

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


def test_smoke_dry_run_caps_data_context_and_steps():
    result = _run("smoke")

    assert result.returncode == 0, result.stderr
    assert "Prepare smoke dataset (64 samples)" in result.stdout
    assert "--total-seq-len 1024" in result.stdout
    assert "--max-steps 2" in result.stdout
    assert "smoke" in result.stdout
    assert "--on-missing generate" in result.stdout
    assert "-smoke-unit-test" in result.stdout
    assert "--no-resume-from-checkpoint" in result.stdout


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
    result = _run("trainer", NODE_RANK="3")

    assert result.returncode != 0
    assert "NODE_RANK" in result.stderr
