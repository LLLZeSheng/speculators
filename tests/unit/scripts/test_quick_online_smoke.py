import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = (
    REPO_ROOT / "examples/train/mtp_glm52_ascend_production_4v4t_4k.example.yaml"
)
GENERATOR = REPO_ROOT / "scripts/make_ascend_mtp_online_smoke_config.py"
RENDERER = REPO_ROOT / "scripts/render_ascend_mtp_cluster_yaml.py"


def _write_filled_production_config(path: Path) -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    for index in range(4):
        text = text.replace(f"FILL_VERIFIER_{index}_IP", f"10.0.0.{index + 1}")
        text = text.replace(f"FILL_TRAINER_{index}_IP", f"10.0.1.{index + 1}")
    text = text.replace("FILL_UNIQUE_SMOKE_ID", "production-unused")
    path.write_text(text, encoding="utf-8")


def test_generator_derives_isolated_deleting_online_smoke(tmp_path: Path):
    source = tmp_path / "production.yaml"
    output = tmp_path / "smoke.yaml"
    _write_filled_production_config(source)

    subprocess.run(
        [
            "python",
            str(GENERATOR),
            "--source",
            str(source),
            "--output",
            str(output),
            "--run-id",
            "unit-1k",
            "--smoke-seq-len",
            "1024",
        ],
        check=True,
    )
    rendered = subprocess.run(
        ["python", str(RENDERER), "--config", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "SMOKE_SEQ_LEN=${SMOKE_SEQ_LEN:-1024}" in rendered
    assert "SMOKE_RUN_ID=${SMOKE_RUN_ID:-unit-1k}" in rendered
    assert "TRAINER_DATA_MODE=${TRAINER_DATA_MODE:-online-cache}" in rendered
    assert "MTP_TRAINING_STRATEGY=${MTP_TRAINING_STRATEGY:-sampled_step}" in rendered
    assert "MTP_ACTIVATION_CHECKPOINTING=${MTP_ACTIVATION_CHECKPOINTING:-0}" in rendered
    assert "FSDP_EXPERTS_PER_UNIT=${FSDP_EXPERTS_PER_UNIT:-2}" in rendered
    assert "VERIFIER_GPU_MEMORY_UTILIZATION=" in rendered
    text = output.read_text(encoding="utf-8")
    expected = "/mnt/xds/mtp/spec_train/hidden_states/quick-online-smoke-unit-1k"
    assert f'"hidden_states_path": "{expected}"' not in text
    assert text.count(expected) == 2
    assert "glm52-w4a8-mg13-production-4k" not in text


def test_quick_smoke_wrapper_is_valid_shell():
    wrapper = REPO_ROOT / "examples/train/run_glm52_quick_online_smoke.sh"
    result = subprocess.run(
        ["bash", "-n", str(wrapper)],
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
