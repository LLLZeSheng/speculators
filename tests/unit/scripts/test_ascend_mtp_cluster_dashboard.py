import json
from pathlib import Path

from scripts.monitor_ascend_mtp_cluster import (
    ClusterMonitor,
    HTML,
    parse_training_node,
    parse_verifier_node,
)


def test_parse_training_node_reports_progress_and_loss(tmp_path: Path):
    log = tmp_path / "trainer0.log"
    log.write_text(
        "Preflight passed\n"
        "Training epoch 2/5 started\n"
        "train.loss=0.8125, epoch=1, global_step=37\n",
        encoding="utf-8",
    )

    node = parse_training_node(0, "10.0.1.1", [("train", log)])

    assert node["state"] == "running"
    assert node["epoch_current"] == 2
    assert node["epoch_total"] == 5
    assert node["step_current"] == 37
    assert node["loss"] == 0.8125


def test_parse_verifier_node_reports_vllm_metrics(tmp_path: Path):
    host_log = tmp_path / "host.log"
    verifier_log = tmp_path / "verifier.log"
    host_log.write_text("Preflight passed\n", encoding="utf-8")
    verifier_log.write_text(
        "Application startup complete\n"
        "Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: "
        "0.1 tokens/s, Running: 1 reqs, Waiting: 14 reqs, GPU KV cache usage: 1.4%\n"
        'POST /v1/completions HTTP/1.1" 200 OK\n',
        encoding="utf-8",
    )

    node = parse_verifier_node(
        0, "10.0.0.1", host_log, verifier_log, {"ok": True, "code": 200}
    )

    assert node["state"] == "healthy"
    assert node["prompt_tps"] == 102.4
    assert node["generation_tps"] == 0.1
    assert node["running_requests"] == 1
    assert node["waiting_requests"] == 14
    assert node["kv_cache_percent"] == 1.4
    assert node["successful_requests_in_tail"] == 1


def test_cluster_snapshot_uses_shared_orchestrator_logs(tmp_path: Path):
    shared = tmp_path / "shared"
    orchestrator = shared / "spec_train/logs/orchestrator"
    detailed = tmp_path / "detailed"
    orchestrator.mkdir(parents=True)
    detailed.mkdir()
    config = tmp_path / "cluster.yaml"
    config.write_text(
        "version: 1\n"
        "cluster_name: dashboard-test\n"
        "container_name_prefix: test-mtp\n"
        f"shared_root: {shared}\n"
        f"log_root: {detailed}\n"
        "verifier_port: 8077\n"
        "verifier_ips:\n"
        "  - 10.0.0.1\n  - 10.0.0.2\n  - 10.0.0.3\n  - 10.0.0.4\n"
        "trainer_ips:\n"
        "  - 10.0.1.1\n  - 10.0.1.2\n  - 10.0.1.3\n  - 10.0.1.4\n",
        encoding="utf-8",
    )
    (orchestrator / "test-mtp-trainer0.host.log").write_text(
        "Training epoch 1/5 started\n", encoding="utf-8"
    )

    snapshot = ClusterMonitor(config, probe=False).snapshot()

    assert snapshot["cluster_name"] == "dashboard-test"
    assert snapshot["trainers"][0]["state"] == "running"
    assert len(snapshot["verifiers"]) == 4
    assert len(snapshot["trainers"]) == 4
    assert snapshot["summary"]["total_verifiers"] == 4
    assert snapshot["summary"]["total_trainers"] == 4
    json.dumps(snapshot)
    assert "/api/status" in HTML


def test_cluster_snapshot_supports_two_trainers(tmp_path: Path):
    shared = tmp_path / "shared"
    (shared / "spec_train/logs/orchestrator").mkdir(parents=True)
    detailed = tmp_path / "detailed"
    detailed.mkdir()
    config = tmp_path / "cluster-4v2t.yaml"
    config.write_text(
        "version: 1\n"
        "cluster_name: dashboard-4v2t\n"
        "container_name_prefix: test-mtp\n"
        f"shared_root: {shared}\n"
        f"log_root: {detailed}\n"
        "verifier_port: 8077\n"
        "verifier_ips:\n"
        "  - 10.0.0.1\n  - 10.0.0.2\n  - 10.0.0.3\n  - 10.0.0.4\n"
        "trainer_ips:\n"
        "  - 10.0.1.1\n  - 10.0.1.2\n",
        encoding="utf-8",
    )

    snapshot = ClusterMonitor(config, probe=False).snapshot()

    assert len(snapshot["verifiers"]) == 4
    assert len(snapshot["trainers"]) == 2
    assert snapshot["summary"]["total_verifiers"] == 4
    assert snapshot["summary"]["total_trainers"] == 2
    assert "d.summary.total_trainers" in HTML
