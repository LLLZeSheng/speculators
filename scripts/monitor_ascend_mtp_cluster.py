#!/usr/bin/env python3
"""Serve a dependency-free dashboard for Ascend multi-node MTP training."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from render_ascend_mtp_cluster_yaml import load_flat_yaml
except ImportError:  # Imported as scripts.monitor_ascend_mtp_cluster in tests.
    from scripts.render_ascend_mtp_cluster_yaml import load_flat_yaml


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
ERROR_RE = re.compile(
    r"Traceback|\b(?:ERROR|RuntimeError|ValueError|AssertionError)\b|"
    r"OutOfMemory|out of memory|Killed|ERR\d{5}|EI\d{4}|Connection error",
    re.IGNORECASE,
)
WARNING_RE = re.compile(r"\bWARNING\b|UserWarning:", re.IGNORECASE)
EPOCH_RE = re.compile(r"Training epoch\s+(\d+)/(\d+)\s+(started|completed)")
GLOBAL_STEP_RE = re.compile(r"[\"']?global_step[\"']?\s*[:=]\s*(\d+)")
LOSS_RE = re.compile(
    r"(?:train\.)?[\"']?loss(?:_epoch)?[\"']?\s*[:=]\s*([-+\d.eE]+)"
)
PROGRESS_RE = re.compile(r"Epoch\s+\d+.*?\b(\d+)/(\d+)\b")
VERIFIER_METRICS_RE = re.compile(
    r"Avg prompt throughput:\s*([\d.]+) tokens/s,\s*"
    r"Avg generation throughput:\s*([\d.]+) tokens/s,\s*"
    r"Running:\s*(\d+) reqs,\s*Waiting:\s*(\d+) reqs,\s*"
    r"GPU KV cache usage:\s*([\d.]+)%"
)
TRAIN_STARTUP_RE = re.compile(
    r"TRAIN_STARTUP phase=(\S+) status=(\S+) rank=(\d+) local_rank=(\d+) "
    r"elapsed_seconds=([\d.]+)(?: detail=(\S+))?"
)
STARTUP_PHASES: dict[str, tuple[str, int]] = {
    "capture_initial_state": ("准备初始权重", 15),
    "fsdp_shard": ("FSDP 参数分片", 35),
    "checkpoint_load": ("加载训练检查点", 55),
    "initial_weight_sync": ("同步初始权重", 65),
    "startup_barrier": ("等待训练节点汇合", 80),
    "optimizer_init": ("初始化优化器", 92),
    "first_batch": ("等待首个训练批次", 98),
    "ready": ("训练初始化完成", 100),
}


def read_tail(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    """Read a bounded UTF-8 tail without loading a multi-gigabyte log."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
                handle.readline()
            data = handle.read()
    except (FileNotFoundError, OSError):
        return ""
    return ANSI_RE.sub("", data.decode("utf-8", errors="replace")).replace("\r", "\n")


def last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    result = None
    for result in pattern.finditer(text):
        pass
    return result


def compact_lines(text: str, count: int = 8) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-count:]


def latest_error(text: str) -> str | None:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if ERROR_RE.search(lines[index]):
            return " ".join(line.strip() for line in lines[index : index + 3])[:600]
    return None


def file_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False, "mtime": None, "age_seconds": None}
    return {
        "path": str(path),
        "exists": True,
        "mtime": stat.st_mtime,
        "age_seconds": max(0, int(time.time() - stat.st_mtime)),
        "size_bytes": stat.st_size,
    }


def classify_training(text: str, age: int | None) -> tuple[str, str]:
    error = latest_error(text)
    last_training = max(text.rfind("Training epoch"), text.rfind("Starting fresh"))
    last_error = max(text.rfind("Traceback"), text.rfind("RuntimeError:"), text.rfind("ERROR"))
    if error and last_error > last_training:
        return "failed", "训练失败"
    if "Training run completed" in text or "Training completed" in text:
        return "complete", "训练完成"
    if "Saving checkpoint to" in text or "Writing model shards:" in text:
        return "saving", "保存检查点"
    startup = last_match(TRAIN_STARTUP_RE, text)
    last_epoch = text.rfind("Training epoch")
    if startup and startup.start() > last_epoch:
        phase_name, _progress = STARTUP_PHASES.get(
            startup.group(1), (startup.group(1), 5)
        )
        status = startup.group(2)
        if status == "failed":
            return "failed", f"{phase_name}失败"
        if status == "heartbeat":
            return "waiting", phase_name
        if startup.group(1) == "first_batch" and status == "started":
            return "waiting", phase_name
        if startup.group(1) == "first_batch" and status == "completed":
            return "running", "训练中"
        if startup.group(1) == "ready":
            return "starting", phase_name
        return "loading", phase_name
    if "Training epoch" in text:
        if age is not None and age > 300:
            return "stale", "训练无新日志"
        return "running", "训练中"
    if "No previous training checkpoint" in text or "LOAD REPORT" in text:
        return "loading", "加载训练模型"
    if "Waiting for local verifier-weight load lock" in text:
        return "waiting", "等待权重加载锁"
    if "Loading weights:" in text or "Loading from local directory" in text:
        return "loading", "加载权重"
    if "Preflight passed" in text:
        return "starting", "预检完成"
    if text:
        return ("stale", "日志已停止") if age is not None and age > 300 else ("starting", "启动中")
    return "missing", "尚无日志"


def parse_training_node(index: int, ip: str, paths: list[tuple[str, Path]]) -> dict[str, Any]:
    existing = [(role, path) for role, path in paths if path.exists()]
    if existing:
        role, path = max(existing, key=lambda item: item[1].stat().st_mtime)
    else:
        role, path = paths[0]
    info = file_info(path)
    text = read_tail(path)
    state, phase = classify_training(text, info["age_seconds"])
    epoch_match = last_match(EPOCH_RE, text)
    step_match = last_match(GLOBAL_STEP_RE, text)
    progress_match = last_match(PROGRESS_RE, text)
    loss_match = last_match(LOSS_RE, text)
    startup_match = last_match(TRAIN_STARTUP_RE, text)
    epoch_current = int(epoch_match.group(1)) if epoch_match else None
    epoch_total = int(epoch_match.group(2)) if epoch_match else None
    if epoch_match and epoch_match.group(3) == "completed":
        epoch_current = min(epoch_current + 1, epoch_total)
    step_current = int(step_match.group(1)) if step_match else None
    step_total = None
    if progress_match:
        step_current = int(progress_match.group(1))
        step_total = int(progress_match.group(2))
    startup_phase = startup_match.group(1) if startup_match else None
    startup_status = startup_match.group(2) if startup_match else None
    startup_elapsed_seconds = (
        float(startup_match.group(5)) if startup_match else None
    )
    startup_progress = (
        STARTUP_PHASES.get(startup_phase, (startup_phase, 5))[1]
        if startup_phase
        else None
    )
    return {
        "kind": "trainer",
        "index": index,
        "ip": ip,
        "role": role,
        "state": state,
        "phase": phase,
        "epoch_current": epoch_current,
        "epoch_total": epoch_total,
        "step_current": step_current,
        "step_total": step_total,
        "loss": float(loss_match.group(1)) if loss_match else None,
        "startup_phase": startup_phase,
        "startup_status": startup_status,
        "startup_elapsed_seconds": startup_elapsed_seconds,
        "startup_progress": startup_progress,
        "recent_errors": len(ERROR_RE.findall(text)),
        "recent_warnings": len(WARNING_RE.findall(text)),
        "latest_error": latest_error(text),
        "tail": compact_lines(text),
        "log": info,
    }


def probe_health(ip: str, port: int, timeout: float = 1.2) -> dict[str, Any]:
    started = time.monotonic()
    try:
        # Verifier IPs are cluster-internal. Never route probes through a
        # download proxy inherited from HTTP_PROXY/HTTPS_PROXY.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{ip}:{port}/health", timeout=timeout) as response:
            code = response.status
        return {"ok": code == 200, "code": code, "latency_ms": round((time.monotonic() - started) * 1000)}
    except (OSError, urllib.error.URLError) as error:
        return {"ok": False, "code": None, "latency_ms": None, "error": str(error)[:160]}


def parse_verifier_node(
    index: int,
    ip: str,
    host_path: Path,
    verifier_path: Path,
    health: dict[str, Any],
) -> dict[str, Any]:
    host_text = read_tail(host_path, 512 * 1024)
    verifier_text = read_tail(verifier_path)
    text = host_text + "\n" + verifier_text
    primary_path = verifier_path if verifier_path.exists() else host_path
    info = file_info(primary_path)
    metric = last_match(VERIFIER_METRICS_RE, text)
    error = latest_error(text)
    ready_marker = "Application startup complete" in text or '"GET /health' in text
    if not health.get("ok") and error:
        state, phase = "failed", "服务异常"
    elif health.get("ok"):
        state, phase = "healthy", "服务健康"
    elif ready_marker:
        state, phase = "stale", "健康检查不可达"
    elif text:
        state, phase = "loading", "模型加载中"
    else:
        state, phase = "missing", "尚无日志"
    return {
        "kind": "verifier",
        "index": index,
        "ip": ip,
        "state": state,
        "phase": phase,
        "health": health,
        "prompt_tps": float(metric.group(1)) if metric else None,
        "generation_tps": float(metric.group(2)) if metric else None,
        "running_requests": int(metric.group(3)) if metric else None,
        "waiting_requests": int(metric.group(4)) if metric else None,
        "kv_cache_percent": float(metric.group(5)) if metric else None,
        "successful_requests_in_tail": len(re.findall(r'POST /v1/(?:chat/)?completions.*? 200 OK', text)),
        "hidden_state_repairs_in_tail": text.count("Restored TP-sharded extract_hidden_states tensor"),
        "recent_errors": len(ERROR_RE.findall(text)),
        "recent_warnings": len(WARNING_RE.findall(text)),
        "latest_error": error,
        "tail": compact_lines(verifier_text or host_text),
        "log": info,
    }


class ClusterMonitor:
    def __init__(self, config_path: Path, probe: bool = True):
        self.config_path = config_path
        self.config = load_flat_yaml(config_path)
        self.probe = probe
        self.shared_root = Path(str(self.config.get("shared_root", "/mnt/xds/mtp")))
        self.orchestrator_root = self.shared_root / "spec_train/logs/orchestrator"
        self.log_root = Path(str(self.config["log_root"]))
        self.prefix = str(self.config["container_name_prefix"])
        self.verifier_ips = [str(value) for value in self.config["verifier_ips"]]
        self.trainer_ips = [str(value) for value in self.config["trainer_ips"]]
        self.verifier_port = int(str(self.config.get("verifier_port", "8077")))

    def snapshot(self) -> dict[str, Any]:
        if self.probe:
            with ThreadPoolExecutor(max_workers=4) as pool:
                health = list(pool.map(lambda ip: probe_health(ip, self.verifier_port), self.verifier_ips))
        else:
            health = [{"ok": False, "code": None, "disabled": True} for _ in self.verifier_ips]
        verifiers = [
            parse_verifier_node(
                index,
                ip,
                self.orchestrator_root / f"{self.prefix}-verifier{index}.host.log",
                self.log_root / f"verifier{index}/verifier.log",
                health[index],
            )
            for index, ip in enumerate(self.verifier_ips)
        ]
        trainers = [
            parse_training_node(
                index,
                ip,
                [
                    ("train", self.orchestrator_root / f"{self.prefix}-trainer{index}.host.log"),
                    ("smoke", self.orchestrator_root / f"{self.prefix}-smoke{index}.host.log"),
                ],
            )
            for index, ip in enumerate(self.trainer_ips)
        ]
        failed = sum(node["state"] == "failed" for node in [*verifiers, *trainers])
        healthy_verifiers = sum(node["state"] == "healthy" for node in verifiers)
        active_trainers = sum(
            node["state"] in {"starting", "loading", "waiting", "running", "saving"}
            for node in trainers
        )
        if failed:
            overall = "failed"
        elif active_trainers:
            overall = "running"
        elif healthy_verifiers == len(verifiers):
            overall = "ready"
        else:
            overall = "starting"
        return {
            "generated_at": time.time(),
            "cluster_name": str(self.config.get("cluster_name", self.prefix)),
            "overall": overall,
            "summary": {
                "healthy_verifiers": healthy_verifiers,
                "total_verifiers": len(verifiers),
                "active_trainers": active_trainers,
                "total_trainers": len(trainers),
                "failed_nodes": failed,
                "total_prompt_tps": round(sum(node["prompt_tps"] or 0 for node in verifiers), 2),
                "total_generation_tps": round(sum(node["generation_tps"] or 0 for node in verifiers), 2),
            },
            "verifiers": verifiers,
            "trainers": trainers,
            "config_path": str(self.config_path),
        }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Ascend MTP Control Center</title>
<style>
:root {
  color-scheme: dark;
  --bg: #070b13;
  --surface: #0d1420;
  --surface-2: #111b2a;
  --surface-3: #162234;
  --line: rgba(148, 163, 184, .14);
  --line-strong: rgba(148, 163, 184, .24);
  --text: #f4f7fb;
  --muted: #8a9aaf;
  --faint: #5b6b7f;
  --cyan: #28d7c0;
  --blue: #5d8dff;
  --violet: #9b7bff;
  --green: #45dda0;
  --amber: #ffbe5c;
  --red: #ff6b7a;
  --shadow: 0 20px 70px rgba(0, 0, 0, .28);
  --radius: 18px;
}
* { box-sizing: border-box; }
html { min-width: 320px; background: var(--bg); }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--text);
  font: 14px/1.5 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(41, 120, 255, .16), transparent 65%),
    radial-gradient(720px 460px at 94% 2%, rgba(40, 215, 192, .09), transparent 64%),
    var(--bg);
}
button { font: inherit; }
.shell { width: min(1540px, 100%); margin: 0 auto; padding: 24px 28px 42px; }
.nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  min-height: 58px;
  margin-bottom: 22px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  position: relative;
  width: 38px;
  height: 38px;
  border: 1px solid rgba(93, 141, 255, .42);
  border-radius: 12px;
  background: linear-gradient(145deg, rgba(93, 141, 255, .22), rgba(40, 215, 192, .08));
  box-shadow: inset 0 0 24px rgba(93, 141, 255, .12);
}
.brand-mark::before, .brand-mark::after {
  content: "";
  position: absolute;
  border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 14px rgba(40, 215, 192, .7);
}
.brand-mark::before { width: 8px; height: 8px; left: 8px; top: 9px; }
.brand-mark::after { width: 6px; height: 6px; right: 8px; bottom: 9px; }
.brand-mark i { position: absolute; inset: 13px 10px; border-top: 1px solid var(--blue); transform: rotate(32deg); }
.brand-copy strong { display: block; font-size: 14px; letter-spacing: .08em; }
.brand-copy span { color: var(--muted); font-size: 11px; letter-spacing: .04em; }
.nav-actions { display: flex; align-items: center; gap: 9px; }
.refresh-info { color: var(--muted); font-size: 12px; margin-right: 4px; }
.control {
  min-height: 34px;
  padding: 7px 12px;
  border: 1px solid var(--line-strong);
  border-radius: 10px;
  color: #cbd6e4;
  background: rgba(17, 27, 42, .78);
  cursor: pointer;
  transition: border-color .2s, background .2s, transform .2s;
}
.control:hover { border-color: rgba(93, 141, 255, .55); background: var(--surface-3); }
.control:active { transform: translateY(1px); }
.hero {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 28px;
  min-height: 168px;
  padding: 30px 32px;
  border: 1px solid var(--line);
  border-radius: 24px;
  background: linear-gradient(120deg, rgba(17, 27, 42, .96), rgba(10, 17, 28, .92));
  box-shadow: var(--shadow);
}
.hero::after {
  content: "";
  position: absolute;
  width: 360px;
  height: 360px;
  right: -90px;
  top: -200px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(40, 215, 192, .13), transparent 68%);
  pointer-events: none;
}
.eyebrow { color: var(--cyan); font: 700 11px/1 ui-monospace, monospace; letter-spacing: .16em; text-transform: uppercase; }
h1 { margin: 12px 0 8px; font-size: clamp(25px, 3vw, 38px); line-height: 1.12; letter-spacing: -.035em; }
.hero-sub { color: var(--muted); font-size: 13px; }
.cluster-state {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 210px;
  padding: 14px 16px;
  border: 1px solid var(--line-strong);
  border-radius: 14px;
  background: rgba(7, 11, 19, .48);
  backdrop-filter: blur(10px);
}
.pulse { width: 11px; height: 11px; border-radius: 50%; background: var(--amber); box-shadow: 0 0 0 5px rgba(255, 190, 92, .1); }
.cluster-state strong { display: block; font-size: 14px; }
.cluster-state small { display: block; margin-top: 2px; color: var(--muted); }
body[data-overall="running"] .pulse, body[data-overall="ready"] .pulse { background: var(--green); box-shadow: 0 0 0 5px rgba(69, 221, 160, .1); }
body[data-overall="failed"] .pulse { background: var(--red); box-shadow: 0 0 0 5px rgba(255, 107, 122, .1); }
.metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 14px 0 0; }
.metric {
  position: relative;
  min-width: 0;
  padding: 17px 18px;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: rgba(13, 20, 32, .86);
}
.metric::before { content: ""; position: absolute; left: 0; top: 18px; bottom: 18px; width: 2px; background: var(--metric-color, var(--blue)); border-radius: 2px; }
.metric-label { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
.metric-value { display: block; overflow: hidden; margin-top: 8px; font-size: clamp(21px, 2.2vw, 29px); font-weight: 720; line-height: 1; letter-spacing: -.03em; text-overflow: ellipsis; }
.metric-unit { margin-left: 4px; color: var(--faint); font-size: 11px; font-weight: 500; letter-spacing: 0; }
.topology {
  display: grid;
  grid-template-columns: auto 1fr auto 1fr auto;
  align-items: center;
  gap: 12px;
  margin-top: 14px;
  padding: 13px 16px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(13, 20, 32, .62);
}
.topology-label { color: var(--muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.topology-line { height: 1px; background: linear-gradient(90deg, var(--line-strong), rgba(40, 215, 192, .4), var(--line-strong)); }
.topology-group { display: flex; align-items: center; gap: 7px; }
.mini-node { width: 9px; height: 9px; border-radius: 3px; background: var(--faint); box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }
.mini-node.healthy, .mini-node.running, .mini-node.complete { background: var(--green); box-shadow: 0 0 8px rgba(69,221,160,.35); }
.mini-node.failed { background: var(--red); box-shadow: 0 0 8px rgba(255,107,122,.4); }
.mini-node.loading, .mini-node.starting { background: var(--blue); }
.mini-node.stale, .mini-node.waiting, .mini-node.saving { background: var(--amber); }
.section { margin-top: 28px; }
.section-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.section-title { display: flex; align-items: center; gap: 10px; }
.section-icon { display: grid; place-items: center; width: 29px; height: 29px; border: 1px solid var(--line); border-radius: 9px; color: var(--cyan); background: var(--surface); font: 700 12px/1 ui-monospace, monospace; }
h2 { margin: 0; font-size: 16px; letter-spacing: -.01em; }
.section-note { margin-top: 2px; color: var(--muted); font-size: 12px; }
.legend { display: flex; gap: 13px; color: var(--muted); font-size: 11px; }
.legend span { display: flex; align-items: center; gap: 5px; }
.legend i { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
.legend .warn { background: var(--amber); }
.legend .bad { background: var(--red); }
.grid { display: grid; grid-template-columns: repeat(4, minmax(250px, 1fr)); gap: 12px; }
.node {
  position: relative;
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: linear-gradient(150deg, rgba(17, 27, 42, .96), rgba(11, 18, 29, .96));
  box-shadow: 0 12px 36px rgba(0, 0, 0, .15);
  transition: transform .2s, border-color .2s, box-shadow .2s;
}
.node:hover { transform: translateY(-2px); border-color: var(--line-strong); box-shadow: 0 18px 44px rgba(0,0,0,.24); }
.node::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: var(--state-color, var(--faint)); }
.node[data-state="healthy"], .node[data-state="running"], .node[data-state="complete"] { --state-color: var(--green); }
.node[data-state="failed"] { --state-color: var(--red); }
.node[data-state="stale"], .node[data-state="waiting"], .node[data-state="saving"] { --state-color: var(--amber); }
.node[data-state="loading"], .node[data-state="starting"] { --state-color: var(--blue); }
.node-main { padding: 17px 17px 15px 19px; }
.node-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.node-ident { display: flex; min-width: 0; gap: 10px; }
.node-index { display: grid; place-items: center; flex: 0 0 auto; width: 34px; height: 34px; border: 1px solid var(--line-strong); border-radius: 10px; color: #dbe7f5; background: var(--surface-3); font: 700 12px/1 ui-monospace, monospace; }
.node-name { font-weight: 700; letter-spacing: -.01em; }
.node-ip { overflow: hidden; margin-top: 2px; color: var(--muted); font: 11px/1.4 ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
.state {
  flex: 0 0 auto;
  max-width: 112px;
  overflow: hidden;
  padding: 4px 8px;
  border: 1px solid color-mix(in srgb, var(--state-color, var(--faint)) 28%, transparent);
  border-radius: 999px;
  color: var(--state-color, var(--muted));
  background: color-mix(in srgb, var(--state-color, var(--faint)) 8%, transparent);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.progress-meta { display: flex; justify-content: space-between; gap: 8px; margin: 17px 0 7px; color: var(--muted); font-size: 10px; }
.progress-meta strong { color: #d7e0ec; font-weight: 600; }
.progress { height: 5px; overflow: hidden; border-radius: 99px; background: #070c14; }
.progress i { display: block; width: 0; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--blue), var(--cyan)); box-shadow: 0 0 12px rgba(40, 215, 192, .25); transition: width .5s ease; }
.facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-top: 15px; }
.fact { min-width: 0; padding: 9px 9px 8px; border: 1px solid rgba(148,163,184,.08); border-radius: 10px; background: rgba(7, 12, 20, .46); }
.fact span { display: block; overflow: hidden; color: var(--muted); font-size: 9px; letter-spacing: .03em; text-overflow: ellipsis; text-transform: uppercase; white-space: nowrap; }
.fact b { display: block; overflow: hidden; margin-top: 4px; color: #e6edf6; font: 650 13px/1.2 ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
.error { margin-top: 12px; padding: 10px 11px; border: 1px solid rgba(255,107,122,.22); border-radius: 10px; color: #ffb1b9; background: rgba(95, 25, 38, .24); font-size: 11px; word-break: break-word; }
.error strong { display: block; margin-bottom: 3px; color: var(--red); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }
.node-foot { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 13px; color: var(--faint); font-size: 10px; }
.node-foot .fresh { color: var(--green); }
.node-foot .old { color: var(--amber); }
details { border-top: 1px solid var(--line); background: rgba(5, 9, 15, .34); }
summary { display: flex; align-items: center; justify-content: space-between; padding: 10px 17px 10px 19px; color: var(--muted); font-size: 10px; cursor: pointer; list-style: none; user-select: none; }
summary::-webkit-details-marker { display: none; }
summary::after { content: "+"; color: var(--faint); font: 15px/1 ui-monospace, monospace; }
details[open] summary::after { content: "−"; }
.tail { max-height: 170px; overflow: auto; margin: 0; padding: 0 17px 15px 19px; color: #91a3b8; font: 10px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; word-break: break-word; }
.empty { padding: 28px; border: 1px dashed var(--line-strong); border-radius: var(--radius); color: var(--muted); text-align: center; }
.footer { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-top: 28px; padding: 16px 2px 0; border-top: 1px solid var(--line); color: var(--faint); font-size: 10px; }
.footer code { overflow: hidden; max-width: 60vw; color: var(--muted); text-overflow: ellipsis; white-space: nowrap; }
.offline-banner { display: none; margin-top: 14px; padding: 11px 14px; border: 1px solid rgba(255,107,122,.25); border-radius: 12px; color: #ffb1b9; background: rgba(95,25,38,.2); }
body.offline .offline-banner { display: block; }
@media (max-width: 1200px) {
  .grid { grid-template-columns: repeat(2, minmax(260px, 1fr)); }
  .metrics { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 720px) {
  .shell { padding: 16px 14px 30px; }
  .nav { align-items: flex-start; }
  .brand-copy span, .refresh-info { display: none; }
  .hero { align-items: flex-start; flex-direction: column; min-height: 0; padding: 24px 20px; }
  .cluster-state { width: 100%; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .metric:last-child { grid-column: span 2; }
  .topology { grid-template-columns: auto 1fr; }
  .topology-line { display: none; }
  .grid { grid-template-columns: 1fr; }
  .legend { display: none; }
  .facts { grid-template-columns: repeat(2, 1fr); }
  .footer { align-items: flex-start; flex-direction: column; }
  .footer code { max-width: 100%; }
}
</style>
</head>
<body data-overall="starting">
<div class="shell">
  <nav class="nav">
    <div class="brand">
      <div class="brand-mark"><i></i></div>
      <div class="brand-copy"><strong>MTP CONTROL</strong><span>ASCEND DISTRIBUTED TRAINING</span></div>
    </div>
    <div class="nav-actions">
      <span class="refresh-info" id="refreshInfo">5 秒后刷新</span>
      <button class="control" id="pauseButton" type="button">暂停</button>
      <button class="control" id="refreshButton" type="button">立即刷新</button>
    </div>
  </nav>

  <header class="hero">
    <div>
      <div class="eyebrow">Cluster overview</div>
      <h1 id="title">Ascend MTP Cluster</h1>
      <div class="hero-sub" id="stamp">正在读取集群状态…</div>
    </div>
    <div class="cluster-state">
      <i class="pulse"></i>
      <div><strong id="overall">正在连接</strong><small id="overallNote">等待第一份状态快照</small></div>
    </div>
  </header>

  <div class="offline-banner" id="offlineBanner">无法读取状态 API，页面将继续自动重试。</div>
  <section class="metrics" id="summary"></section>
  <section class="topology">
    <span class="topology-label">拓扑状态</span>
    <div class="topology-line"></div>
    <div class="topology-group" id="verifierTopology"></div>
    <div class="topology-line"></div>
    <div class="topology-group" id="trainerTopology"></div>
  </section>

  <section class="section">
    <div class="section-head">
      <div class="section-title"><span class="section-icon">V</span><div><h2>Verifier 节点</h2><div class="section-note">隐藏状态生成与推理服务</div></div></div>
      <div class="legend"><span><i></i>正常</span><span><i class="warn"></i>等待</span><span><i class="bad"></i>异常</span></div>
    </div>
    <div class="grid" id="verifiers"></div>
  </section>

  <section class="section">
    <div class="section-head">
      <div class="section-title"><span class="section-icon">T</span><div><h2>Trainer 节点</h2><div class="section-note">FSDP 分布式训练进度</div></div></div>
    </div>
    <div class="grid" id="trainers"></div>
  </section>

  <footer class="footer"><span>只读监控 · 健康检查与日志解析 · 自动刷新</span><code id="config"></code></footer>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '—').replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[char]));
const number = value => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
const age = seconds => seconds == null ? '未知' : seconds < 10 ? '刚刚' : seconds < 60 ? `${seconds} 秒前` : seconds < 3600 ? `${Math.floor(seconds / 60)} 分钟前` : `${Math.floor(seconds / 3600)} 小时前`;
const stateText = {failed:'集群异常', running:'训练运行中', ready:'Verifier 就绪', starting:'正在启动'};
const stateNote = {failed:'检测到需要处理的节点', running:'在线生成与训练链路工作中', ready:'等待训练任务启动', starting:'正在等待节点和日志'};
let paused = false;
let remaining = 5;
let refreshing = false;

function metric(label, value, unit, color) {
  return `<article class="metric" style="--metric-color:${color}"><div class="metric-label"><span>${esc(label)}</span></div><strong class="metric-value">${esc(value)}${unit ? `<span class="metric-unit">${esc(unit)}</span>` : ''}</strong></article>`;
}
function fact(label, value) {
  return `<div class="fact"><span>${esc(label)}</span><b title="${esc(value)}">${esc(value)}</b></div>`;
}
function progressInfo(node) {
  if (node.kind === 'verifier') {
    const value = Math.max(0, Math.min(100, node.kv_cache_percent || 0));
    return {label:'KV Cache', value, text:node.kv_cache_percent == null ? '暂无数据' : `${number(node.kv_cache_percent)}%`};
  }
  if (node.startup_progress != null && ['started', 'heartbeat'].includes(node.startup_status)) {
    const elapsed = node.startup_elapsed_seconds == null ? '' : ` · ${number(node.startup_elapsed_seconds)}s`;
    return {label:'训练初始化', value:node.startup_progress, text:`${node.phase}${elapsed}`};
  }
  if (node.step_total) {
    const value = Math.max(0, Math.min(100, 100 * (node.step_current || 0) / node.step_total));
    return {label:'当前 Epoch 步数', value, text:`${number(node.step_current || 0)} / ${number(node.step_total)}`};
  }
  if (node.epoch_total) {
    const value = Math.max(0, Math.min(100, 100 * (node.epoch_current || 0) / node.epoch_total));
    return {label:'训练 Epoch', value, text:`${number(node.epoch_current || 0)} / ${number(node.epoch_total)}`};
  }
  if (node.startup_progress != null) {
    const elapsed = node.startup_elapsed_seconds == null ? '' : ` · ${number(node.startup_elapsed_seconds)}s`;
    return {label:'训练初始化', value:node.startup_progress, text:`${node.phase}${elapsed}`};
  }
  return {label:'启动进度', value:node.state === 'complete' ? 100 : 0, text:node.phase};
}
function nodeCard(node) {
  const progress = progressInfo(node);
  const freshness = node.log && node.log.age_seconds != null && node.log.age_seconds < 60;
  const facts = node.kind === 'verifier' ? [
    fact('Prompt TPS', number(node.prompt_tps)),
    fact('Generation TPS', number(node.generation_tps)),
    fact('Running / Waiting', `${number(node.running_requests)} / ${number(node.waiting_requests)}`),
    fact('健康延迟', node.health && node.health.latency_ms != null ? `${node.health.latency_ms} ms` : '—'),
    fact('成功请求', number(node.successful_requests_in_tail)),
    fact('警告 / 错误', `${number(node.recent_warnings)} / ${number(node.recent_errors)}`)
  ] : [
    fact('当前阶段', node.phase || '—'),
    fact('Epoch', node.epoch_total ? `${number(node.epoch_current || 0)} / ${number(node.epoch_total)}` : '—'),
    fact('Global Step', number(node.step_current)),
    fact('Loss', number(node.loss)),
    fact('日志更新', age(node.log && node.log.age_seconds)),
    fact('警告 / 错误', `${number(node.recent_warnings)} / ${number(node.recent_errors)}`)
  ];
  const error = node.latest_error ? `<div class="error"><strong>Latest error</strong>${esc(node.latest_error)}</div>` : '';
  const tail = (node.tail || []).length ? esc(node.tail.join('\n')) : '暂无日志内容';
  return `<article class="node" data-state="${esc(node.state)}">
    <div class="node-main">
      <div class="node-head">
        <div class="node-ident"><span class="node-index">${node.kind === 'verifier' ? 'V' : 'T'}${node.index}</span><div><div class="node-name">${node.kind === 'verifier' ? 'Verifier' : 'Trainer'} ${node.index}</div><div class="node-ip">${esc(node.ip)}</div></div></div>
        <span class="state">${esc(node.phase)}</span>
      </div>
      <div class="progress-meta"><span>${esc(progress.label)}</span><strong>${esc(progress.text)}</strong></div>
      <div class="progress"><i style="width:${progress.value}%"></i></div>
      <div class="facts">${facts.join('')}</div>
      ${error}
      <div class="node-foot"><span class="${freshness ? 'fresh' : 'old'}">● ${freshness ? '日志活跃' : '日志更新 ' + age(node.log && node.log.age_seconds)}</span><span>${esc(node.state)}</span></div>
    </div>
    <details><summary>查看最近日志</summary><pre class="tail">${tail}</pre></details>
  </article>`;
}
function topology(nodes, prefix) {
  return `<span class="topology-label">${prefix}</span>${nodes.map(node => `<i class="mini-node ${esc(node.state)}" title="${prefix}${node.index} · ${esc(node.phase)}"></i>`).join('')}`;
}
function render(data) {
  document.body.classList.remove('offline');
  document.body.dataset.overall = data.overall;
  document.title = `${data.cluster_name} · MTP Control`;
  $('title').textContent = data.cluster_name;
  $('stamp').textContent = `状态快照 ${new Date(data.generated_at * 1000).toLocaleString()} · 5 秒自动刷新`;
  $('overall').textContent = stateText[data.overall] || data.overall;
  $('overallNote').textContent = stateNote[data.overall] || '集群状态已更新';
  const summary = data.summary;
  $('summary').innerHTML = [
    metric('Verifier 健康', `${summary.healthy_verifiers} / ${summary.total_verifiers}`, '', 'var(--green)'),
    metric('Trainer 活跃', `${summary.active_trainers} / ${summary.total_trainers}`, '', 'var(--cyan)'),
    metric('异常节点', summary.failed_nodes, '', summary.failed_nodes ? 'var(--red)' : 'var(--green)'),
    metric('Prompt 吞吐', number(summary.total_prompt_tps), 'tok/s', 'var(--blue)'),
    metric('Generation 吞吐', number(summary.total_generation_tps), 'tok/s', 'var(--violet)')
  ].join('');
  $('verifierTopology').innerHTML = topology(data.verifiers, 'V');
  $('trainerTopology').innerHTML = topology(data.trainers, 'T');
  $('verifiers').innerHTML = data.verifiers.length ? data.verifiers.map(nodeCard).join('') : '<div class="empty">未配置 verifier</div>';
  $('trainers').innerHTML = data.trainers.length ? data.trainers.map(nodeCard).join('') : '<div class="empty">未配置 trainer</div>';
  $('config').textContent = data.config_path;
  $('config').title = data.config_path;
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  $('refreshButton').textContent = '刷新中…';
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    document.body.classList.add('offline');
    $('offlineBanner').textContent = `状态刷新失败：${error.message}。页面将继续自动重试。`;
    $('stamp').textContent = `最后刷新失败 · ${new Date().toLocaleTimeString()}`;
  } finally {
    refreshing = false;
    remaining = 5;
    $('refreshButton').textContent = '立即刷新';
  }
}
$('pauseButton').addEventListener('click', () => {
  paused = !paused;
  $('pauseButton').textContent = paused ? '继续刷新' : '暂停';
  $('refreshInfo').textContent = paused ? '自动刷新已暂停' : `${remaining} 秒后刷新`;
});
$('refreshButton').addEventListener('click', refresh);
setInterval(() => {
  if (paused) return;
  remaining -= 1;
  if (remaining <= 0) refresh();
  $('refreshInfo').textContent = `${Math.max(remaining, 0)} 秒后刷新`;
}, 1000);
refresh();
</script>
</body>
</html>'''


def make_handler(monitor: ClusterMonitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path == "/" or self.path.startswith("/?"):
                self.send_payload(HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                payload = json.dumps(monitor.snapshot(), ensure_ascii=False).encode()
                self.send_payload(payload, "application/json; charset=utf-8")
            elif self.path == "/health":
                self.send_payload(b'{"ok":true}', "application/json")
            else:
                self.send_payload(b'{"error":"not found"}', "application/json", 404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--no-probe", action="store_true", help="do not call verifier /health endpoints")
    parser.add_argument("--json", action="store_true", help="print one snapshot and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor = ClusterMonitor(args.config, probe=not args.no_probe)
    if args.json:
        print(json.dumps(monitor.snapshot(), ensure_ascii=False, indent=2))
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
    print(f"DASHBOARD_LISTEN=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
