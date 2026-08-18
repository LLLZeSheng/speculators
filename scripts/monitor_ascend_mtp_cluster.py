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
    epoch_current = int(epoch_match.group(1)) if epoch_match else None
    epoch_total = int(epoch_match.group(2)) if epoch_match else None
    if epoch_match and epoch_match.group(3) == "completed":
        epoch_current = min(epoch_current + 1, epoch_total)
    step_current = int(step_match.group(1)) if step_match else None
    step_total = None
    if progress_match:
        step_current = int(progress_match.group(1))
        step_total = int(progress_match.group(2))
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
        self.shared_root = Path(str(self.config.get("shared_root", "/kos_ulan")))
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
        active_trainers = sum(node["state"] in {"running", "saving"} for node in trainers)
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
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascend MTP Cluster</title><style>
:root{color-scheme:dark;--bg:#07111f;--panel:#0e1c2f;--line:#203650;--text:#e7f0fb;--muted:#8fa8c2;--ok:#3ddc97;--warn:#ffbd59;--bad:#ff667a;--blue:#5ba8ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#123358 0,transparent 36%),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:26px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-end}h1{margin:0;font-size:27px;letter-spacing:.02em}.sub,.muted{color:var(--muted)}
.badge{display:inline-flex;align-items:center;gap:7px;padding:6px 11px;border:1px solid var(--line);border-radius:999px;background:#0a1728}.dot{width:9px;height:9px;border-radius:50%;background:var(--warn)}
.summary{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:12px;margin:22px 0}.metric,.node{background:linear-gradient(145deg,#10233a,#0b1829);border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 35px #0004}.metric{padding:16px}.metric b{display:block;font-size:23px;margin-top:5px}.section{margin-top:22px}.section h2{font-size:17px;margin:0 0 11px}.grid{display:grid;grid-template-columns:repeat(4,minmax(245px,1fr));gap:12px}.node{padding:15px;min-width:0}.node-head{display:flex;justify-content:space-between;gap:8px}.name{font-weight:750;font-size:16px}.state{font-size:12px;padding:3px 8px;border-radius:999px;background:#ffffff12}.healthy,.running,.complete{color:var(--ok)}.failed{color:var(--bad)}.stale,.waiting,.saving{color:var(--warn)}.loading,.starting{color:var(--blue)}
.kv{height:7px;border-radius:5px;background:#071321;overflow:hidden;margin:10px 0}.kv i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#8b77ff)}.facts{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:12px}.fact{padding:8px;background:#071421;border-radius:8px}.fact span{display:block;color:var(--muted);font-size:11px}.err{margin-top:10px;color:#ff9aaa;background:#3b1420;padding:8px;border-radius:8px;word-break:break-word}.tail{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;color:#a8bdd2;font:11px/1.45 ui-monospace,monospace;max-height:92px;overflow:auto;white-space:pre-wrap}.foot{margin:22px 0;color:var(--muted)}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,1fr)}.summary{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){main{padding:16px}.top{align-items:flex-start;flex-direction:column}.grid,.summary{grid-template-columns:1fr}}
</style></head><body><main><div class="top"><div><h1 id="title">Ascend MTP Cluster</h1><div class="sub" id="stamp">正在读取集群日志…</div></div><div class="badge"><i class="dot" id="dot"></i><span id="overall">连接中</span></div></div><div class="summary" id="summary"></div><section class="section"><h2>Verifier / 隐藏状态生成</h2><div class="grid" id="verifiers"></div></section><section class="section"><h2>Trainer / 分布式训练</h2><div class="grid" id="trainers"></div></section><div class="foot">每 5 秒自动刷新 · 只读日志与健康检查 · <span id="config"></span></div></main>
<script>
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtAge=s=>s==null?'—':s<60?s+' 秒':s<3600?Math.floor(s/60)+' 分':Math.floor(s/3600)+' 小时';
const fact=(k,v)=>`<div class="fact"><span>${esc(k)}</span>${esc(v)}</div>`;
function nodeCard(n){let progress=n.kind==='trainer'&&n.epoch_total?Math.min(100,100*(n.epoch_current||0)/n.epoch_total):(n.kv_cache_percent||0);let facts=n.kind==='verifier'?[fact('Prompt TPS',n.prompt_tps),fact('Generation TPS',n.generation_tps),fact('请求 Running / Waiting',`${n.running_requests??'—'} / ${n.waiting_requests??'—'}`),fact('KV Cache',n.kv_cache_percent==null?'—':n.kv_cache_percent+'%'),fact('尾部成功请求',n.successful_requests_in_tail),fact('HS 分片修复',n.hidden_state_repairs_in_tail)]:[fact('角色',n.role),fact('Epoch',n.epoch_total?`${n.epoch_current??0} / ${n.epoch_total}`:'—'),fact('Step',n.step_total?`${n.step_current??0} / ${n.step_total}`:(n.step_current??'—')),fact('Loss',n.loss??'—'),fact('日志距今',fmtAge(n.log.age_seconds)),fact('警告 / 错误',`${n.recent_warnings} / ${n.recent_errors}`)];return `<article class="node"><div class="node-head"><div><div class="name">${n.kind==='verifier'?'Verifier':'Trainer'} ${n.index}</div><div class="muted">${esc(n.ip)}</div></div><div class="state ${esc(n.state)}">${esc(n.phase)}</div></div><div class="kv"><i style="width:${progress}%"></i></div><div class="facts">${facts.join('')}</div>${n.latest_error?`<div class="err">${esc(n.latest_error)}</div>`:''}<div class="tail">${esc((n.tail||[]).join('\n'))}</div></article>`}
function metric(k,v){return `<div class="metric"><span class="muted">${esc(k)}</span><b>${esc(v)}</b></div>`}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();document.title=d.cluster_name+' · MTP';document.getElementById('title').textContent=d.cluster_name;document.getElementById('stamp').textContent='更新时间 '+new Date(d.generated_at*1000).toLocaleString();document.getElementById('overall').textContent=d.overall;let dot=document.getElementById('dot');dot.style.background=d.overall==='failed'?'var(--bad)':d.overall==='running'||d.overall==='ready'?'var(--ok)':'var(--warn)';document.getElementById('summary').innerHTML=[metric('健康 Verifier',d.summary.healthy_verifiers+' / '+d.summary.total_verifiers),metric('活跃 Trainer',d.summary.active_trainers+' / '+d.summary.total_trainers),metric('异常节点',d.summary.failed_nodes),metric('Prompt TPS',d.summary.total_prompt_tps),metric('Generation TPS',d.summary.total_generation_tps)].join('');document.getElementById('verifiers').innerHTML=d.verifiers.map(nodeCard).join('');document.getElementById('trainers').innerHTML=d.trainers.map(nodeCard).join('');document.getElementById('config').textContent=d.config_path}catch(e){document.getElementById('stamp').textContent='刷新失败: '+e}}refresh();setInterval(refresh,5000);
</script></body></html>'''


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
