# GLM-5.2 8-GPU Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the current DeepSeek-V4-Flash deployment and bring up the baseline GLM-5.2 NVFP4 weights as a healthy OpenAI-compatible vLLM service on all eight GPUs.

**Architecture:** A clean supervisor shutdown releases the eight GPU allocations and port 8000. One detached vLLM 0.26.0 parent then launches eight tensor-parallel ranks with expert parallelism, logs to `/tmp/glm52-vllm.log`, and serves `glm-5.2` on port 8000.

**Tech Stack:** Bash, Supervisor, vLLM 0.26.0, PyTorch 2.11.0+cu130, compressed-tensors 0.17.0, CUDA/NCCL, curl

## Global Constraints

- Load only `/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1`.
- Use visible GPUs `0,1,2,3,4,5,6,7` with TP=8 and expert parallelism.
- Do not supply a speculative-decoding configuration.
- Bind the OpenAI-compatible API to `0.0.0.0:8000` with served name `glm-5.2`.
- Start with maximum model length 4096 and FP8 KV cache.
- Do not automatically restart DeepSeek if GLM startup fails.

---

### Task 1: Validate the Exact Runtime and Targets

**Files:**
- Read: `/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1/config.json`
- Read: `/mnt/paas/h00947744/DeepSeek-V4-Flash-DSpark/deploy/supervisord.conf`

**Interfaces:**
- Consumes: Running host process table, model directory, vLLM installation, GPU inventory
- Produces: A go/no-go preflight result with no external state changes

- [ ] **Step 1: Confirm all required model artifacts exist**

Run:

```bash
test -s /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1/config.json
test -s /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1/model.safetensors.index.json
test "$(find /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1 -maxdepth 1 -name 'model-*-of-00011.safetensors' | wc -l)" -eq 11
```

Expected: exit status 0 for all three checks.

- [ ] **Step 2: Confirm local runtime versions and GLM registration**

Run:

```bash
/mnt/pass/miniconda3/bin/python3.13 -c 'from importlib.metadata import version; print(version("vllm"), version("compressed-tensors"), version("transformers"))'
rg -n '"GlmMoeDsaForCausalLM"' /mnt/pass/miniconda3/lib/python3.13/site-packages/vllm/model_executor/models/registry.py
```

Expected: versions begin with `0.26.0 0.17.0`, and the registry contains `GlmMoeDsaForCausalLM`.

- [ ] **Step 3: Identify the expected DeepSeek supervisor and GPU consumers**

Run:

```bash
ps -eo pid,ppid,lstart,args | rg '/mnt/paas/h00947744/DeepSeek-V4-Flash-DSpark/deploy/supervisord.conf|/mnt/paas/DeepSeek-V4-Flash'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Expected: one supervisor references the exact expected configuration, its workers reference `/mnt/paas/DeepSeek-V4-Flash`, and all eight GPUs show their allocations.

### Task 2: Stop DeepSeek Cleanly and Verify Resource Release

**Files:**
- Read: `/mnt/paas/h00947744/DeepSeek-V4-Flash-DSpark/deploy/supervisord.conf`

**Interfaces:**
- Consumes: Validated supervisor from Task 1
- Produces: Eight idle GPUs and released ports 8000, 8100-8103, and 8200-8203

- [ ] **Step 1: Request a clean Supervisor shutdown**

Run:

```bash
/mnt/pass/miniconda3/bin/supervisorctl -c /mnt/paas/h00947744/DeepSeek-V4-Flash-DSpark/deploy/supervisord.conf shutdown
```

Expected: `Shut down` or an equivalent successful supervisor response.

- [ ] **Step 2: Poll until DeepSeek workers and router exit**

Run every 10 seconds for up to 180 seconds:

```bash
ps -eo pid,ppid,args | rg '/mnt/paas/DeepSeek-V4-Flash|xpyd_proxy.py' || true
```

Expected: no matching model worker, vLLM engine, or proxy process remains.

- [ ] **Step 3: Verify ports and GPU memory are released**

Run:

```bash
ss -H -lntp | rg ':(8000|810[0-3]|820[0-3])\b' || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Expected: no listed service port is listening, and GPU memory usage drops from the DeepSeek allocation on all eight devices.

### Task 3: Launch the 8-GPU GLM-5.2 Service

**Files:**
- Create: `/tmp/glm52-launch.sh`
- Create at runtime: `/tmp/glm52-vllm.log`
- Create at runtime: `/tmp/glm52-vllm.pid`

**Interfaces:**
- Consumes: Eight released GPUs and port 8000 from Task 2
- Produces: A detached vLLM process loading `glm-5.2`

- [ ] **Step 1: Create the exact launch script**

Create `/tmp/glm52-launch.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO

exec /mnt/pass/miniconda3/bin/vllm serve \
  /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --moe-backend flashinfer_cutlass \
  --kv-cache-dtype fp8 \
  --safetensors-load-strategy prefetch \
  --max-model-len 4096 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

- [ ] **Step 2: Syntax-check the script and confirm DSpark is absent**

Run:

```bash
bash -n /tmp/glm52-launch.sh
! rg -n 'speculative|dspark|spec-model' /tmp/glm52-launch.sh
```

Expected: both commands exit with status 0.

- [ ] **Step 3: Launch detached and record the parent PID**

Run:

```bash
chmod 0755 /tmp/glm52-launch.sh
nohup setsid /tmp/glm52-launch.sh > /tmp/glm52-vllm.log 2>&1 < /dev/null &
echo $! > /tmp/glm52-vllm.pid
```

Expected: `/tmp/glm52-vllm.pid` contains a live PID whose command line references `/tmp/glm52-launch.sh` or the target vLLM model path.

- [ ] **Step 4: Confirm all eight ranks begin loading**

Run:

```bash
ps -eo pid,ppid,lstart,args | rg 'GLM-5.2-NVFP4-W4A4-MG39-BNT3|VLLM::EngineCore|VLLM::Worker'
tail -n 120 /tmp/glm52-vllm.log
```

Expected: the parent stays alive, worker initialization appears, and the log contains no fatal argument or architecture error.

### Task 4: Monitor Startup and Verify the API

**Files:**
- Read: `/tmp/glm52-vllm.log`
- Read: `/tmp/glm52-vllm.pid`

**Interfaces:**
- Consumes: Detached GLM vLLM process from Task 3
- Produces: Evidence that the GLM OpenAI API is healthy and generates text

- [ ] **Step 1: Monitor process, logs, and GPU allocation until ready**

Poll at intervals of at most 30 seconds:

```bash
test -d "/proc/$(cat /tmp/glm52-vllm.pid)"
tail -n 120 /tmp/glm52-vllm.log
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
curl --silent --show-error --fail http://127.0.0.1:8000/health
```

Expected: the process remains live while logs advance; eventually the health request exits 0.

- [ ] **Step 2: Verify the advertised model name**

Run:

```bash
curl --silent --show-error --fail http://127.0.0.1:8000/v1/models
```

Expected: JSON contains `"id":"glm-5.2"` or the whitespace-equivalent representation.

- [ ] **Step 3: Verify one minimal chat completion**

Run:

```bash
curl --silent --show-error --fail \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0}' \
  http://127.0.0.1:8000/v1/chat/completions
```

Expected: valid JSON contains a non-empty assistant message and no `error` object.

- [ ] **Step 4: Capture final process and GPU evidence**

Run:

```bash
ps -eo pid,ppid,lstart,args | rg 'GLM-5.2-NVFP4-W4A4-MG39-BNT3|VLLM::EngineCore|VLLM::Worker'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Expected: GLM ranks occupy all eight GPUs and no DeepSeek worker appears.
