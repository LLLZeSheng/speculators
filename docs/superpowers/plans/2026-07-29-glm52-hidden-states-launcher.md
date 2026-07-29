# GLM-5.2 Hidden-State Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and validate a dedicated eight-GPU GLM-5.2 vLLM launcher that collects hidden states from auxiliary layers 8, 23, 39, 55, and 70 plus final layer 78.

**Architecture:** One strict-mode Bash entry point creates the local output directory and invokes the repository's `scripts/launch_vllm.py` with fixed extraction settings. Arguments after `--` reproduce the existing known-working GLM-5.2 vLLM configuration while the Python wrapper injects the hidden-state speculative and file-connector configurations.

**Tech Stack:** Bash, Python 3.13, vLLM 0.26.0, Transformers, speculators hidden-state launcher

## Global Constraints

- Create `/mnt/paas/spec_train/start_glm5.2_hidden_states.sh` without modifying `/mnt/paas/spec_train/start_glm5.2.sh`.
- Load only `/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1`.
- Write hidden states under `/mnt/paas/spec_train/hidden_states/glm5.2`.
- Pass auxiliary target layers `8 23 39 55 70`; allow `launch_vllm.py` to append final layer `78`.
- Use GPUs `0,1,2,3,4,5,6,7`, TP=8, and expert parallelism.
- Preserve all GLM-specific serving arguments from the existing launcher.
- Do not start or stop a live model while verifying the new script.

---

### Task 1: Create and Validate the Hidden-State Launcher

**Files:**
- Create: `/mnt/paas/spec_train/start_glm5.2_hidden_states.sh`
- Reference: `/mnt/paas/spec_train/start_glm5.2.sh`
- Reference: `/mnt/paas/spec_train/speculators/scripts/launch_vllm.py`

**Interfaces:**
- Consumes: Local GLM-5.2 weights, the fixed Python 3.13 runtime, eight CUDA devices, and the speculators hidden-state launcher.
- Produces: An executable Bash command that serves `glm-5.2` on `0.0.0.0:8000` and writes requested hidden states through the file connector.

- [ ] **Step 1: Run the launcher contract check before creation**

Run:

```bash
bash -n /mnt/paas/spec_train/start_glm5.2_hidden_states.sh
```

Expected: FAIL because the new launcher does not exist yet.

- [ ] **Step 2: Create the minimal launcher**

Create `/mnt/paas/spec_train/start_glm5.2_hidden_states.sh` with exactly:

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_PATH="/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1"
readonly HIDDEN_STATES_PATH="/mnt/paas/spec_train/hidden_states/glm5.2"
readonly LAUNCH_VLLM="/mnt/paas/spec_train/speculators/scripts/launch_vllm.py"
readonly PYTHON_BIN="/mnt/pass/miniconda3/bin/python3.13"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/mnt/pass/miniconda3/lib/python3.13/site-packages/nvidia/cu13
export VLLM_LOGGING_LEVEL=INFO

mkdir -p "${HIDDEN_STATES_PATH}"

exec "${PYTHON_BIN}" "${LAUNCH_VLLM}" "${MODEL_PATH}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --target-layer-ids 8 23 39 55 70 \
  -- \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-5.2 \
  --max-model-len 20480 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --moe-backend auto
```

- [ ] **Step 3: Verify Bash syntax and the fixed contract**

Run:

```bash
bash -n /mnt/paas/spec_train/start_glm5.2_hidden_states.sh
rg -n --fixed-strings \
  -e '--target-layer-ids 8 23 39 55 70' \
  -e '/mnt/paas/spec_train/hidden_states/glm5.2' \
  -e '--tensor-parallel-size 8' \
  -e '--attention-backend FLASHINFER_MLA_SPARSE' \
  /mnt/paas/spec_train/start_glm5.2_hidden_states.sh
```

Expected: `bash -n` exits 0 and all four required strings are printed.

- [ ] **Step 4: Dry-run the generated vLLM command without loading the model**

Run:

```bash
/mnt/pass/miniconda3/bin/python3.13 \
  /mnt/paas/spec_train/speculators/scripts/launch_vllm.py \
  /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1 \
  --hidden-states-path /mnt/paas/spec_train/hidden_states/glm5.2 \
  --target-layer-ids 8 23 39 55 70 \
  --dry-run \
  -- \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-5.2 \
  --max-model-len 20480 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --moe-backend auto
```

Expected: exit status 0. The printed `--speculative_config` JSON contains `"eagle_aux_hidden_state_layer_ids": [8, 23, 39, 55, 70, 78]`; the `--kv_transfer_config` JSON contains `"shared_storage_path": "/mnt/paas/spec_train/hidden_states/glm5.2"`; and the command ends with `--no-enable-chunked-prefill`.

- [ ] **Step 5: Verify no unintended files changed**

Run:

```bash
git -C /mnt/paas/spec_train/speculators status --short
sha256sum /mnt/paas/spec_train/start_glm5.2.sh
```

Expected: the repository has no uncommitted implementation changes, because the requested launcher lives one directory above the Git repository. The baseline launcher's checksum is reported and it remains unmodified.

The launcher itself cannot be committed by this repository because its required path is outside `/mnt/paas/spec_train/speculators`.
