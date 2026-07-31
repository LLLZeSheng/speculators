# GLM-5.2 Hidden-State Collection Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, manually invoked shell command that collects 10 GLM-5.2 hidden-state samples by default and accepts a positive integer override.

**Architecture:** A single strict-mode Bash wrapper validates its argument, verifies that the OpenAI-compatible endpoint serves the exact model ID `glm-5.2`, prints its fixed configuration, and delegates collection to the existing offline generator. The wrapper does not manage the model service or delete existing outputs; the generator's existing-index logic provides resume behavior.

**Tech Stack:** Bash, curl, Python 3.13, vLLM OpenAI-compatible API, Speculators offline data generator

## Global Constraints

- Create `/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh` only; do not modify either existing GLM launcher.
- Default maximum sample count is 10; accept one positive integer positional override.
- Require exact served model ID `glm-5.2` at `http://127.0.0.1:8000/v1` before collection.
- Read `/mnt/paas/spec_train/hf_dataset_v2_glm52_8192` and write `/mnt/paas/spec_train/hidden_states/glm5.2`.
- Use concurrency 2, request timeout 600 seconds, one retry, output validation, and fail-fast behavior.
- Preserve existing `hs_<index>.safetensors` files and rely on generator resume semantics.
- Do not start, stop, or replace any model service.

---

### Task 1: Manual Hidden-State Collection Wrapper

**Files:**
- Create: `/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh`
- Reference: `/mnt/paas/spec_train/start_glm5.2_hidden_states.sh`
- Reference: `/mnt/paas/spec_train/speculators/scripts/data_generation_offline.py`

**Interfaces:**
- Consumes: zero or one positional argument containing a positive integer maximum sample count; an already-running OpenAI-compatible service at `http://127.0.0.1:8000/v1`.
- Produces: resumable `hs_<index>.safetensors` files under `/mnt/paas/spec_train/hidden_states/glm5.2`; exit 0 on successful generator completion and nonzero on validation, endpoint, identity, or generator failure.

- [ ] **Step 1: Run failing existence and interface checks**

Run:

```bash
test -x /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh 0
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh abc
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh 10 extra
```

Expected: the first command fails because the script does not exist. The remaining commands cannot execute for the same reason, establishing the red state.

- [ ] **Step 2: Implement the strict-mode wrapper**

Create an executable Bash script with these behaviors:

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly ENDPOINT="http://127.0.0.1:8000/v1"
readonly EXPECTED_MODEL="glm-5.2"
readonly DATASET_PATH="/mnt/paas/spec_train/hf_dataset_v2_glm52_8192"
readonly OUTPUT_PATH="/mnt/paas/spec_train/hidden_states/glm5.2"
readonly GENERATOR="/mnt/paas/spec_train/speculators/scripts/data_generation_offline.py"
readonly PYTHON_BIN="/mnt/pass/miniconda3/bin/python3.13"
readonly SPECULATORS_SRC="/mnt/paas/spec_train/speculators/src"
readonly CONNECTORS_SRC="/mnt/paas/spec_train/speculators/hs_connectors/src"

if (( $# > 1 )); then
  echo "Usage: $0 [MAX_SAMPLES]" >&2
  exit 2
fi

readonly MAX_SAMPLES="${1:-10}"
if [[ ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer: ${MAX_SAMPLES}" >&2
  exit 2
fi

if ! models_json="$(curl --silent --show-error --fail --max-time 5 "${ENDPOINT}/models")"; then
  echo "GLM-5.2 service is unavailable at ${ENDPOINT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c \
  'import json, sys; data=json.loads(sys.argv[1]); raise SystemExit(0 if any(item.get("id") == sys.argv[2] for item in data.get("data", [])) else 1)' \
  "${models_json}" "${EXPECTED_MODEL}"; then
  echo "Endpoint ${ENDPOINT} is not serving ${EXPECTED_MODEL}" >&2
  exit 1
fi

echo "Collecting GLM-5.2 hidden states"
echo "  endpoint: ${ENDPOINT}"
echo "  dataset: ${DATASET_PATH}"
echo "  output: ${OUTPUT_PATH}"
echo "  max samples: ${MAX_SAMPLES}"
echo "Existing hs_<index>.safetensors files are retained and skipped."

exec env \
  PYTHONPATH="${SPECULATORS_SRC}:${CONNECTORS_SRC}" \
  "${PYTHON_BIN}" "${GENERATOR}" \
  --model "${EXPECTED_MODEL}" \
  --endpoint "${ENDPOINT}" \
  --preprocessed-data "${DATASET_PATH}" \
  --output "${OUTPUT_PATH}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency 2 \
  --validate-outputs \
  --request-timeout 600 \
  --max-retries 1 \
  --fail-on-error
```

Set executable mode:

```bash
chmod 0755 /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
```

- [ ] **Step 3: Verify syntax and invalid arguments**

Run:

```bash
bash -n /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
test -x /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh 0
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh abc
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh 10 extra
```

Expected: `bash -n` exits 0; the script is executable; each invalid invocation exits 2 and does not contact the endpoint.

- [ ] **Step 4: Verify fixed collection arguments statically**

Run:

```bash
rg -n --fixed-strings \
  -e '--model "${EXPECTED_MODEL}"' \
  -e '--concurrency 2' \
  -e '--validate-outputs' \
  -e '--request-timeout 600' \
  -e '--max-retries 1' \
  -e '--fail-on-error' \
  /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
```

Expected: all six required generator arguments are printed.

- [ ] **Step 5: Verify unavailable-service protection without changing outputs**

Record the existing output file names, sizes, and mtimes; run the default command while port 8000 is unavailable; then compare the snapshot again.

```bash
find /mnt/paas/spec_train/hidden_states/glm5.2 -maxdepth 1 -type f \
  -printf '%f %s %T@\n' | sort > /tmp/glm52-hs-before.txt
/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
find /mnt/paas/spec_train/hidden_states/glm5.2 -maxdepth 1 -type f \
  -printf '%f %s %T@\n' | sort > /tmp/glm52-hs-after.txt
cmp /tmp/glm52-hs-before.txt /tmp/glm52-hs-after.txt
```

Expected: the wrapper exits 1 with an unavailable-service error; `cmp` exits 0, proving no hidden-state output changed.

- [ ] **Step 6: Final review**

Run:

```bash
sed -n '1,240p' /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
bash -n /mnt/paas/spec_train/collect_glm5.2_hidden_states.sh
```

Expected: the entire script matches the approved design and syntax validation exits 0.
