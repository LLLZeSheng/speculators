# GLM-5.2 DSpark TensorBoard Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh` run from `speculators_venv` and write every training run's structured metrics to TensorBoard event files.

**Architecture:** Install TensorBoard into the existing Python 3.10 virtual environment, then update the single existing eight-GPU launcher to use that environment and pass `--logger tensorboard`. Verify the launcher contract before and after the edit, then exercise Speculators' production metric-logger path and read the resulting event file back with TensorBoard.

**Tech Stack:** Bash, Python 3.10, uv, PyTorch `SummaryWriter`, TensorBoard event accumulator, Speculators metric logger

## Global Constraints

- Target host is `root@192.168.1.218`.
- Virtual environment is `/mnt/paas/spec_train/speculators_venv`.
- Launcher is `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`.
- Install and enable TensorBoard only; do not install or enable W&B, Trackio, or MLflow.
- Do not start the full eight-GPU training job during validation.
- Preserve `GLM52_PYTHON_BIN` and `GLM52_TORCHRUN_BIN` overrides.
- Preserve all pre-existing tracked and untracked repository changes.
- Use `https://mirrors.aliyun.com/pypi/simple` for Python package downloads.

---

### Task 1: Establish the Failing Launcher Contract

**Files:**
- Test: `/tmp/test_glm52_tensorboard_launcher.sh`
- Read: `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`

**Interfaces:**
- Consumes: The current launcher text.
- Produces: A reproducible shell contract that requires the dedicated Python and torchrun paths plus `--logger tensorboard`.

- [ ] **Step 1: Create the launcher contract test**

```bash
#!/usr/bin/env bash
set -euo pipefail
launcher=/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh
grep -Fq 'GLM52_PYTHON_BIN:-/mnt/paas/spec_train/speculators_venv/bin/python' "$launcher"
grep -Fq 'GLM52_TORCHRUN_BIN:-/mnt/paas/spec_train/speculators_venv/bin/torchrun' "$launcher"
grep -Eq '^[[:space:]]+--logger[[:space:]]+tensorboard$' "$launcher"
```

- [ ] **Step 2: Run the contract test and verify the pre-change failure**

Run: `bash /tmp/test_glm52_tensorboard_launcher.sh`

Expected: non-zero exit because the current launcher points to `/mnt/pass/miniconda3` and contains no `--logger tensorboard` argument.

### Task 2: Install the TensorBoard Runtime

**Files:**
- Modify through package installation only: `/mnt/paas/spec_train/speculators_venv/`

**Interfaces:**
- Consumes: The existing Speculators Python 3.10 virtual environment.
- Produces: Importable `tensorboard` and `torch.utils.tensorboard.SummaryWriter` packages plus the `tensorboard` executable.

- [ ] **Step 1: Install TensorBoard from the verified mirror**

Run: `/root/.local/bin/uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --python /mnt/paas/spec_train/speculators_venv/bin/python tensorboard`

Expected: installation exits zero.

- [ ] **Step 2: Check dependency compatibility and imports**

Run: `/root/.local/bin/uv pip check --python /mnt/paas/spec_train/speculators_venv/bin/python`

Run: `/mnt/paas/spec_train/speculators_venv/bin/python -c 'import tensorboard; from torch.utils.tensorboard import SummaryWriter; print(tensorboard.__version__)'`

Expected: all installed packages are compatible and TensorBoard prints its version.

### Task 3: Integrate TensorBoard into the Existing Launcher

**Files:**
- Modify: `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`
- Test: `/tmp/test_glm52_tensorboard_launcher.sh`

**Interfaces:**
- Consumes: The runtime from Task 2 and the current launcher.
- Produces: A launcher whose defaults use `speculators_venv` and whose `train_cmd` enables TensorBoard.

- [ ] **Step 1: Change only the two runtime defaults and logger argument**

```diff
-readonly PYTHON_BIN="${GLM52_PYTHON_BIN:-/mnt/pass/miniconda3/bin/python3.13}"
-readonly TORCHRUN_BIN="${GLM52_TORCHRUN_BIN:-/mnt/pass/miniconda3/bin/torchrun}"
+readonly PYTHON_BIN="${GLM52_PYTHON_BIN:-/mnt/paas/spec_train/speculators_venv/bin/python}"
+readonly TORCHRUN_BIN="${GLM52_TORCHRUN_BIN:-/mnt/paas/spec_train/speculators_venv/bin/torchrun}"
```

Add this argument beside `--log-dir` and `--run-name`:

```bash
  --logger tensorboard
```

- [ ] **Step 2: Run syntax and contract tests**

Run: `bash -n /mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`

Run: `bash /tmp/test_glm52_tensorboard_launcher.sh`

Expected: both commands exit zero.

- [ ] **Step 3: Confirm the exact diff is limited to three lines**

Run: compare the saved pre-edit SHA-256 and unified diff against the edited launcher.

Expected: only the Python default, torchrun default, and TensorBoard logger argument differ.

### Task 4: Verify Real TensorBoard Event Production

**Files:**
- Create temporarily: `/tmp/speculators_tensorboard_smoke.py`
- Create during validation: `/tmp/speculators_tensorboard_smoke/`
- Read: `/mnt/paas/spec_train/speculators/src/speculators/train/logger.py`

**Interfaces:**
- Consumes: `setup_metric_logger(loggers, run_name, output_dir)` from Speculators.
- Produces: An event file containing scalar tag `smoke/loss` with value `0.25` at step `7`.

- [ ] **Step 1: Write the event smoke program**

```python
import logging
import logging.config
import shutil
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from speculators.train.logger import setup_metric_logger

output_dir = Path("/tmp/speculators_tensorboard_smoke")
shutil.rmtree(output_dir, ignore_errors=True)
setup_metric_logger("tensorboard", "validation", output_dir)
logging.getLogger("speculators.metrics").info(
    {"smoke": {"loss": 0.25}}, extra={"step": 7}
)
logging.shutdown()
event_files = list((output_dir / "validation").glob("events.out.tfevents.*"))
assert event_files, "TensorBoard event file was not created"
events = EventAccumulator(str(output_dir / "validation"))
events.Reload()
scalar = events.Scalars("smoke/loss")
assert len(scalar) == 1
assert scalar[0].step == 7
assert scalar[0].value == 0.25
print(event_files[0])
```

- [ ] **Step 2: Run the smoke program from the editable checkout**

Run: `cd /mnt/paas/spec_train/speculators && /mnt/paas/spec_train/speculators_venv/bin/python /tmp/speculators_tensorboard_smoke.py`

Expected: exit zero and print the generated event-file path.

- [ ] **Step 3: Validate the launcher without starting training**

Run: `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh --dry-run`

Expected: preflight succeeds, printed command contains the venv torchrun plus `--logger tensorboard`, and output ends with `Dry-run complete; torchrun was not started.`

- [ ] **Step 4: Confirm repository state and document how to view metrics**

Run: `git -C /mnt/paas/spec_train/speculators status --short --branch`

Expected: no user source modification is lost or overwritten.

View command:

```bash
source /mnt/paas/spec_train/speculators_venv/bin/activate
tensorboard --logdir /mnt/paas/spec_train/output/dspark_glm52_nuoya_100k_h200/metrics --host 0.0.0.0 --port 6006
```
