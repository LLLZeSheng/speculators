# GLM-5.2 DSpark 5-Layer Non-Causal Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and statically validate an isolated launcher for a 5-layer, non-causal intra-block GLM-5.2 DSpark experiment without starting training.

**Architecture:** Copy the operational structure of the active baseline launcher into a separately named shell script. Change only draft depth and intra-block attention mode, isolate all outputs and TensorBoard naming, and add a preflight refusal when the experiment output directory is non-empty.

**Tech Stack:** Bash, torchrun, Speculators training CLI, TensorBoard.

## Global Constraints

- Do not modify `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`.
- Do not start `torchrun` or create experiment output artifacts during validation.
- Change only `--num-layers 3` to `--num-layers 5` and enable `--sliding-window-non-causal` among model/training parameters.
- Keep the dataset, hidden states, vocabulary mappings, block size 8, sliding window 2048, target layer IDs, loss weights, optimizer, learning rate, scheduler, 10 epochs, and seed 42 unchanged.
- Use output `/mnt/paas/spec_train/output/dspark_glm52_nuoya_hs781890_h200_5l_noncausal`.
- Use TensorBoard run `glm52-dspark-nuoya-hs781890-h200-5l-noncausal`.

---

### Task 1: Create and validate the isolated launcher

**Files:**
- Create: `/mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh`
- Reference: `/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh`

**Interfaces:**
- Consumes: the existing model, dataset, hidden-state files, vocabulary maps, virtual environment, and Speculators checkout used by the baseline launcher.
- Produces: an executable zero-argument Bash launcher with isolated output and TensorBoard paths.

- [ ] **Step 1: Record the baseline checksum and verify the new launcher does not exist**

Run:

```bash
sha256sum /mnt/paas/spec_train/train_glm5.2_dspark_h200.sh
test ! -e /mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh
```

Expected: the baseline checksum is printed and the absence check exits successfully.

- [ ] **Step 2: Create the launcher with the approved two model changes and output guard**

Create a zero-argument Bash script that mirrors the baseline, uses the approved output/run names, sets `--num-layers 5`, adds `--sliding-window-non-causal`, and exits with an explanatory error before `mkdir` or `torchrun` when the experiment output directory already exists and is non-empty.

- [ ] **Step 3: Make the launcher executable and run syntax validation**

Run:

```bash
chmod 755 /mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh
bash -n /mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh
```

Expected: `bash -n` exits with status 0 and emits no output.

- [ ] **Step 4: Run static assertions without launching training**

Run a read-only Python assertion script that verifies:

```python
from pathlib import Path

baseline = Path("/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh").read_text()
experiment = Path(
    "/mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh"
).read_text()

assert '--num-layers 5' in experiment
assert '--num-layers 3' not in experiment
assert '--sliding-window-non-causal' in experiment
assert '--sliding-window 2048' in experiment
assert 'dspark_glm52_nuoya_hs781890_h200_5l_noncausal' in experiment
assert 'glm52-dspark-nuoya-hs781890-h200-5l-noncausal' in experiment
assert 'OUTPUT_PATH' in experiment and 'non-empty' in experiment
assert '--num-layers 3' in baseline
assert '--sliding-window-non-causal' not in baseline
```

Expected: the assertion script exits with status 0.

- [ ] **Step 5: Confirm isolation and baseline integrity**

Run:

```bash
sha256sum /mnt/paas/spec_train/train_glm5.2_dspark_h200.sh
test ! -e /mnt/paas/spec_train/output/dspark_glm52_nuoya_hs781890_h200_5l_noncausal
```

Expected: the checksum matches Step 1 and no experiment output directory exists.
