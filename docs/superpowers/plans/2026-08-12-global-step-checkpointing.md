# Global-Step Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the GLM-5.2 MTP3 training checkpoint every 1000 global optimizer steps.

**Architecture:** Add an optional step interval to the CLI schema and trainer configuration. Centralize the periodic-save predicate so exact step scheduling takes precedence while the existing epoch/fraction behavior remains unchanged when the option is absent.

**Tech Stack:** Python, Pydantic, PyTorch trainer, pytest, Bash.

## Global Constraints

- Use successful global optimizer steps, including a restored `global_step`, as the interval clock.
- Preserve existing `checkpoint_freq` behavior when `checkpoint_steps` is unset.
- Configure the production launcher with exactly 1000 steps.

---

### Task 1: Define scheduling behavior

**Files:**
- Modify: `tests/unit/train/test_checkpoint.py`
- Modify: `tests/unit/train/test_mid_epoch_resume.py`

- [ ] Add tests for steps 999, 1000, 1001, and a resumed epoch whose local and global steps differ.
- [ ] Run the focused tests and verify they fail because `checkpoint_steps` is not supported.

### Task 2: Implement trainer and CLI support

**Files:**
- Modify: `src/speculators/train/trainer.py`
- Modify: `src/speculators/train/config/schema.py`
- Modify: `scripts/train.py`

- [ ] Add the validated CLI field and propagate it to `TrainerConfig`.
- [ ] Implement the exact global-step predicate with precedence over `checkpoint_freq`.
- [ ] Run focused tests and verify they pass.

### Task 3: Configure production and restart services

**Files:**
- Modify: `/mnt/paas/spec_train/train_glm52_mtp3_8k.sh`
- Modify: `/mnt/paas/spec_train/tests/test_train_glm52_mtp3_8k.py`

- [ ] Add a failing launcher contract test for `--checkpoint-steps 1000`.
- [ ] Update the launcher and pass its tests and shell syntax check.
- [ ] Commit both repositories, stop the current run, and resume from the latest complete checkpoint.
- [ ] Restart TensorBoard on port 6006 and verify HTTP availability and training metrics.
