# GLM MoE DSA MTP Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and stitch GLM-5.2's native MTP layer with the existing Speculators MTP pipeline.

**Architecture:** Extend the generic MTP converter with a config-aware GLM native layout and add Transformers GLM components to the MTP model registry. Reuse the existing recursive MTP objective and make stitch mapping verifier-aware.

**Tech Stack:** Python 3.10+, PyTorch, Transformers 5.14, safetensors, pytest.

## Global Constraints

- Work only in `/mnt/paas/speculators-glm-moe-dsa-mtp` on `feat/glm-moe-dsa-mtp`.
- Preserve current Qwen MTP behavior.
- Unit tests must not load the 19G production MTP shard or use GPUs.

---

### Task 1: GLM model components

**Files:**
- Modify: `src/speculators/models/base_components.py`
- Modify: `src/speculators/models/mtp/model_definitions.py`
- Test: `tests/unit/models/test_mtp_config.py`
- Test: `tests/unit/models/test_mtp_model.py`

**Interfaces:**
- Produces: `mtp_model_classes["glm_moe_dsa"]` using a GLM-specific MTP decoder wrapper.

- [ ] Write a failing registration test using a small `GlmMoeDsaConfig`.
- [ ] Run the focused test and confirm `glm_moe_dsa` is unsupported.
- [ ] Register GLM decoder, norm, rotary, and MTP wrapper components.
- [ ] Run GLM and existing Qwen MTP model tests.
- [ ] Commit the model component change.

### Task 2: Native GLM checkpoint conversion

**Files:**
- Modify: `src/speculators/convert/mtp/converter.py`
- Modify: `src/speculators/convert/mtp/__init__.py`
- Test: `tests/unit/convert/test_mtp_converter.py`

**Interfaces:**
- Produces: config-aware native MTP prefix resolution and `remap_mtp_key_to_native(key, model_type, num_hidden_layers)`.

- [ ] Write failing tests for GLM prefix detection and forward/reverse key mappings.
- [ ] Run the focused tests and confirm the GLM layout is rejected.
- [ ] Implement GLM extra-layer extraction while preserving `mtp.*` handling.
- [ ] Run converter tests, including MoE expert fusion cases.
- [ ] Commit converter support.

### Task 3: GLM-aware stitching

**Files:**
- Modify: `scripts/stitch_mtp.py`
- Create: `tests/unit/convert/test_mtp_stitch.py`

**Interfaces:**
- Consumes: `remap_mtp_key_to_native` from Task 2.
- Produces: verifier-aware native key selection before shard replacement.

- [ ] Write failing stitch mapping and small sharded-checkpoint tests.
- [ ] Run the focused tests and confirm Qwen-only mapping is used.
- [ ] Load verifier config and select GLM reverse mappings during stitch.
- [ ] Run stitch and converter unit tests.
- [ ] Commit stitching support.

### Task 4: Documentation and real-checkpoint smoke test

**Files:**
- Modify: `docs/user_guide/algorithms/mtp.md`
- Modify: `examples/train/mtp_qwen3_5_9b_gsm8k_online.sh` only if generic comments need correction.

**Interfaces:**
- Consumes: complete GLM conversion and stitch pipeline.

- [ ] Document GLM's extra-layer layout and `--num-speculative-steps 3` usage.
- [ ] Run converter extraction against the local GLM-5.2 verifier without saving tensors twice.
- [ ] Run focused pytest, Ruff, and the full non-network MTP unit suite.
- [ ] Review the diff for changes outside the isolated clone.
- [ ] Commit documentation and verification updates.
