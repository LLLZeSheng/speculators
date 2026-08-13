# GLM-5.2 Ascend Online MTP3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a tested four-node Ascend launcher that serves GLM-5.2 W4A8 on one 16-NPU node and trains one BF16 native MTP layer recursively for three steps on three 16-NPU nodes.

**Architecture:** Keep verifier inference and draft training as separate processes connected by the existing file hidden-state connector on `/mnt/xds/sfs`. Add one generic training option to distinguish the model ID served by vLLM from the BF16 checkpoint used to initialize MTP weights, then layer an environment-configured Ascend preflight and role-based shell launcher on top.

**Tech Stack:** Bash, Python 3.10+, PyTorch/torch_npu 2.10, Hugging Face Transformers/Datasets, vLLM Ascend, HCCL, pytest, Ruff.

## Global Constraints

- Hardware is four nodes with 16 Ascend 910B3 NPUs per node.
- The verifier checkpoint defaults to `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`.
- The BF16/native-MTP checkpoint defaults to `/mnt/xds/sfs/GLM-5.2`.
- The token dataset defaults to `/mnt/xds/sfs/datasets/glm52-dspark-train` and is overrideable through `DATA_PATH`.
- One shared MTP layer must be recursively applied for exactly three speculative steps.
- Verifier quantization must not alter BF16 MTP parameters or optimizer states.
- Existing CUDA launchers and their defaults must remain behaviorally unchanged.
- The active CUDA checkout `/mnt/paas/spec_train/speculators` must not be modified.
- Actual NPU kernel support is accepted only after preflight and smoke execution on the A3 cluster.

---

### Task 1: Separate MTP Initialization From the Online Service Model ID

**Files:**
- Modify: `src/speculators/train/config/schema.py`
- Modify: `scripts/train.py`
- Modify: `tests/unit/train/test_cli_args.py`
- Modify: `tests/unit/train/test_data.py`

**Interfaces:**
- Consumes: existing `GenerationArgs`, `TrainConfig.flatten()`, and `ArrowDataset(model=...)`.
- Produces: `generation_model_name_or_path: str | None`; `scripts/train.py` passes `generation_model_name_or_path or verifier_name_or_path` to both train and validation datasets.

- [ ] **Step 1: Write failing CLI resolution tests**

Add assertions that `--generation-model-name-or-path glm52-w4a8` resolves to that value and that omission resolves to `None`.

- [ ] **Step 2: Run the focused CLI tests and verify RED**

Run: `pytest -q tests/unit/train/test_cli_args.py -k generation_model`

Expected: FAIL because `GenerationArgs` does not expose the new field.

- [ ] **Step 3: Add the optional generation model field**

Add this field to `GenerationArgs`:

```python
generation_model_name_or_path: str | None = Field(
    default=None,
    description=(
        "Model ID expected from the online vLLM endpoint. Defaults to "
        "--verifier-name-or-path. Set this when verifier inference uses a "
        "quantized checkpoint while draft weights are initialized from BF16."
    ),
)
```

- [ ] **Step 4: Run the focused CLI tests and verify GREEN**

Run: `pytest -q tests/unit/train/test_cli_args.py -k generation_model`

Expected: PASS.

- [ ] **Step 5: Write a failing data plumbing test**

Patch `scripts.train.create_train_val_loaders`, execute the dataloader setup path with a minimal config, and assert its `verifier_name_or_path` keyword is the explicit W4A8 generation ID rather than the BF16 initialization path. Cover the fallback to `verifier_name_or_path` in a second assertion.

- [ ] **Step 6: Run the plumbing test and verify RED**

Run: `pytest -q tests/unit/train/test_data.py -k generation_model`

Expected: FAIL because `scripts/train.py` always passes `args.verifier_name_or_path`.

- [ ] **Step 7: Thread the resolved service model ID to both datasets**

In `scripts/train.py`, resolve:

```python
generation_model = (
    args.generation_model_name_or_path or args.verifier_name_or_path
)
```

Pass `generation_model` as `verifier_name_or_path` to `create_train_val_loaders`; retain the BF16 `args.verifier_name_or_path` for model construction and verifier weight loading.

- [ ] **Step 8: Run focused and configuration regression tests**

Run: `pytest -q tests/unit/train/test_cli_args.py tests/unit/train/test_data.py tests/unit/train/config`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

Stage only the four Task 1 paths and commit with `feat(train): separate online generation model id`.

---

### Task 2: Remove CUDA-Only Lifecycle Calls From the Training Entry Point

**Files:**
- Modify: `scripts/train.py`
- Create: `tests/unit/train/test_accelerator_lifecycle.py`

**Interfaces:**
- Consumes: `torch.manual_seed`, `torch.accelerator.current_accelerator`, and `torch.accelerator.empty_cache`.
- Produces: `set_seed()` that does not call a CUDA-specific seed API, plus `empty_accelerator_cache()` for CUDA and NPU cleanup.

- [ ] **Step 1: Write failing accelerator lifecycle tests**

Test that `set_seed(42)` never calls `torch.cuda.manual_seed_all`, and test that `empty_accelerator_cache()` calls `torch.accelerator.empty_cache()` only when `current_accelerator()` is not `None`.

- [ ] **Step 2: Run the lifecycle tests and verify RED**

Run: `pytest -q tests/unit/train/test_accelerator_lifecycle.py`

Expected: FAIL because `set_seed` calls CUDA directly and `empty_accelerator_cache` does not exist.

- [ ] **Step 3: Implement accelerator-generic lifecycle behavior**

Rely on `torch.manual_seed`, whose contract seeds all devices, remove the redundant CUDA-only seed call, and add:

```python
def empty_accelerator_cache() -> None:
    if torch.accelerator.current_accelerator() is not None:
        torch.accelerator.empty_cache()
```

Use it at the end of `main`. Keep `--deterministic-cuda` CUDA-specific and guard its cuDNN changes with `torch.cuda.is_available()`.

- [ ] **Step 4: Run lifecycle and train-entry regression tests**

Run: `pytest -q tests/unit/train/test_accelerator_lifecycle.py tests/unit/train/test_draft_config_init.py tests/unit/train/test_rope_config.py`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Stage the two Task 2 paths and commit with `fix(train): use accelerator-generic lifecycle calls`.

---

### Task 3: Add the Ascend Preflight Validator

**Files:**
- Create: `scripts/preflight_ascend_mtp.py`
- Create: `tests/unit/scripts/test_preflight_ascend_mtp.py`

**Interfaces:**
- Consumes: two local HF checkpoint directories, one `datasets.load_from_disk` directory, a shared hidden-state directory, optional endpoint URL, and required NPU count.
- Produces: `compare_model_configs()`, `validate_dataset()`, `validate_native_mtp_weights()`, `validate_shared_directory()`, `validate_npu_runtime()`, `validate_endpoint()`, and a zero/nonzero CLI exit status.

- [ ] **Step 1: Write failing config compatibility tests**

Use temporary `config.json` fixtures to assert quantization-only differences pass while mismatched `model_type`, `hidden_size`, `vocab_size`, or `num_hidden_layers` fail with the exact field name.

- [ ] **Step 2: Run config compatibility tests and verify RED**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py -k config`

Expected: FAIL because the preflight module does not exist.

- [ ] **Step 3: Implement direct JSON config comparison**

Read `config.json`, unwrap `text_config` when present, and compare only the structural fields required by GLM MTP. Do not import or instantiate the full model.

- [ ] **Step 4: Run config tests and verify GREEN**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py -k config`

Expected: PASS.

- [ ] **Step 5: Write failing checkpoint, dataset, and shared-path tests**

Create tiny safetensors index fixtures with and without `model.layers.<num_hidden_layers>.` keys, tiny Arrow datasets with valid and missing columns, and writable/non-directory shared paths. Assert actionable failures.

- [ ] **Step 6: Run the new validator tests and verify RED**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py -k 'checkpoint or dataset or shared'`

Expected: FAIL because these validators are missing.

- [ ] **Step 7: Implement metadata-only local validation**

Validate native MTP presence from `model.safetensors.index.json` or safetensors headers without loading tensor contents; validate dataset columns `input_ids`, `loss_mask`, and `seq_len`; create and remove a unique write probe in the shared directory.

- [ ] **Step 8: Write failing runtime and endpoint tests**

Inject fake accelerator/package probes and a local HTTP test server. Assert device type `npu`, sufficient device count, required package presence, and `/health` status handling. Assert `--skip-device-check` bypasses NPU-only checks for development dry runs.

- [ ] **Step 9: Run runtime tests and verify RED**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py -k 'runtime or endpoint'`

Expected: FAIL because runtime checks are missing.

- [ ] **Step 10: Implement runtime, endpoint, and CLI orchestration**

Report detected versions of `torch_npu`, `vllm`, and `vllm-ascend`; require an NPU accelerator and the requested device count unless skipped; query `<endpoint>/health` with a bounded timeout when supplied. Aggregate every successful check and fail fast with a concise `Preflight failed:` message.

- [ ] **Step 11: Run the full preflight suite**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py`

Expected: PASS.

- [ ] **Step 12: Commit Task 3**

Stage the two Task 3 paths and commit with `feat(ascend): add GLM MTP preflight checks`.

---

### Task 4: Add the Four-Node Role-Based Launcher and Operator Guide

**Files:**
- Create: `examples/train/mtp_glm52_ascend_online.sh`
- Create: `tests/unit/scripts/test_ascend_mtp_launcher.py`
- Create: `docs/user_guide/tutorials/train_mtp_ascend_online.md`

**Interfaces:**
- Consumes: `ROLE`, `VERIFIER_HOST`, `MASTER_ADDR`, `NODE_RANK`, model/data/shared paths, and the preflight CLI.
- Produces: `preflight`, `verifier`, `trainer`, and `smoke` roles; deterministic command rendering under `DRY_RUN=1`; separate logs; atomic shared native-MTP initialization.

- [ ] **Step 1: Write failing dry-run launcher tests**

Execute the absent script in subprocesses and assert:

- verifier command uses 16 visible NPUs, TP=16, the W4A8 path, target layer 78, the shared connector path, and served name `glm52-w4a8-verifier`;
- trainer command uses three nodes, 16 processes per node, HCCL rendezvous values, BF16 verifier path, shared converted MTP path, W4A8 generation model ID, MTP3, BF16, FSDP, online generation, and checkpoint interval 1000;
- smoke command caps sequence length, samples, and steps without changing the online path;
- invalid roles and missing trainer topology variables fail before launch.

- [ ] **Step 2: Run launcher tests and verify RED**

Run: `pytest -q tests/unit/scripts/test_ascend_mtp_launcher.py`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement configuration, validation, and command rendering**

Use strict Bash mode and environment defaults from the design. Build commands as arrays, print shell-escaped resolved commands, and execute only when `DRY_RUN!=1`. Set `ASCEND_RT_VISIBLE_DEVICES=0,...,15`, `HCCL_CONNECT_TIMEOUT`, `HCCL_EXEC_TIMEOUT`, and `PYTORCH_NPU_ALLOC_CONF` only when not already set by the operator.

- [ ] **Step 4: Implement verifier and preflight roles**

The verifier role invokes `scripts/launch_vllm.py` with the file connector, target layer 78, TP=16, max length 8192, served model name, host, and port. The preflight role invokes `scripts/preflight_ascend_mtp.py` for local model/data/shared checks; trainer roles additionally check the verifier endpoint.

- [ ] **Step 5: Implement atomic native-MTP initialization**

Before `torchrun`, trainer node rank zero acquires `flock` in the shared initialization parent, runs `python -m speculators convert` from the BF16 checkpoint into a temporary sibling, atomically renames it to `MTP_DRAFT_PATH`, and writes a ready marker. Other trainer nodes wait with `MTP_PREPARE_TIMEOUT` and fail clearly if the marker never appears. `DRY_RUN=1` prints this conversion without mutating storage.

- [ ] **Step 6: Implement trainer and smoke roles**

Launch `torchrun` with `NNODES=3`, `NPROC_PER_NODE=16`, `--fsdp-shard`, `--from-pretrained "$MTP_DRAFT_PATH"`, `--verifier-name-or-path "$MTP_INIT_MODEL_PATH"`, `--generation-model-name-or-path "$SERVED_MODEL_NAME"`, online file hidden states, recursive MTP3, BF16 hidden states, cosine schedule, TensorBoard logging, and global-step checkpoints. Smoke overrides to `MAX_STEPS=2`, `MAX_SAMPLES=64`, `TOTAL_SEQ_LEN=1024`, and a separate output/log suffix.

- [ ] **Step 7: Add signal-safe logging**

Create role-specific log directories, route each long-running command through `tee`, preserve the child exit status, and forward `TERM`/`INT` to the child process. Do not remove checkpoints or datasets during cleanup.

- [ ] **Step 8: Run launcher tests and shell syntax validation**

Run: `pytest -q tests/unit/scripts/test_ascend_mtp_launcher.py`

Run: `bash -n examples/train/mtp_glm52_ascend_online.sh`

Expected: PASS.

- [ ] **Step 9: Write the four-node operator guide**

Document tested package versions, shared paths, role commands for the verifier and trainer node ranks 0/1/2, preflight order, smoke order, full-run order, log paths, checkpoint paths, TensorBoard command, environment overrides, stopping behavior, and the distinction between W4A8 verifier inference and BF16 MTP optimization.

- [ ] **Step 10: Run documentation and example regression checks**

Run: `pytest -q tests/unit/train/config/test_resolution.py tests/unit/scripts/test_ascend_mtp_launcher.py`

Run: `ruff check scripts/preflight_ascend_mtp.py tests/unit/scripts/test_preflight_ascend_mtp.py tests/unit/scripts/test_ascend_mtp_launcher.py`

Run: `ruff format --check scripts/preflight_ascend_mtp.py tests/unit/scripts/test_preflight_ascend_mtp.py tests/unit/scripts/test_ascend_mtp_launcher.py`

Expected: PASS.

- [ ] **Step 11: Commit Task 4**

Stage the three Task 4 paths and commit with `feat(ascend): launch online GLM MTP3 training`.

---

### Task 5: Full Regression, Review, and Publication

**Files:**
- Verify all modified files from Tasks 1-4.

**Interfaces:**
- Consumes: completed feature branch.
- Produces: a clean, reviewed branch pushed to `origin/feat/ascend-online-mtp3`.

- [ ] **Step 1: Run focused feature tests**

Run: `pytest -q tests/unit/scripts/test_preflight_ascend_mtp.py tests/unit/scripts/test_ascend_mtp_launcher.py tests/unit/train/test_accelerator_lifecycle.py tests/unit/train/test_cli_args.py tests/unit/train/test_data.py`

Expected: PASS.

- [ ] **Step 2: Run GLM MTP regression tests**

Run: `pytest -q tests/unit/models/test_mtp_model.py tests/unit/models/test_mtp_attention.py tests/unit/models/test_mtp_config.py tests/unit/models/test_mtp_data.py tests/unit/convert/test_mtp_converter.py tests/unit/convert/test_mtp_stitch.py`

Expected: PASS, with environment-dependent skips allowed and reported.

- [ ] **Step 3: Run repository quality checks on changed Python files**

Run: `ruff check scripts/train.py scripts/preflight_ascend_mtp.py src/speculators/train/config/schema.py tests/unit/scripts/test_preflight_ascend_mtp.py tests/unit/scripts/test_ascend_mtp_launcher.py tests/unit/train/test_accelerator_lifecycle.py tests/unit/train/test_cli_args.py tests/unit/train/test_data.py`

Run: `ruff format --check` on the same Python paths.

Run: `git diff --check 1732ebb..HEAD`

Expected: PASS.

- [ ] **Step 4: Perform code review against the design**

Confirm verifier/trainer model separation, shared-file semantics, MTP recursion count, BF16 optimizer target, no CUDA launcher regression, safe shell quoting, and actionable failure messages. Fix every correctness finding with a failing regression test first.

- [ ] **Step 5: Commit review fixes if required**

Stage only reviewed fix paths and commit with a message describing the actual correction. Skip this commit when review finds no changes.

- [ ] **Step 6: Verify branch status and push**

Confirm the worktree contains no unrelated changes, show the commits relative to `1732ebb`, and push `feat/ascend-online-mtp3` to `origin` as requested.
