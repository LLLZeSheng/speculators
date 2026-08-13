# GLM-5.2 Ascend Online MTP3 Training Design

## Goal

Add an isolated, reproducible launcher for continuing GLM-5.2 native MTP
training on four Ascend 910B3 nodes. One 16-NPU node serves the W4A8 verifier
and three 16-NPU nodes train one BF16 MTP layer with three recursive prediction
steps. The implementation must not change the existing CUDA launch workflow.

## Fixed Environment

- Hardware: four nodes, 16 Ascend 910B3 NPUs per node.
- Verifier checkpoint: `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`.
- BF16/native-MTP checkpoint: `/mnt/xds/sfs/GLM-5.2`.
- `torch_npu`: `2.10.0.post2`.
- `vllm`: `0.23.1rc1.dev1451+gd02df748b.d20260811`.
- `vllm-ascend`: `0.23.0rc1`.
- Shared filesystem: `/mnt/xds/sfs`, mounted at the same path on every node.
- Default token dataset: `/mnt/xds/sfs/datasets/glm52-dspark-train`.

The default dataset path is intentionally configurable because its final path
has not been supplied. `DATA_PATH` overrides it without editing the launcher.

## Chosen Architecture

The launcher is role based. It is invoked independently on each node with one
of four roles:

1. `preflight` validates checkpoints, tokenizer compatibility, dataset schema,
   shared storage, package versions, and visible NPU counts.
2. `verifier` starts a tensor-parallel vLLM Ascend service on one 16-NPU node.
3. `trainer` starts one rank group on each of the three training nodes using
   `torchrun`, HCCL, and 16 processes per node.
4. `smoke` applies conservative sample, sequence-length, and step limits while
   retaining the same verifier-to-trainer data path as the full run.

The roles are explicit rather than coordinated through SSH. This avoids making
assumptions about node hostnames, login policy, or the cluster scheduler. The
operator supplies `MASTER_ADDR`, `NODE_RANK`, and `VERIFIER_HOST`; all remaining
parameters have documented defaults.

## Model Semantics

The verifier and trainable model have separate paths:

- vLLM loads the W4A8 checkpoint and produces the hidden states used as online
  conditioning targets.
- The trainer initializes the native trainable MTP layer from the BF16
  GLM-5.2 checkpoint. This avoids attempting to optimize packed W4A8 tensors.
- The implementation validates that the BF16 and W4A8 checkpoints use the same
  tokenizer, vocabulary, hidden size, and GLM-MoE-DSA architecture.
- `num_speculative_steps=3` means one shared MTP layer is called recursively
  three times. It does not instantiate or optimize three independent layers.

The continuation target remains BF16. Quantization is confined to verifier
inference; optimizer states and trainable MTP weights are not quantized.

## Online Hidden-State Flow

1. A trainer worker reads tokenized examples containing `input_ids`,
   `loss_mask`, and `seq_len`. Any cached hidden-state columns are ignored.
2. On a missing hidden state, the worker calls the W4A8 verifier's
   OpenAI-compatible endpoint.
3. vLLM `extract_hidden_states` writes a safetensors payload through the file
   connector into
   `/mnt/xds/sfs/spec_train/online_hidden_states/glm52-w4a8`.
4. The trainer reads the payload, validates returned token IDs and tensor
   shapes, performs the BF16 MTP3 step, and deletes the transient payload.

Deleting consumed payloads bounds shared-disk usage. A configurable cache mode
will remain available for debugging or repeated short runs.

## Launcher Interface

The primary example script will expose configuration through environment
variables. Required per-node values are:

- `ROLE=verifier|trainer|preflight|smoke`
- `VERIFIER_HOST=<verifier node address>` for trainer roles
- `MASTER_ADDR=<rank-zero training node address>` for trainer roles
- `NODE_RANK=0|1|2` for trainer roles

Important defaults are:

- `VERIFIER_MODEL_PATH=/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`
- `MTP_INIT_MODEL_PATH=/mnt/xds/sfs/GLM-5.2`
- `DATA_PATH=/mnt/xds/sfs/datasets/glm52-dspark-train`
- `HIDDEN_STATES_PATH=/mnt/xds/sfs/spec_train/online_hidden_states/glm52-w4a8`
- `OUTPUT_PATH=/mnt/xds/sfs/spec_train/checkpoints/glm52-w4a8-mtp3`
- `LOG_ROOT=/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3`
- `NNODES=3`, `NPROC_PER_NODE=16`, and verifier tensor parallel size `16`

The script prints the resolved configuration before launch. `DRY_RUN=1`
prints commands and performs non-NPU validation without starting processes.

## Compatibility Changes

Existing accelerator-generic paths will be retained. Any implementation change
outside the new launcher must be limited to verified CUDA assumptions that
prevent NPU execution, such as explicit CUDA seed/device calls. Such changes
will use the current accelerator API and preserve CUDA behavior.

The existing `scripts/launch_vllm.py` hidden-state configuration will be reused
where possible. It may gain an explicit shared-storage-path option so the file
connector is usable across nodes. Existing callers must keep their current
default behavior.

## Failure Handling

Preflight fails before allocation-heavy startup when any of these conditions is
detected:

- checkpoint or dataset paths are missing;
- checkpoint tokenizers or structural config fields differ;
- the dataset lacks required token columns;
- the shared hidden-state directory is not writable;
- visible NPU count is lower than the configured process or tensor-parallel
  count;
- the verifier endpoint is unreachable from a trainer node;
- native MTP critical tensors cannot be loaded from the BF16 checkpoint.

Runtime processes write separate verifier and per-node trainer logs. Shell
signal handlers terminate child processes. No launcher removes checkpoints or
dataset files; cleanup is restricted to transient connector payloads owned by
the current run.

## Verification

Automated verification will cover:

- shell syntax and formatting;
- dry-run command construction for all roles;
- environment-variable validation and defaults;
- tokenizer/config compatibility checks using small temporary fixtures;
- preservation of existing CUDA launcher behavior;
- focused existing GLM MTP conversion, model, and stitching tests.

The current development host is not an Ascend cluster. Therefore, passing local
tests proves command and validation correctness, not kernel-level NPU support.
Before full training, the operator must run `preflight`, start the verifier,
and complete the bounded `smoke` role on the actual four-node environment.

## Deliverables

- A role-based Ascend GLM-5.2 online MTP3 launch example.
- A reusable preflight validator or equivalent testable helper.
- Focused NPU compatibility fixes only where required.
- Operator documentation with exact four-node command examples.
- Tests and dry-run verification.

All work stays on `feat/ascend-online-mtp3`; the active CUDA training checkout
and branch remain untouched.
