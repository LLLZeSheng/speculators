# Train GLM-5.2 MTP3 Online on Eight Ascend 910C Nodes

This recipe uses four 16-NPU nodes as W4A8C8 verifier services and four
16-NPU nodes as BF16 MTP trainers. Trainer `i` sends online hidden-state
requests only to verifier `i`. Training is one 64-rank FSDP job; the same
trainable native MTP layer is recursively unrolled for three speculative
steps.

## 1. Prerequisites

- Eight Ascend 910C (Atlas A3) nodes, 16 NPUs per node
- Docker image `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`
- `/kos_ulan` mounted on all eight nodes
- this repository available at the same path on all nodes, preferably
  `/kos_ulan/spec_train/speculators`
- W4A8C8 verifier at
  `/mnt/xds/dev/s00838505/GLM-5.2-w4a8c8` on all four verifier nodes
- an unquantized BF16 GLM-5.2 checkpoint with its native MTP layer on
  `/kos_ulan`
- a Hugging Face dataset containing `input_ids`, `loss_mask`, and `seq_len`

The W4A8C8 model is inference-only. MTP initialization, parameters, forward
calculation, gradients, and optimizer state remain BF16. Therefore
`MTP_INIT_MODEL_PATH` must never point to the W4A8C8 checkpoint.

The expected image versions are `torch_npu==2.10.0.post2`, `vllm==0.23.0`,
and `vllm-ascend==0.23.0rc1`. Preflight checks these before model loading.

## 2. Fill one shared configuration file

Create the configuration once on shared storage:

```bash
mkdir -p /kos_ulan/spec_train/config /kos_ulan/spec_train/logs
cp examples/train/mtp_glm52_ascend_online_4v4t.env.example \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
vim /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

At minimum, replace these values:

- `FILL_VERIFIER_0_IP` through `FILL_VERIFIER_3_IP`
- `FILL_TRAINER_0_IP` through `FILL_TRAINER_3_IP`
- `FILL_HCCL_NIC_NAME`, such as `eth0` or `bond0`
- `FILL_BF16_GLM52_PATH`
- `FILL_HF_DATASET_PATH`
- `FILL_UNIQUE_SMOKE_ID`

The addresses must be routable between nodes and appear in `hostname -I` on
their respective nodes. The wrapper uses them to infer the local role and
rank. If a host has ambiguous addresses, provide its intended address only for
that invocation with `NODE_IP=...`.

The remaining paths and training knobs have usable defaults in the example.
Do not put passwords or registry credentials into this file.

## 3. Inspect the resolved commands

On any node, dry-run without loading a model:

```bash
cd /kos_ulan/spec_train/speculators
DRY_RUN=1 bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

Confirm the printed role, rank, local IP, verifier endpoint, model paths,
Docker image, and all 16 `/dev/davinci*` devices. The wrapper mounts the shared
filesystem, this checkout, the Ascend devices, and host driver files. It runs
`pip install --no-deps -e`, so it does not replace the image's PyTorch or vLLM
stack.

## 4. Start four verifiers

Run the same command on verifier nodes 0 through 3:

```bash
cd /kos_ulan/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/verifier-container-$(hostname).log 2>&1 &
```

Each verifier uses DP2 x TP8, expert parallelism, Ascend quantization, DSA-CP,
sparse C8 layer-index acceleration, and all 16 NPUs. Its application log is:

```text
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
```

Check all four configured verifier IPs before training:

```bash
curl -f http://VERIFIER_IP:8077/health
```

The four services share a port because they run on different hosts. Metadata
publication to `/kos_ulan/spec_train/metadata/glm52-w4a8c8` is protected by a
shared lock and an atomic ready marker.

## 5. Run the four-trainer smoke test

The configuration defaults to `TRAINER_MODE=smoke`. Once every verifier is
healthy, run the same command close together on trainer nodes 0 through 3:

```bash
cd /kos_ulan/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/smoke-container-$(hostname).log 2>&1 &
```

The wrapper maps trainer `i` to rank `i`, rendezvous address trainer 0, and
verifier `i`. Smoke mode selects 64 samples, caps context at 1024 tokens, and
runs two optimizer steps through the real online hidden-state path. Success
requires all 64 ranks to finish without HCCL errors, OOM, unsupported
operators, non-finite loss, generation failures, or checkpoint errors.

Choose a new `SMOKE_RUN_ID` before repeating a smoke test; existing smoke
output is deliberately not overwritten.

## 6. Start full training

After smoke succeeds, start the same command on all four trainer nodes with an
environment override:

```bash
cd /kos_ulan/spec_train/speculators
TRAINER_MODE=trainer nohup \
  bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/trainer-container-$(hostname).log 2>&1 &
```

Start all four commands close together. Defaults are 4096 tokens per NPU,
five epochs, AdamW, cosine decay with 3% warmup, and one checkpoint per 1000
global optimizer steps. Increase `TOTAL_SEQ_LEN` only after a representative
smoke test proves memory headroom. `VERIFIER_MAX_MODEL_LEN` must exceed
`TOTAL_SEQ_LEN` by at least one token.

Trainer logs:

```text
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node3.log
```

Checkpoints:

```text
/kos_ulan/spec_train/checkpoints/glm52-w4a8c8-mtp3
```

TensorBoard:

```bash
tensorboard \
  --logdir /kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/metrics \
  --host 0.0.0.0 --port 6006
```

## 7. Stop and resume

Stop trainer containers on all four trainer nodes together. A forwarded
`SIGTERM` lets the training process create an `interrupted` recovery snapshot.
Normal startup resumes the latest numbered checkpoint. An interrupted snapshot
should be inspected and explicitly renamed to a new numeric checkpoint only if
it must be used; data position is restored only at an epoch boundary.

If a shared MTP initialization, metadata, or smoke directory exists without
its `.ready` marker, first confirm that no node is producing it. Move the
incomplete directory aside rather than deleting it blindly, then rerun.
