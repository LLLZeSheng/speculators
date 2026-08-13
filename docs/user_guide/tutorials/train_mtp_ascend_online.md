# Train GLM-5.2 MTP3 Online on Ascend

This recipe continues the native GLM-5.2 MTP layer on four Ascend 910B3
nodes. One 16-NPU node serves the W4A8 verifier. Three 16-NPU nodes train one
BF16 MTP layer with FSDP and HCCL. The same trainable layer is recursively
unrolled for three speculative steps; the recipe does not create three MTP
layers.

## Environment

The launcher targets this validated software and storage layout:

- 16 Ascend 910B3 NPUs per node, four nodes total
- `torch_npu==2.10.0.post2`
- `vllm==0.23.1rc1.dev1451+gd02df748b.d20260811`
- `vllm-ascend==0.23.0rc1`
- W4A8 verifier: `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`
- BF16 native MTP source: `/mnt/xds/sfs/GLM-5.2`
- shared filesystem mounted as `/mnt/xds/sfs` on every node

The verifier checkpoint is used only for online hidden-state inference. MTP
parameters, forward computation, gradients, and optimizer state remain BF16.

Install this branch in the training environment on all nodes. Source the
Ascend toolkit environment before invoking the launcher. Keep the vLLM Ascend
environment on the verifier node compatible with the versions above.

## Configuration

The entry point is:

```bash
examples/train/mtp_glm52_ascend_online.sh
```

It is configured with environment variables. These defaults can be overridden
without editing the script:

```bash
export DATA_PATH=/mnt/xds/sfs/datasets/glm52-dspark-train
export HIDDEN_STATES_PATH=/mnt/xds/sfs/spec_train/online_hidden_states/glm52-w4a8
export MTP_DRAFT_PATH=/mnt/xds/sfs/spec_train/initial/glm52-bf16-mtp3
export OUTPUT_PATH=/mnt/xds/sfs/spec_train/checkpoints/glm52-w4a8-mtp3
export LOG_ROOT=/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3
```

`DATA_PATH` may point at the existing DSpark token dataset. It must contain
`input_ids`, `loss_mask`, and `seq_len`. Cached hidden states from another
verifier are not read. The launcher uses `--force-generate` to bypass cached
files, asks the W4A8 service for every sample, and removes each transient
payload after the trainer consumes it. Online generation failures terminate
the job instead of producing empty training batches.

## Dry Run

Inspect every resolved command without loading a model or creating shared
artifacts:

```bash
ROLE=verifier DRY_RUN=1 \
  bash examples/train/mtp_glm52_ascend_online.sh

ROLE=trainer DRY_RUN=1 \
  VERIFIER_HOST=10.0.0.10 \
  MASTER_ADDR=10.0.0.20 \
  NODE_RANK=0 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Replace the example addresses below with addresses routable over the cluster's
HCCL and service networks:

- `10.0.0.10`: verifier node
- `10.0.0.20`: trainer node 0 and rendezvous host
- `10.0.0.21`: trainer node 1
- `10.0.0.22`: trainer node 2

## Preflight

Run the local checks on each node before allocating the full job:

```bash
ROLE=preflight \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Preflight reads checkpoint metadata, verifies the critical native MTP tensors
and referenced shards, compares the BF16 and W4A8 architecture and tokenizer
assets, checks the token dataset, probes shared storage, verifies the pinned
software versions, and verifies 16 visible NPUs. It does not load the full GLM
checkpoint.

## Start the Verifier

On the verifier node:

```bash
ROLE=verifier \
  bash examples/train/mtp_glm52_ascend_online.sh
```

The service uses tensor parallel size 16, an 8193-token maximum request length,
and exposes `glm52-w4a8-verifier` on port 8000. Its log is:

```text
/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3/verifier/verifier.log
```

Do not start training until this returns HTTP 200 from a trainer node:

```bash
curl -f http://10.0.0.10:8000/health
```

## Smoke Test

Start these three commands close together in separate terminals or through the
cluster launcher. Node 0 atomically converts the BF16 native MTP layer once;
nodes 1 and 2 wait for its shared ready marker.

Choose one fresh identifier and export the same value on all three trainer
nodes. Reusing an identifier is rejected when its output directory exists:

```bash
export SMOKE_RUN_ID=smoke-20260813-01
```

Trainer node 0:

```bash
ROLE=smoke SMOKE_RUN_ID=$SMOKE_RUN_ID \
  VERIFIER_HOST=10.0.0.10 MASTER_ADDR=10.0.0.20 NODE_RANK=0 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Trainer node 1:

```bash
ROLE=smoke SMOKE_RUN_ID=$SMOKE_RUN_ID \
  VERIFIER_HOST=10.0.0.10 MASTER_ADDR=10.0.0.20 NODE_RANK=1 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Trainer node 2:

```bash
ROLE=smoke SMOKE_RUN_ID=$SMOKE_RUN_ID \
  VERIFIER_HOST=10.0.0.10 MASTER_ADDR=10.0.0.20 NODE_RANK=2 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Smoke mode selects 64 samples into run-specific shared storage, truncates each
online request to a 1024-token context, and performs two optimizer steps through
the real online hidden-state path. It writes separate
`-smoke-$SMOKE_RUN_ID` checkpoint, metric, and log directories and disables
checkpoint resume. Treat smoke
as successful only when all 48 ranks finish without HCCL, OOM, unsupported
operator, non-finite loss, or checkpoint errors.

## Full Training

After smoke succeeds, replace `ROLE=smoke` with `ROLE=trainer` on the same
three nodes. For example, trainer node 0 runs:

```bash
ROLE=trainer VERIFIER_HOST=10.0.0.10 MASTER_ADDR=10.0.0.20 NODE_RANK=0 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Use `NODE_RANK=1` and `NODE_RANK=2` on the other trainer nodes. The default
training configuration uses 4096 tokens per NPU batch, BF16 hidden states,
AdamW, cosine decay with 3% warmup, five epochs, and a checkpoint every 1000
global steps. The 4096-token default is conservative because FSDP does not
shard activations. Increase it to 8192 only after an actual 910B3 smoke run
demonstrates sufficient memory headroom. The verifier context must be at least
one token longer because hidden-state extraction requests one output token;
the launcher rejects `TOTAL_SEQ_LEN >= VERIFIER_MAX_MODEL_LEN`. Override any
setting when needed:

```bash
EPOCHS=1 CHECKPOINT_STEPS=500 TOTAL_SEQ_LEN=4096 ROLE=trainer \
  VERIFIER_HOST=10.0.0.10 MASTER_ADDR=10.0.0.20 NODE_RANK=0 \
  bash examples/train/mtp_glm52_ascend_online.sh
```

Trainer logs are written as:

```text
/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3/trainer-node0.log
/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3/trainer-node1.log
/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3/trainer-node2.log
```

Checkpoints are under:

```text
/mnt/xds/sfs/spec_train/checkpoints/glm52-w4a8-mtp3
```

Start TensorBoard on a reachable node with:

```bash
tensorboard \
  --logdir /mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3/metrics \
  --host 0.0.0.0 \
  --port 6006
```

## Stop and Resume

Send `SIGTERM` or press Ctrl-C on every trainer node. The launcher forwards the
signal to `torchrun`; the trainer's graceful-shutdown path saves a manual
recovery snapshot under `OUTPUT_PATH/interrupted`. Automatic resume deliberately
ignores that directory and restarts from the latest numbered checkpoint. To
load the interrupted weights instead, first follow the warning printed by the
checkpointer and rename `interrupted` to a new numeric checkpoint directory.
That recovery starts at an epoch boundary and does not preserve the exact
within-epoch data position. Stop the verifier only after every trainer process
has exited.

If preflight reports an existing MTP or smoke directory without `.ready`, do
not remove it automatically. Inspect whether another node is still producing
it, then move or remove the incomplete directory manually before retrying.
