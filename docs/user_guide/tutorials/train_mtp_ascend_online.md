# Train GLM-5.2 MTP3 Online on Eight Ascend 910C Nodes

This is the production runbook for four 16-NPU verifier nodes plus four
16-NPU trainer nodes. Trainer `i` uses verifier `i`; the trainers form one
64-rank FSDP job.

The intended data lifecycle is:

```text
epoch 1: online generation -> persistent hidden-state cache -> training
epoch 2+: read the same cache -> ordinary offline-style training
```

The first epoch uses `--on-missing generate --on-generate cache` and does not
use `--force-generate`. Consequently every later access is a local/shared-file
cache hit. An explicit `offline` mode is provided for strict resumes; it uses
`--on-missing raise` and does not pass a vLLM endpoint.

For the Chinese version, see
[八台 Ascend 910C 在线训练 GLM-5.2 MTP3](train_mtp_ascend_online_zh.md).

## 1. Fixed environment and model contract

- image: `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`
- shared filesystem: `/kos_ulan` on all eight nodes
- checkout: preferably `/kos_ulan/spec_train/speculators`
- prepared verifier:
  `/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4`
- verifier source weights:
  `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`
- a BF16 GLM-5.2 checkpoint containing the native MTP layer on `/kos_ulan`
- a Hugging Face dataset containing `input_ids`, `loss_mask`, and `seq_len`

The W4A8 verifier is inference-only. MTP initialization and training remain
BF16, so `MTP_INIT_MODEL_PATH` must point to the BF16 model. The verifier must
use `VERIFIER_QUANTIZATION_MODE=ascend`; see
[the MG13 ModelSlim diagnosis](glm52_mixed_compressed_tensors.md).

Expected image packages are `torch_npu==2.10.0.post2`, `vllm==0.23.0`, and
`vllm-ascend==0.23.0rc1`.

## 2. Create the shared configuration

### Recommended: configure the cluster from one control host

Set up passwordless SSH from the control host to all eight nodes. If the
standard model and dataset paths listed below exist, the only required inputs
are the eight IPs and one Docker container-name prefix:

```bash
cd /kos_ulan/spec_train/speculators
bash examples/train/manage_mtp_glm52_ascend_online_4v4t.sh configure \
  --verifier-ips V0,V1,V2,V3 \
  --trainer-ips T0,T1,T2,T3 \
  --container-prefix glm52-online-mtp3
```

The generated shared configuration is:

```text
/kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

The defaults are:

```text
BF16 MTP initialization model: /kos_ulan/models/GLM-5.2
training dataset:              /kos_ulan/datasets/glm52-mtp-online
repository:                    /kos_ulan/spec_train/speculators
training context:              32768 tokens
verifier model length:         32769 tokens
verifier batched-token budget: 32768 tokens
online request timeout:        900 seconds
```

The one-token difference between the training context and model length is
intentional: hidden-state extraction sends one generation token. The launcher
disables chunked prefill, so `VERIFIER_MAX_BATCHED_TOKENS` must be at least
`TOTAL_SEQ_LEN`. With DP2, one verifier node can actively prefill about two
full 32K samples at once; additional requests queue. `VERIFIER_MAX_NUM_SEQS=8`
keeps capacity for shorter samples without claiming eight simultaneous 32K
prefills.

Override a non-standard path during `configure` with `--mtp-init-model`,
`--data-path`, or `--repo-path`. The manager resolves `NIC_NAME` separately on
each host from its configured IP, so the shared configuration does not require
a common interface name.

The complete managed workflow is:

```bash
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
bash "$MANAGER" preflight
bash "$MANAGER" start-verifiers
bash "$MANAGER" wait-verifiers
bash "$MANAGER" smoke
bash "$MANAGER" status
# After all four smoke containers finish successfully:
bash "$MANAGER" train
```

For an offline cache-only resume:

```bash
bash "$MANAGER" offline
```

To inspect or stop only containers with this configured prefix:

```bash
bash "$MANAGER" status
bash "$MANAGER" stop
```

The manager never stores an SSH password. Use `SSH_USER`, `SSH_PORT`, and
`SSH_IDENTITY_FILE` when their defaults are unsuitable.

To inspect every generated remote command without opening SSH sessions or
starting containers, prefix an operation with `MANAGER_DRY_RUN=1`.

### Manual configuration

```bash
cd /kos_ulan/spec_train/speculators
mkdir -p /kos_ulan/spec_train/config /kos_ulan/spec_train/logs
cp examples/train/mtp_glm52_ascend_online_4v4t.env.example \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
vim /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

Fill all eight IPs, `NIC_NAME`, `MTP_INIT_MODEL_PATH`, `DATA_PATH`, and a
unique `SMOKE_RUN_ID`. Verify these critical values:

```bash
VERIFIER_MODEL_PATH=/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
VERIFIER_SOURCE_MODEL_PATH=/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
VERIFIER_QUANTIZATION_MODE=ascend
VERIFIER_MAX_MODEL_LEN=32769
VERIFIER_MAX_BATCHED_TOKENS=32768
VERIFIER_MAX_NUM_SEQS=8
TOTAL_SEQ_LEN=32768
REQUEST_TIMEOUT=900
TRAINER_MODE=smoke
TRAINER_DATA_MODE=online-cache
EPOCHS=5
```

Use one identical file on all nodes. IPs must be mutually routable and appear
in `hostname -I`. Use `NODE_IP=<chosen-ip>` for a host with ambiguous IPs.

`NIC_NAME=auto` is supported and resolves the interface from `NODE_IP` on each
host before Docker starts.

## 3. Dry-run and smoke test

On one verifier and one trainer, inspect commands without loading models:

```bash
cd /kos_ulan/spec_train/speculators
DRY_RUN=1 bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

Verifier output must contain `--quantization ascend`, must use the v4 path,
and must not invoke `prepare_mixed_quant_model.py`.

Start all four verifiers with the same command on each verifier node:

```bash
cd /kos_ulan/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/verifier-container-$(hostname).log 2>&1 &
```

Wait for all four health checks:

```bash
curl -f http://VERIFIER_IP:8077/health
```

With `TRAINER_MODE=smoke`, run the wrapper close together on all four trainer
nodes. Smoke uses 64 samples and two steps, and deletes its generated files so
it does not contaminate the production cache:

```bash
cd /kos_ulan/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/smoke-container-$(hostname).log 2>&1 &
```

All 64 trainer ranks must exit successfully. Use a new `SMOKE_RUN_ID` for a
repeat.

## 4. Production: online-cache first epoch, cache hits thereafter

Set `TRAINER_MODE=trainer`, keep `TRAINER_DATA_MODE=online-cache`, and keep the
final total `EPOCHS` value (for example 5) from the beginning. Start on all
four trainer nodes within the HCCL rendezvous timeout:

```bash
cd /kos_ulan/spec_train/speculators
TRAINER_MODE=trainer TRAINER_DATA_MODE=online-cache nohup \
  bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/trainer-container-$(hostname).log 2>&1 &
```

Do not launch epoch 1 as a separate `EPOCHS=1` job. Starting with the final
epoch count preserves one continuous optimizer and cosine scheduler. At the
end of epoch 1, both training and validation partitions have been traversed,
the numbered checkpoint exists, and hidden states are persistent under:

```text
/kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8
```

Epochs 2 through 5 first look up each sample in that cache, so they follow the
offline data path and issue no generation request for complete entries. Keep
the verifiers alive for the first production run as a safety net; their
request counters should become flat after epoch 1.

## 5. Strict offline resume

After epoch 1 has completed successfully, any later restart may use strict
offline mode:

```bash
cd /kos_ulan/spec_train/speculators
TRAINER_MODE=trainer TRAINER_DATA_MODE=offline nohup \
  bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env \
  > /kos_ulan/spec_train/logs/trainer-offline-container-$(hostname).log 2>&1 &
```

Start it on all four trainers. It automatically resumes the latest numbered
checkpoint, passes no verifier endpoint, and fails on the first missing cache
entry. This fail-closed behavior prevents accidental partial online/offline
training. Verifiers may be stopped only after this strict run has begun
fetching batches successfully on every trainer.

Never change `DATA_PATH`, `TRAIN_DATA_RATIO`, `TOTAL_SEQ_LEN`,
`HIDDEN_STATES_PATH`, `OUTPUT_PATH`, or `RUN_NAME` between the two modes.

## 6. Logs, checkpoints, and TensorBoard

Verifier logs:

```text
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
```

Trainer logs and checkpoints:

```text
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log
...
/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node3.log
/kos_ulan/spec_train/checkpoints/glm52-w4a8c8-mtp3/
```

TensorBoard:

```bash
tensorboard \
  --logdir /kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/metrics \
  --host 0.0.0.0 --port 6006
```

Useful checks:

```bash
grep -E "Training epoch|Validation epoch|Saving checkpoint|ERROR|Traceback" \
  /kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log | tail -100
find /kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8 \
  -type f -name 'hs_*.safetensors' | wc -l
du -sh /kos_ulan/spec_train/online_hidden_states/glm52-w4a8c8
```

Success means finite loss, no HCCL or generation error, a completed epoch-1
validation, a numbered checkpoint, increasing cache file count during epoch
1, and stable cache count during later epochs.

## 7. Native-MTP inference patch boundary

The hidden-state launcher uses a speculative configuration whose method is
`extract_hidden_states`; it does not use native `deepseek_mtp`. It therefore
runs the target model to generate layer-78 hidden states without constructing
or loading the native MTP drafter's shared embedding/head, and does not need
`patch_vllm_glm52_mtp_shared_weights.py`.

Apply that loader patch later in any inference or benchmark container which
loads native MTP speculative decoding. It is an inference-side requirement,
not an online-training requirement. See
[the MG13 ModelSlim diagnosis](glm52_mixed_compressed_tensors.md).

## 8. Recovery rules

- Stop or restart all four trainer nodes together.
- A normal restart resumes the latest numbered checkpoint.
- Do not treat an `interrupted` directory as a completed epoch checkpoint
  without inspection.
- Do not delete a shared cache while any trainer or verifier is running.
- If a shared initialization/metadata directory lacks `.ready`, first prove no
  producer is active, then move the incomplete directory aside for diagnosis.
