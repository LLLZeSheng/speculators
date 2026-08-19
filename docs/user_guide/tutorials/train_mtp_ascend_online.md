# Train GLM-5.2 MTP3 Online on Ascend 910C Nodes

This runbook supports four 16-NPU verifier nodes plus either four trainer nodes
(64-rank FSDP) or two trainer nodes (32-rank FSDP). Existing 4V4T behavior is
unchanged.

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
- shared filesystem: `/mnt/xds/mtp` on every configured node
- checkout: `/mnt/xds/mtp/spec_train/speculators`
- prepared verifier:
  `/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4`
- verifier source weights:
  `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1`
- the prepared MG13 runtime view containing `mtp.safetensors` and
  `quant_model_weights.safetensors.index.json`
- a Hugging Face dataset containing `input_ids`, `loss_mask`, and `seq_len`

The main MG13 verifier layers remain inference-only W4A8. Its native layer-78
MTP tensors and shared embedding/head tensors are floating point, however, so
the prepared v4 runtime view is also the preferred `MTP_INIT_MODEL_PATH`.
Speculators reads the MTP tensors from `mtp.safetensors` (or the ModelSlim
index), checks that every trainable tensor is floating point, and reads shared
weights through `quant_model_weights.safetensors.index.json`. No source model
file is modified. The verifier must use `VERIFIER_QUANTIZATION_MODE=ascend`; see
[the MG13 ModelSlim diagnosis](glm52_mixed_compressed_tensors.md).

Expected image packages are `torch_npu==2.10.0.post2`, `vllm==0.23.0`, and
`vllm-ascend==0.23.0rc1`.

## 2. Define the shared YAML yourself

The YAML is user-maintained source of truth. The manager only reads and
validates it; it never generates or rewrites it. Copy the template, then edit
the machine addresses, container settings, and paths directly:

```bash
cd /mnt/xds/mtp/spec_train/speculators
mkdir -p /mnt/xds/mtp/spec_train/config
cp examples/train/mtp_glm52_ascend_online_4v4t.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
vim /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

For 4 verifiers + 2 trainers, copy the additive template instead:

```bash
cp examples/train/mtp_glm52_ascend_online_4v2t.example.yaml \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v2t.yaml
```

No other switch is needed. The manager sets `NNODES=2`; local ranks on trainer
0 alternate between verifiers 0 and 2, while trainer 1 uses verifiers 1 and 3.
All four verifier nodes therefore remain active. With 4V4T, trainer `i` still
uses verifier `i`.

The shared configuration is:

```text
/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

It explicitly contains four verifier IPs, two or four trainer IPs, image,
container mode, container mounts, repository, model/data/output paths,
32K sizing, and per-role installation policy. Replace every `FILL_*` value.

Choose one container lifecycle mode:

```yaml
container_mode: create
container_name_prefix: glm52-w4a8-mg13-speculator-training
container_repo_path: /mnt/xds/mtp/spec_train/speculators
repo_path: /mnt/xds/mtp/spec_train/speculators
container_mounts:
  - /mnt/xds/sfs:/mnt/xds/sfs
  - /mnt/xds/mtp:/mnt/xds/mtp
  - /mnt/xds/mtp/spec_train/speculators:/mnt/xds/mtp/spec_train/speculators
  - /root/.cache:/root/.cache
```

`create` does not use or require `existing_container_name`. It runs the
configured image with host networking, 1 GiB shared memory by default,
all 16 NPUs, Ascend driver files, and the YAML mount list. Add arbitrary
`host_path:container_path[:ro|rw]` entries to that list. The manager runs under
nohup, so it intentionally omits terminal-only `-it`.

Set `container_mode: existing` to reuse one already-running container with the
configured `existing_container_name` on every host. In this mode the wrapper
uses `docker exec`; image, shm, device, and mount settings must already be
correct on that container. The manager's `stop` command first terminates all
tracked MTP processes and then runs `docker restart -t 30` once for the reused
container on each of the eight hosts. It never removes the reused containers.
This clears stale NPU workers and process-local state; services unrelated to
this workflow must therefore not share those containers.
The PID receives SIGTERM first, but the manager waits only 15 seconds by
default because the following container restart is the final cleanup boundary.
Override this with `STOP_GRACE_SECONDS=N` when a job needs a different grace.

For role-scoped cleanup in `existing` mode, use `restart-verifiers` to stop
only verifier jobs and restart the four verifier containers, or
`restart-trainers` to stop both trainer and smoke jobs and restart the four
trainer containers. The latter restarts each shared trainer container only
once. After `restart-verifiers`, run `start-verifiers` again to load the model.

If a pre-created container mounts only `/mnt/xds/mtp:/mnt/xds/mtp` and therefore has
no `/workspace/speculators`, set both `repo_path` and `container_repo_path` to
the repository path visible through `/mnt/xds/mtp`, for example
`/mnt/xds/mtp/spec_train/speculators`. Create mode intentionally omits
`--rm`; remove a stopped same-name container manually before recreating it, or
switch to existing mode.

Validate it locally before any SSH or Docker operation:

```bash
MANAGER=examples/train/manage_mtp_glm52_ascend_online_4v4t.sh
bash "$MANAGER" validate-config \
  --config /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

Important configured values are:

```text
native MTP initialization:     /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
training dataset:              /mnt/xds/mtp/spec_train/dataset/hf/nuoya-average2k8k-32k
repository:                    /mnt/xds/mtp/spec_train/speculators
training context:              32768 tokens
verifier model length:         32769 tokens
verifier batched-token budget: 32784 tokens
online request timeout:        900 seconds
```

The one-token difference between the training context and model length is
intentional: hidden-state extraction sends one generation token. TP16 sequence
parallel pads the profile size to a multiple of sixteen, so the batched-token
budget is 32784 rather than 32769. DP1 x TP16 is required to leave enough
per-device activation headroom on 61 GiB NPUs; each verifier node runs one
full 32K prefill at a time and additional requests queue. The memory-safe
profile uses `VERIFIER_MAX_NUM_SEQS=1`, eager execution, and disables
shared-expert multistream overlap to preserve activation headroom on 61 GiB
NPUs. It uses `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64`; torch_npu does
not allow `max_split_size_mb` and `expandable_segments` simultaneously.

Set non-standard paths directly in YAML. With `nic_name: auto`, the wrapper
resolves the interface separately on each host from its configured IP.

The complete managed workflow is:

```bash
CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
bash "$MANAGER" preflight --config "$CONFIG"
bash "$MANAGER" start-verifiers --config "$CONFIG"
bash "$MANAGER" wait-verifiers --config "$CONFIG"
bash "$MANAGER" smoke --config "$CONFIG"
bash "$MANAGER" status --config "$CONFIG"
# After all four smoke containers finish successfully:
bash "$MANAGER" train --config "$CONFIG"
```

`start-verifiers` is idempotent. It skips healthy nodes, avoids a duplicate
launch when a verifier process exists but is still becoming healthy, and only
starts nodes with no running verifier job. To target one YAML entry, run:

```bash
bash "$MANAGER" start-verifier --index 2 --config "$CONFIG"
```

The index is the zero-based position in `verifier_ips` (`0..3`).

For an offline cache-only resume:

```bash
bash "$MANAGER" offline --config "$CONFIG"
```

To inspect or stop only containers with this configured prefix:

```bash
bash "$MANAGER" status --config "$CONFIG"
bash "$MANAGER" stop --config "$CONFIG"
```

The manager never stores an SSH password. Use `SSH_USER`, `SSH_PORT`, and
`SSH_IDENTITY_FILE` when their defaults are unsuitable.

To inspect every generated remote command without opening SSH sessions or
starting containers, prefix an operation with `MANAGER_DRY_RUN=1`.

### Prepare the 1.2M Nuoya mixture

The resumable converter processes each JSONL independently before publishing
one shuffled 32K Hugging Face dataset. Completed staging shards are reused
after interruption:

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup python scripts/prepare_glm52_nuoya_32k.py \
  --model /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4 \
  > /mnt/xds/mtp/spec_train/dataset/prepare-nuoya-32k.log 2>&1 &
```

Its default sources are `average-2k-nuoya` and `average-8k`, and its output is
`/mnt/xds/mtp/spec_train/dataset/hf/nuoya-average2k8k-32k`. The output
`conversion_manifest.json` records row count, length percentiles, total token
count, and the estimated BF16 hidden-state cache size for one 6144-wide layer.

### Review the user-maintained YAML

Fill all configured IPs, `nic_name`, `mtp_init_model_path`, `data_path`, and a
unique `smoke_run_id`. Verify these critical YAML values:

```yaml
verifier_model_path: /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
verifier_source_model_path: /mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
verifier_quantization_mode: ascend
mtp_init_model_path: /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
verifier_max_model_len: 32769
verifier_max_batched_tokens: 32784
verifier_max_num_seqs: 1
total_seq_len: 32768
request_timeout: 900
trainer_mode: smoke
trainer_data_mode: online-cache
epochs: 5
install_speculators_verifier: false
install_speculators_trainer: false
```

Use one identical file on all nodes. IPs must be mutually routable and appear
in `hostname -I`. Use `NODE_IP=<chosen-ip>` for a host with ambiguous IPs.

Do not point `mtp_init_model_path` at the standalone `mtp.safetensors` file.
It must be the model directory because conversion also needs `config.json`,
tokenizer files, and the shared embedding/head tensors. A conventional BF16
GLM-5.2 directory remains supported as a fallback.

`nic_name: auto` is supported and resolves the interface from `NODE_IP` on each
host before Docker starts.

### What happens inside each container

No interactive container setup is required. The host wrapper makes the role
behavior deterministic:

- verifier, trainer, and smoke do not run pip. The launcher adds both
  `/mnt/xds/mtp/spec_train/speculators/src` and
  `/mnt/xds/mtp/spec_train/speculators/hs_connectors/src` to `PYTHONPATH`;
  trainer then starts the 64-rank `torchrun` job;
- all roles use the same configured vLLM-Ascend image.

Trainer ranks serialize the initial verifier embedding/head read through the
host-local lock `/tmp/speculators-glm52-verifier-weights.lock`. This prevents
16 ranks from faulting the same SFS-backed safetensors pages concurrently; the
first rank warms the page cache and the remaining ranks load in turn. The lock
is advisory and is automatically released if its owning process exits.

This is a required compatibility rule: vLLM 0.23 in the image requires
`setuptools<81`, while this repository's editable build isolation requires
`setuptools>=82`. Do not upgrade setuptools in the serving image merely to
perform an editable install. Enable `install_speculators_trainer` only for a
custom image where that packaging constraint has been resolved.

The two installation switches live in YAML and may be overridden for a custom
image. Leave the verifier switch false unless its image lacks imports needed
by a locally modified launcher.

## 3. Dry-run and smoke test

On one verifier and one trainer, inspect commands without loading models:

```bash
cd /mnt/xds/mtp/spec_train/speculators
DRY_RUN=1 bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
```

Verifier output must contain `--quantization ascend`, must use the v4 path,
and must not invoke `prepare_mixed_quant_model.py`.

Start all four verifiers with the same command on each verifier node:

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml \
  > /mnt/xds/mtp/spec_train/logs/verifier-container-$(hostname).log 2>&1 &
```

Wait for all four health checks:

```bash
curl -f http://VERIFIER_IP:8077/health
```

With `TRAINER_MODE=smoke`, run the wrapper close together on all four trainer
nodes. Smoke uses 64 samples and two steps, and deletes its generated files so
it does not contaminate the production cache:

```bash
cd /mnt/xds/mtp/spec_train/speculators
nohup bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml \
  > /mnt/xds/mtp/spec_train/logs/smoke-container-$(hostname).log 2>&1 &
```

All 32 or 64 trainer ranks must exit successfully. Use a new `SMOKE_RUN_ID` for a
repeat.

## 4. Production: online-cache first epoch, cache hits thereafter

Set `TRAINER_MODE=trainer`, keep `TRAINER_DATA_MODE=online-cache`, and keep the
final total `EPOCHS` value (for example 5) from the beginning. Start on all
configured trainer nodes within the HCCL rendezvous timeout:

```bash
cd /mnt/xds/mtp/spec_train/speculators
TRAINER_MODE=trainer TRAINER_DATA_MODE=online-cache nohup \
  bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml \
  > /mnt/xds/mtp/spec_train/logs/trainer-container-$(hostname).log 2>&1 &
```

Do not launch epoch 1 as a separate `EPOCHS=1` job. Starting with the final
epoch count preserves one continuous optimizer and cosine scheduler. At the
end of epoch 1, both training and validation partitions have been traversed,
the numbered checkpoint exists, and hidden states are persistent under:

```text
/mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8
```

Epochs 2 through 5 first look up each sample in that cache, so they follow the
offline data path and issue no generation request for complete entries. Keep
the verifiers alive for the first production run as a safety net; their
request counters should become flat after epoch 1.

## 5. Strict offline resume

After epoch 1 has completed successfully, any later restart may use strict
offline mode:

```bash
cd /mnt/xds/mtp/spec_train/speculators
TRAINER_MODE=trainer TRAINER_DATA_MODE=offline nohup \
  bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml \
  > /mnt/xds/mtp/spec_train/logs/trainer-offline-container-$(hostname).log 2>&1 &
```

Start it on all four trainers. It automatically resumes the latest numbered
checkpoint, passes no verifier endpoint, and fails on the first missing cache
entry. This fail-closed behavior prevents accidental partial online/offline
training. Verifiers may be stopped only after this strict run has begun
fetching batches successfully on every trainer.

Never change `DATA_PATH`, `TRAIN_DATA_RATIO`, `TOTAL_SEQ_LEN`,
`HIDDEN_STATES_PATH`, `OUTPUT_PATH`, or `RUN_NAME` between the two modes.

## 6. Logs, checkpoints, and TensorBoard

### Control-node web dashboard

The manager starts a dependency-free, read-only dashboard after
`start-verifiers`, `smoke`, `train`, or `offline` when the YAML contains:

```yaml
dashboard_host: 0.0.0.0
dashboard_port: 6007
dashboard_auto_start: true
```

Open `http://<control-node-ip>:6007`. The page refreshes every five seconds
and combines the shared host-wrapper logs, detailed verifier logs, and direct
`/health` probes. It shows all verifier and trainer nodes, startup/training
phase, epoch/step/loss, prompt and generation throughput, queue depth, KV-cache
usage, log age, and the latest error. It runs only on the control node and
installs nothing in verifier or trainer containers. Health probes explicitly
bypass `HTTP_PROXY` and `HTTPS_PROXY` so private cluster addresses remain local.

Manual operations are:

```bash
bash "$MANAGER" dashboard --config "$CONFIG"
bash "$MANAGER" dashboard-status --config "$CONFIG"
bash "$MANAGER" stop-dashboard --config "$CONFIG"
```

Set `DASHBOARD_ADVERTISE_HOST=<control-node-ip>` when the control node has
multiple interfaces and the printed URL selects the wrong one. The dashboard
has no authentication; bind it to `127.0.0.1` and use an SSH tunnel, or protect
port 6007 with the cluster firewall, when the network is not trusted. Cluster
`stop` deliberately leaves the page running so the final logs remain visible.

Verifier logs:

```text
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/verifier0/verifier.log
...
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/verifier3/verifier.log
```

Verifier startup automatically runs
`scripts/patch_vllm_glm52_final_hidden_state.py` and
`scripts/patch_vllm_ascend_hidden_state_cache.py`, plus
`scripts/patch_vllm_hidden_state_enolck.py` and
`scripts/patch_vllm_hidden_state_connector_tp_gather.py`. In vLLM 0.23 the
DeepSeek/GLM forward path samples auxiliary states before decoder blocks
(`0..77`), while MTP layer id `78` denotes the final normalized output after
the last block. The second patch prevents Ascend's MLA merge path from treating
`HiddenStateCacheSpec` as a quantized cache carrying `scale_dim`. These patches change neither model
weights nor model configuration and are idempotent. The ENOLCK patch handles
shared filesystems that reject `flock`: it synchronously finishes each hidden-
state file before returning its path, instead of killing EngineCore or exposing
an incomplete safetensors file. The original source is
backed up as `deepseek_v2.py.before-glm-final-aux-hidden-state-fix`. To inspect
or restore it manually:

```bash
python scripts/patch_vllm_glm52_final_hidden_state.py --check
python scripts/patch_vllm_glm52_final_hidden_state.py --restore
python scripts/patch_vllm_ascend_hidden_state_cache.py --check
python scripts/patch_vllm_ascend_hidden_state_cache.py --restore
python scripts/patch_vllm_hidden_state_enolck.py --check
python scripts/patch_vllm_hidden_state_enolck.py --restore
python scripts/patch_vllm_hidden_state_connector_tp_gather.py --check
python scripts/patch_vllm_hidden_state_connector_tp_gather.py --restore
```

The verifier deliberately sets `enable_dsa_cp=false`. With DSA context
parallelism enabled, vLLM-Ascend exposes only the worker-local sequence shard
to `ExampleHiddenStatesConnector` (for example, 128 hidden-state rows for a
1024-token prompt at CP=8), while `token_ids` still describes the full prompt.
The connector-boundary compatibility patch also compares the extracted tensor
against the authoritative full `token_ids` length, performs a TP gather if a
short shard still reaches the saver, and refuses to publish any mismatched
file. Disabling DSA CP remains the primary path; the save-boundary check is a
fail-closed second line of defense.

Trainer logs and checkpoints:

```text
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log
...
/mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node3.log
/mnt/xds/mtp/spec_train/checkpoints/glm52-w4a8c8-mtp3/
```

TensorBoard:

```bash
tensorboard \
  --logdir /mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/metrics \
  --host 0.0.0.0 --port 6006
```

Useful checks:

```bash
grep -E "Training epoch|Validation epoch|Saving checkpoint|ERROR|Traceback" \
  /mnt/xds/mtp/spec_train/logs/glm52-w4a8c8-mtp3/trainer-node0.log | tail -100
find /mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8 \
  -type f -name 'hs_*.safetensors' | wc -l
du -sh /mnt/xds/mtp/spec_train/online_hidden_states/glm52-w4a8c8
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

- Stop or restart all configured trainer nodes together.
- A normal restart resumes the latest numbered checkpoint.
- Do not treat an `interrupted` directory as a completed epoch checkpoint
  without inspection.
- Do not delete a shared cache while any trainer or verifier is running.
- If a shared initialization/metadata directory lacks `.ready`, first prove no
  producer is active, then move the incomplete directory aside for diagnosis.
