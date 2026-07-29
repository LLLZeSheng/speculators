# GLM-5.2 8-GPU Serving Design

## Goal

Replace the currently running local DeepSeek-V4-Flash service with a baseline
GLM-5.2 OpenAI-compatible API. The GLM service must use all eight visible GPUs
and must not enable DSpark or any other speculative decoder.

## Inputs and Constraints

- Target weights:
  `/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1`
- Weight format: compressed-tensors mixed precision with NVFP4 W4A4 experts.
- Model architecture: `GlmMoeDsaForCausalLM` (`glm_moe_dsa`).
- Serving runtime: `/mnt/pass/miniconda3/bin/vllm`, local version 0.26.0.
- Existing DeepSeek-V4-Flash workers and router occupy all eight GPUs and TCP
  port 8000. They are managed by:
  `/mnt/paas/h00947744/DeepSeek-V4-Flash-DSpark/deploy/supervisord.conf`.
- The initial GLM launch prioritizes a reliable load over maximum context
  length or graph-capture performance.

## Selected Topology

Use one vLLM API server with tensor parallel size 8 and expert parallelism
enabled:

- TP: 8
- EP: enabled across the TP ranks
- PCP, PP, DP: 1
- GPUs: 0 through 7
- API bind address: `0.0.0.0:8000`
- Served model name: `glm-5.2`

This is simpler than an 8-rank TP/PCP hybrid and uses all eight GPUs without
introducing PCP-specific execution constraints during the first load.

## Lifecycle

1. Confirm the live DeepSeek supervisor belongs to the expected configuration.
2. Shut it down through `supervisorctl`, allowing its process groups to exit
   cleanly.
3. Verify ports 8000, 8100-8103, and 8200-8203 are released and that GPU memory
   has returned to an idle state.
4. Start vLLM detached, writing stdout/stderr to
   `/tmp/glm52-vllm.log` and its parent PID to `/tmp/glm52-vllm.pid`.
5. Wait for model loading to finish without treating a long safetensors load as
   an immediate failure.
6. Validate the health endpoint, model listing, and one minimal completion.

## vLLM Configuration

The initial launch uses:

- `--tensor-parallel-size 8`
- `--enable-expert-parallel`
- `--moe-backend flashinfer_cutlass`
- `--kv-cache-dtype fp8`
- `--safetensors-load-strategy prefetch`
- `--max-model-len 4096`
- `--max-num-batched-tokens 32768`
- `--gpu-memory-utilization 0.90`
- `--enforce-eager`
- `--trust-remote-code`

`VLLM_USE_V2_MODEL_RUNNER=1` is set because the checked-in GLM-5.2 NVFP4
evaluation configuration for this local vLLM build uses the V2 runner. No
`--speculative-config` argument is supplied. `CUDA_VISIBLE_DEVICES` is set
explicitly to `0,1,2,3,4,5,6,7`.

## Failure Handling

- If DeepSeek does not stop cleanly, do not launch GLM on partially occupied
  GPUs; report the remaining processes first.
- If vLLM rejects TP=8 or the NVFP4 backend, preserve the complete log and stop
  the partial GLM process group before changing topology or quantization
  settings.
- If model loading exceeds the initial observation window but the process is
  alive and logs continue advancing, continue monitoring rather than restarting.
- Do not automatically restart DeepSeek after a GLM failure. Its original
  supervisor configuration and launch files remain intact for explicit
  rollback.

## Verification

Success requires all of the following:

1. A live vLLM parent process references the target GLM path and TP size 8.
2. All eight GPUs contain GLM worker allocations and no DeepSeek worker remains.
3. `GET http://127.0.0.1:8000/health` succeeds.
4. `GET http://127.0.0.1:8000/v1/models` lists `glm-5.2`.
5. A short non-streaming chat-completion request returns generated text.

## Out of Scope

- DSpark or MTP speculative decoding
- Long-context tuning beyond the initial 4096-token limit
- Throughput benchmarking or production auto-restart integration
- Changes to model weights or the vLLM installation
