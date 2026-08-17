# GLM-5.2 Ascend PP2 Hidden-State Verifier Plan

## Status and activation boundary

This is a ready-to-apply fallback plan.  It does **not** change the current
verifier topology until the user explicitly asks to apply it after another
DP1 x TP16 startup failure.

The target topology per 16-NPU verifier node is:

```text
DP=1 x PP=2 x TP=8 = 16 NPU workers
```

The four verifier nodes remain independent endpoints.  Trainer `i` continues
to use verifier `i`; the four trainer nodes remain one 64-rank FSDP job.

## Why PP instead of PD

Hidden-state extraction is almost entirely a 32K prefill workload and emits
only one output token.  PD separation leaves the full prefill model and its
activation peak on the P node, so it does not address the observed OOM.
PP2 splits the 78 decoder blocks between two stages and directly reduces the
per-worker model-state peak.  Decode latency is not an optimization target.

## Configuration changes

Add a user-maintained YAML field:

```yaml
verifier_dp_size: 1
verifier_pp_size: 2
verifier_tp_size: 8
verifier_max_model_len: 32769
verifier_max_batched_tokens: 32776
verifier_max_num_seqs: 1
```

`32776` is the smallest multiple of TP8 that covers the 32769-token model
window.  Keep eager execution, `max_num_seqs=1`, sparse LI C8, and the
verifier-only allocator setting `max_split_size_mb:64`.  Keep chunked prefill
disabled because vLLM hidden-state extraction explicitly does not support it.

Update the YAML renderer and validation as follows:

- map `verifier_pp_size` to `VERIFIER_PP_SIZE`;
- require `dp * pp * tp == 16`;
- require `max_num_batched_tokens >= max_model_len`;
- require `max_num_batched_tokens % tp == 0`;
- forward `VERIFIER_PP_SIZE` through the host/container wrapper;
- calculate preflight `required_devices` as `dp * pp * tp`.

Add this vLLM argument only when the plan is activated:

```text
--pipeline-parallel-size 2
```

## vLLM-Ascend compatibility patch

Create an idempotent, signature-checked patch script for
`vllm_ascend/worker/model_runner_v1.py`.  It must back up the original file,
support `--check` and `--restore`, compile the patched source before replacing
it, and fail closed if the expected source anchors are not unique.

The current code configures auxiliary hidden states as follows:

```python
should_configure_aux_hidden_states = (
    self.use_aux_hidden_state_outputs
    if pp_group.world_size == 1
    else self._eagle3_uses_aux_hidden_state()
)
```

Under PP, `extract_hidden_states` creates its drafter only on the last stage,
so `self.use_aux_hidden_state_outputs` is already false on stage 0 and true on
stage 1.  Change the PP condition to allow either the last-stage extraction
flag or EAGLE3:

```python
should_configure_aux_hidden_states = (
    self.use_aux_hidden_state_outputs
    or self._eagle3_uses_aux_hidden_state()
)
```

Do not run `patch_eagle3_pp_aux_propagation()` for extraction-only mode.  That
propagation is needed when EAGLE3 collects intermediate states across PP
stages; this MTP job requests only layer id 78, which exists after final norm
on the last PP stage.  Gate that block with
`self._eagle3_uses_aux_hidden_state()`.

The existing final-hidden-state patch remains required.  On stage 1,
`DeepseekV2Model.end_layer == 78`; after final norm it appends that state when
layer 78 is requested.  Stage 0 returns `IntermediateTensors` and never enters
the auxiliary-state tuple path.  The existing Ascend hidden-state cache merge
patch also remains required.

## Expected PP data flow

```text
tokens/positions
      |
      v
PP stage 0: decoder blocks 0..38
      |
      | IntermediateTensors(hidden_states, residual)
      v
PP stage 1: decoder blocks 39..77 -> final norm
      |
      | (final_hidden_states, [layer_78_hidden_states])
      v
ExtractHiddenStatesProposer -> ExampleHiddenStatesConnector -> shared cache
```

There is no cross-document concatenation and no change to token IDs, loss
masks, target-layer semantics, quantized weights, or model `config.json`.

## Implementation sequence

1. Stop all verifier jobs and wait for every worker to exit.
2. Verify all 16 logical NPUs on every verifier node are free.
3. Apply the PP compatibility patch in verifier containers only.
4. Render and validate the PP2 YAML.
5. Dry-run one verifier and confirm DP1, PP2, TP8, 32769/32776, eager mode,
   both existing hidden-state patches, and the new PP patch.
6. Start verifier0 only; do not start trainers yet.
7. Wait for `/health`, then run 128-token, 8K, and 32K extraction probes.
8. Validate each output before starting the remaining verifiers.
9. Start verifier1..3, wait for all health endpoints, then run the existing
   smoke training job.

## Acceptance checks

Startup acceptance:

- exactly 16 workers are distributed over all logical NPUs;
- no worker OOMs during weight load, profile, or KV-cache initialization;
- no `IntermediateTensors` unpack error on PP stage 0;
- no missing auxiliary state or layer-78 tuple error on PP stage 1;
- all four `/health` endpoints return HTTP 200.

Data acceptance for each probe:

- one complete `.safetensors` file is atomically published;
- `hidden_states.shape == [num_tokens, 1, 6144]`;
- `token_ids.shape == [num_tokens]`;
- every hidden-state value is finite;
- saved token IDs match the request's tokenized input/output sequence;
- no partial temporary file remains after completion.

Training acceptance:

- two-step smoke training completes on all 64 trainer ranks;
- cache misses generate once and later accesses are cache hits;
- loss and gradients are finite;
- no trainer changes verifier assignment.

## Memory and performance expectation

PP2 should materially reduce static layer memory per worker relative to the
58+ GiB observed under TP16.  Exact memory is checkpoint- and stage-dependent
because embeddings, final head, experts, and runtime workspaces are not evenly
distributed.  The go/no-go requirement is at least 4 GiB free on the most
loaded rank after initialization and a successful 32K probe.  Throughput may
be lower because a single request incurs a pipeline bubble; correctness and
memory stability take priority.

## Rollback

If PP startup or the 32K probe fails:

1. stop all verifier processes and confirm NPUs are empty;
2. run the new PP patch script with `--restore` in each verifier container;
3. restore YAML to DP1 x PP1 x TP16 and batched tokens 32784;
4. do not delete any completed hidden-state cache files;
5. preserve verifier logs and patch-status output for diagnosis.

No trainer checkpoint, dataset, hidden-state cache, model weight, or model
configuration is mutated by applying or rolling back this plan.
