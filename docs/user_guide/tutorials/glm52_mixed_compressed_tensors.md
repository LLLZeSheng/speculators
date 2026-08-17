# GLM-5.2 MG13 W4A8 on vLLM Ascend

## Final diagnosis

`/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1` labels its quantization method as
`compressed-tensors`, but its actual checkpoint contains Ascend ModelSlim
metadata and auxiliary tensors. It must be served with the ModelSlim loader:

```text
--quantization ascend
```

Changing only `config.json` is insufficient. The runtime
`quant_model_description.json` must also describe every module instantiated by
vLLM, including the native MTP layer.

The successful non-destructive runtime view is:

```text
/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
```

It was produced from the untouched source by:

```bash
bash examples/train/prepare_glm52_w4a8_mg13_model.sh \
  /mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1 \
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
```

## What the preparation script changes

The script creates a new directory and never edits the source model. It:

1. links the original safetensors and large immutable assets;
2. copies the small configuration and quantization metadata files;
3. removes the misleading embedded `compressed-tensors`
   `quantization_config` from the runtime `config.json`;
4. keeps `quant_model_description.json` as the ModelSlim source of truth;
5. finds native MTP layers from the safetensors index instead of assuming a
   hard-coded layer number;
6. marks MTP parameters missing from ModelSlim metadata as `FLOAT` only when
   those parameters really exist in the checkpoint;
7. writes a manifest and an atomic `.ready` marker only after validation.

The runtime view is small because weights are symlinks. Both the runtime path
and its source path must therefore be visible inside the container.

## Why the earlier attempts failed

The errors exposed three separate mismatches:

- `compressed-tensors` versus CLI `ascend`: the configuration selected two
  different quantization loaders.
- `layers.3.mlp.experts.w2_scale_bias` and `w2_weight_offset`: the
  compressed-tensors fused expert mapping did not match the ModelSlim tensor
  representation. Removing individual auxiliary entries only revealed the
  next mismatch and was not a valid conversion.
- `model.layers.78.self_attn.q_a_proj.weight`: vLLM creates the native MTP
  decoder layer, but the ModelSlim description originally covered only base
  layers. The v4 preparation adds valid `FLOAT` metadata for the real MTP
  tensors.

Do not reuse the experimental directories `v1-ct-split-*`,
`v1-ascend-modelslim`, `v1-ascend-modelslim-v2`, or
`v1-ascend-modelslim-v3`. Do not delete `scale_bias` or `weight_offset` from
the original model, and restore any local vLLM patch that filtered those
tensors.

## Launch contract

Use these settings:

```bash
VERIFIER_MODEL_PATH=/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4
VERIFIER_SOURCE_MODEL_PATH=/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1
VERIFIER_QUANTIZATION_MODE=ascend
```

The resolved vLLM command must contain `--quantization ascend`. It must not run
`prepare_mixed_quant_model.py` and must not create another `-ct-split-*` view.

Before launch:

```bash
test -f "$VERIFIER_MODEL_PATH/.ready"
test -f "$VERIFIER_MODEL_PATH/quant_model_description.json"
cat "$VERIFIER_MODEL_PATH/runtime_model_manifest.json"
```

The manifest should report one or more MTP float entries. A manifest with
`"mtp_float_entries_added": 0` is an obsolete pre-v4 view for this model.

## Audit native MTP weight mapping

Successful model construction proves only that ModelSlim accepted the layer
description. It does not prove that the native MTP embedding, transformer
block, and logits head received the intended checkpoint values. Run the static
auditor inside the serving image:

```bash
cd /kos_ulan/spec_train/speculators
python scripts/check_glm52_mtp_mapping.py \
  /mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4 \
  --check-values \
  --json-output /tmp/glm52-mtp-mapping.json
```

The command checks:

- `num_hidden_layers` and `num_nextn_predict_layers` against checkpoint keys;
- all MTP `.weight` entries against `quant_model_description.json`;
- representative checkpoint-to-runtime rewrites, including insertion of
  `.mtp_block` for transformer parameters;
- the layer-local `enorm`, `hnorm`, `eh_proj`, shared norm, embedding, and LM
  head tensors;
- representative tensor samples for finite, non-degenerate values; and
- the installed vLLM-Ascend `patch_deepseek_mtp.py` for generic-loader or
  explicit shared-weight handling.

Pay special attention to:

```text
model.layers.78.embed_tokens.weight
model.layers.78.shared_head.head.weight
```

Native MTP shares these parameters conceptually with the target model. If the
layer-local copies are absent, the runtime loader must explicitly bind
`model.embed_tokens.weight` and `lm_head.weight`. A static
`PASS_WITH_WARNINGS` means the files are internally consistent but this
runtime binding still needs confirmation. `FAIL` means the checkpoint or
ModelSlim description is already inconsistent and serving should not be used
for acceptance measurements.

The final runtime evidence remains the server startup log. Check it with:

```bash
grep -Ei \
  'MTP draft model loaded|MTP speculative decoding layer|Following weights were not initialized|shared_head|embed_tokens|layers\.78|missing|not loaded' \
  SERVER_LOG | tail -500
```

An absence of errors is not sufficient on versions that validate only that at
least one parameter from each MTP layer was loaded. The auditor deliberately
reports the critical shared parameters separately for this reason. See the
upstream GLM missing-shared-weight report:
[vllm-project/vllm-ascend#6754](https://github.com/vllm-project/vllm-ascend/issues/6754).

### Fix missing shared embedding/head routing

If the audit reports both layer-local shared weights as missing and native MTP
acceptance remains exactly zero, patch the installed vLLM loader inside the
serving container:

```bash
python scripts/patch_vllm_glm52_mtp_shared_weights.py
python scripts/patch_vllm_glm52_mtp_shared_weights.py --check
```

The patch routes `model.embed_tokens.weight` and `lm_head.weight` before the
generic MTP loader discards non-layer keys. It also adds a mandatory post-load
check, so an uninitialized draft embedding or logits head fails startup instead
of silently producing zero acceptance. The script is idempotent and creates:

```text
/vllm-workspace/vllm/vllm/model_executor/models/deepseek_mtp.py.before-glm-mtp-shared-weight-fix
```

Stop all old vLLM workers and start the service again after applying it. To
restore the original source:

```bash
python scripts/patch_vllm_glm52_mtp_shared_weights.py --restore
```

For an image with a different source location, pass the explicit file:

```bash
python scripts/patch_vllm_glm52_mtp_shared_weights.py \
  --target /path/to/vllm/model_executor/models/deepseek_mtp.py
```

#### Exact runtime mapping and ownership

The mapping is Python loader behavior, not model metadata. The patch modifies
`DeepSeekMTP.load_weights()` in:

```text
/vllm-workspace/vllm/vllm/model_executor/models/deepseek_mtp.py
```

It handles the target-model keys before the generic loader rejects keys which
are not scoped under an MTP layer:

```text
model.embed_tokens.weight
  -> MTP model.embed_tokens.weight

lm_head.weight
  -> model.layers.<mtp_start_layer>.shared_head.head.weight
```

For this GLM-5.2 checkpoint, `<mtp_start_layer>` is 78. The implementation does
not hard-code 78; it reads `self.model.mtp_start_layer_idx`, so the same patch
can be reused when the MTP layer starts at another index.

Ascend defines `AscendDeepSeekMTP` in:

```text
/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_mtp.py
```

That class inherits the vLLM `DeepSeekMTP` loading behavior, so the routing fix
belongs in the vLLM file above. Inspect the installed change with:

```bash
grep -n -A35 -B5 GLM_MTP_SHARED_WEIGHT_FIX_V1 \
  /vllm-workspace/vllm/vllm/model_executor/models/deepseek_mtp.py
```

#### Model configuration impact

The shared-weight routing patch does not modify checkpoint tensors,
`config.json`, or `quant_model_description.json`.

The earlier non-destructive ModelSlim preparation is a separate operation:

- the original `/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1` remains untouched;
- the `v1-ascend-modelslim-v4` runtime view has its own copied `config.json`,
  with the conflicting embedded `compressed-tensors` block removed; and
- the runtime view has an augmented `quant_model_description.json` that marks
  real, unquantized layer-78 MTP checkpoint weights as `FLOAT`.

#### PD-disaggregated serving

Apply the loader patch to every independent container which starts vLLM with a
native MTP `--speculative-config`. This normally includes every decode
container. It also includes a prefill container if its command retains the MTP
option. A prefill container which never constructs an MTP drafter does not need
the patch.

Container filesystems are commonly independent. Run this check in every
applicable P/D container, and repeat it after replacing or rebuilding an image:

```bash
python scripts/patch_vllm_glm52_mtp_shared_weights.py --check
```

The required result is `PATCH_STATUS=applied`. No additional model-config
rewrite is required for PD serving. All nodes must still use the same runtime
model view and `--quantization ascend`. Completely stop old workers before
restarting, because an already running Python process retains the old loader
code in memory.

## Related upstream work

[vllm-project/vllm-ascend#5889](https://github.com/vllm-project/vllm-ascend/pull/5889)
adds dynamic W4A8 MoE support for compressed-tensors. That PR is useful for
checkpoints genuinely stored in compressed-tensors format, but it does not by
itself convert this MG13 ModelSlim checkpoint or fill its MTP metadata. The
runtime-view preparation above is still required for this exact model.
