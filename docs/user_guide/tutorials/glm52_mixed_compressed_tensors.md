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

## Related upstream work

[vllm-project/vllm-ascend#5889](https://github.com/vllm-project/vllm-ascend/pull/5889)
adds dynamic W4A8 MoE support for compressed-tensors. That PR is useful for
checkpoints genuinely stored in compressed-tensors format, but it does not by
itself convert this MG13 ModelSlim checkpoint or fill its MTP metadata. The
runtime-view preparation above is still required for this exact model.
