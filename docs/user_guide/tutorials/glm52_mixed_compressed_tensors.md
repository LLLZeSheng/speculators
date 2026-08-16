# GLM-5.2 Mixed W8A8/W4A8 `compressed-tensors` on vLLM Ascend

## Problem

Some GLM-5.2 W4A8C8 checkpoints declare:

```json
"quant_method": "compressed-tensors"
```

but encode different module bit widths inside one quantization scheme:

```json
"weights": {
  "num_bits": {
    "self_attn.q_a_proj": 8,
    "mlp.shared_experts": 8,
    "mlp.experts": 4
  },
  "strategy": "channel",
  "type": "int"
}
```

This describes the intended model correctly—ordinary linear layers and shared
experts use W8A8 while routed MoE experts use W4A8—but it is not the standard
`compressed-tensors` schema consumed by vLLM. A `QuantizationScheme` expects
one integer `weights.num_bits` value. Model loading therefore fails validation
when the entire dictionary is passed as that integer.

A separate failure occurs if the launcher adds `--quantization ascend`.
That flag selects the ModelSlim/Ascend-native loader and conflicts with a
checkpoint whose `quant_method` is `compressed-tensors`.

## Upstream support and the remaining metadata mismatch

vLLM Ascend added dynamic W4A8 MoE support for `compressed-tensors` in
[vllm-project/vllm-ascend#5889](https://github.com/vllm-project/vllm-ascend/pull/5889).
That operator support is necessary, but the checkpoint must still express
mixed precision as separate standard configuration groups.

The two concerns are distinct:

1. PR #5889 supplies the Ascend W4A8 dynamic MoE execution path.
2. This repository normalizes the checkpoint's non-standard dictionary-valued
   `num_bits` metadata so vLLM can select that path.

## Repository solution

The verifier launcher calls `scripts/prepare_mixed_quant_model.py` before
starting vLLM. The tool:

1. verifies that `quant_method` is `compressed-tensors`;
2. detects a dictionary-valued `weights.num_bits`;
3. splits each mixed group into standard groups such as `group_0_w4` and
   `group_0_w8`;
4. converts module suffixes into explicit regular-expression targets;
5. creates an immutable runtime model view; and
6. links all original weights and tokenizer assets instead of copying them.

For example, the routed expert entry becomes conceptually:

```json
"group_0_w4": {
  "targets": ["re:.*mlp\\.experts(?:\\..*)?$"],
  "weights": {
    "num_bits": 4,
    "strategy": "channel",
    "type": "int"
  }
}
```

The W8 targets are placed in a separate group with `num_bits: 8`. Activation
quantization fields are copied unchanged to both groups.

Runtime views are stored under:

```text
/kos_ulan/spec_train/runtime_models/glm52-w4a8c8/verifierN/
```

The final directory name includes a fingerprint of the source configuration
and weight metadata. A `.ready` marker is written only after every link and the
new `config.json` are complete. Each verifier has its own parent directory, so
nodes do not race when the shared filesystem is mounted with `nolock`.

## Safety properties

- The source model directory is never modified.
- Safetensors are neither rewritten nor duplicated.
- An ambiguous mixed group is rejected rather than guessed.
- An incomplete runtime directory is rejected rather than reused.
- Repeated starts reuse the same completed fingerprinted view.
- The transformation changes metadata only; it does not requantize weights.

Do not manually replace the dictionary with a single `4` or `8`. That would
apply one scheme to modules stored with different representations and can
produce a load failure or incorrect inference.

## Launcher configuration

For this checkpoint, use:

```bash
VERIFIER_QUANTIZATION_MODE=compressed-tensors
```

The generated vLLM command intentionally contains no `--quantization` option.
vLLM reads `quant_method=compressed-tensors` from the normalized runtime
configuration.

If the verifier is later replaced by a genuine ModelSlim/Ascend-native
checkpoint, set:

```bash
VERIFIER_QUANTIZATION_MODE=ascend
```

Only that mode adds:

```text
--quantization ascend
```

`VERIFIER_QUANTIZATION_MODE=auto` inspects the source configuration at runtime
and follows the same rule.

## Verification

Inspect the resolved command without loading weights:

```bash
DRY_RUN=1 bash examples/train/run_mtp_glm52_ascend_online_container.sh \
  /kos_ulan/spec_train/config/glm52-mtp3-4v4t.env
```

For a `compressed-tensors` checkpoint, confirm that the output:

- invokes `scripts/prepare_mixed_quant_model.py`;
- does not contain `--quantization ascend`; and
- uses the runtime model path when the real verifier starts.

At startup the application log prints a line similar to:

```text
[quantization] method=compressed-tensors metadata=normalized model=...-ct-split-...
```

Then check the normalized groups:

```bash
python - <<'PY'
import glob, json

paths = glob.glob(
    "/kos_ulan/spec_train/runtime_models/glm52-w4a8c8/"
    "verifier*/**/config.json",
    recursive=True,
)
for path in paths:
    config = json.load(open(path, encoding="utf-8"))
    groups = config["quantization_config"]["config_groups"]
    print(path, {name: group["weights"]["num_bits"] for name, group in groups.items()})
PY
```

Expected mixed groups contain scalar values, for example:

```text
{'group_0_w4': 4, 'group_0_w8': 8}
```

The original model's `config.json` must still contain the original dictionary;
that confirms the repair remained non-invasive.
