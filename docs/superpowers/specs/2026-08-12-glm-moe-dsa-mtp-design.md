# GLM MoE DSA MTP Support Design

## Goal

Add native GLM-5.2 `glm_moe_dsa` MTP fine-tuning support without changing the
existing Qwen MTP paths. The implementation must initialize from the verifier's
extra `model.layers.<num_hidden_layers>.*` NEXTN layer, train it with the existing
MTP objective, and stitch trained weights back into the verifier format.

## Architecture

`MTPConverter` will identify the native checkpoint layout from the verifier
configuration and weight keys. Existing `mtp.*` checkpoints keep their current
mapping. GLM checkpoints use `model.layers.<num_hidden_layers>.*`; their `enorm`,
`hnorm`, `eh_proj`, and `shared_head.norm` weights map to the generic MTP layer's
token norm, hidden norm, input projection, and final norm. Decoder attention and
MoE weights map directly after the existing expert fusion step.

The model registry will add Transformers' `GlmMoeDsaDecoderLayer`, RMSNorm, and
rotary embedding. A GLM-specific MTP wrapper selects the verifier's final sparse
layer configuration while retaining the generic recursive MTP forward and loss.

Stitching will inspect the verifier `model_type`. For `glm_moe_dsa`, generic MTP
keys are mapped back to the extra layer index and fused MoE tensors are expanded
to the checkpoint's per-expert layout. Other model types retain the current Qwen
mapping.

## Compatibility And Safety

- Require Transformers to expose `glm_moe_dsa`; imports remain optional.
- Preserve all existing Qwen key mappings and tests.
- Support exactly one native MTP layer, recursively used for MTP3 or MTP5.
- Reject GLM checkpoints that declare native MTP but lack the extra layer.
- Use only CPU-sized synthetic configs in unit tests.
- Do not modify or import code from `/mnt/paas/spec_train/speculators`.

## Verification

Unit tests cover native prefix resolution, bidirectional key mapping, expert
fusion, GLM model registration, a small forward pass, and stitch behavior. The
real GLM-5.2 checkpoint is then used for a read-only extraction/config smoke test;
no training process or GPU is started.
