# GLM-5.2 DSpark 5-Layer Non-Causal Experiment

## Objective

Prepare a second training launcher that tests whether a deeper dense draft backbone
and bidirectional intra-block attention improve DSpark acceptance metrics relative
to the active 3-layer causal baseline.

## Scope

Create a new launcher at:

`/mnt/paas/spec_train/train_glm5.2_dspark_5l_noncausal_h200.sh`

The active launcher and training process remain unchanged. The new launcher is
prepared and validated but is not started.

## Controlled Changes

The experiment changes exactly two model settings from the baseline:

- `--num-layers 3` becomes `--num-layers 5`.
- `--sliding-window-non-causal` is enabled so positions within each synthetic
  draft block can attend bidirectionally.

All other training parameters remain identical, including the dataset, hidden
states, vocabulary mappings, block size 8, sliding window 2048, target layer IDs,
loss weights, optimizer, learning rate, scheduler, epoch count, and random seed.

## Isolation

Use a separate output directory:

`/mnt/paas/spec_train/output/dspark_glm52_nuoya_hs781890_h200_5l_noncausal`

Use a separate TensorBoard run name:

`glm52-dspark-nuoya-hs781890-h200-5l-noncausal`

The launcher must refuse to start if the experiment output directory already
contains checkpoint, metric-event, or training-log data. This prevents accidental
overwrite or unintended resume behavior.

## Validation

Before handoff:

- Run `bash -n` against the new launcher.
- Confirm the baseline launcher is byte-for-byte unchanged.
- Confirm the command contains 5 layers and the non-causal flag.
- Confirm the new output and run names do not overlap the active run.
- Do not launch `torchrun` or create experiment output artifacts during validation.

## Comparison Criteria

After the future experiment completes one validation epoch, compare it with the
baseline using `val/accept_len_epoch`, `val/accept_rate_epoch`, per-position
accuracy, validation loss, and end-to-end draft/verification latency. Raw loss
alone is not the success criterion.
