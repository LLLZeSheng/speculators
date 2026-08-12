# Global-Step Checkpointing Design

## Goal

Allow production training to save checkpoints at an exact optimizer-step interval and configure the GLM-5.2 MTP3 run to save every 1000 global steps.

## Design

Add an optional `checkpoint_steps` trainer argument. When set, it takes precedence over fractional or epoch-based `checkpoint_freq` for periodic checkpointing. The trainer evaluates the interval after a successful optimizer step by using the restored and incremented `global_step`, so resume and epoch boundaries do not reset the cadence.

The GLM launcher will pass `--checkpoint-steps 1000` and stop passing the approximate `--checkpoint-freq 0.005`. Existing callers that omit the new argument retain current behavior.

## Runtime Procedure

Keep the current run alive until tests pass. Then stop it, resume from the latest complete numeric checkpoint, restart TensorBoard on port 6006, and verify the resolved configuration and live processes.

## Tests

- Unit-test exact global-step triggering, non-triggering adjacent steps, precedence over `checkpoint_freq`, and resume alignment.
- Test CLI schema propagation into `TrainerConfig`.
- Test that the production launcher pins `--checkpoint-steps 1000` and removes the old fractional interval.
- Run focused trainer/config tests, formatting checks, and shell syntax validation before restarting training.
