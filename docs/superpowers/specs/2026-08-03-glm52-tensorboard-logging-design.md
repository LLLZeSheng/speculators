# GLM-5.2 DSpark TensorBoard Logging Design

## Goal

Ensure every training run launched by
`/mnt/paas/spec_train/train_glm5.2_dspark_h200.sh` uses the dedicated
`speculators_venv` environment and records structured training metrics in a local
TensorBoard event stream.

## Scope

The change installs only TensorBoard. W&B, Trackio, and MLflow remain disabled.
It updates the existing launcher rather than adding a second entry point, and it
does not automatically start the full eight-GPU training job.

## Runtime and Dependencies

TensorBoard is installed into
`/mnt/paas/spec_train/speculators_venv`. The launcher defaults change from
the obsolete `/mnt/pass/miniconda3` runtime to:

- `/mnt/paas/spec_train/speculators_venv/bin/python`
- `/mnt/paas/spec_train/speculators_venv/bin/torchrun`

The current production launcher uses fixed paths rather than environment-variable
overrides, so the update changes only those two fixed assignments.

## Launcher Integration

The existing `train_cmd` gains `--logger tensorboard`. Its existing arguments
already define:

- `--log-dir /mnt/paas/spec_train/output/dspark_glm52_nuoya_100k_h200/metrics`
- `--run-name glm52-dspark-nuoya-100k-h200`

The Speculators `TensorBoardHandler` therefore writes event files beneath:

`/mnt/paas/spec_train/output/dspark_glm52_nuoya_100k_h200/metrics/glm52-dspark-nuoya-100k-h200/`

Only distributed rank 0 writes metric events, preventing duplicate eight-rank
series. Human-readable stdout/stderr continues to be appended to
`/mnt/paas/spec_train/output/dspark_glm52_nuoya_100k_h200/train.log` through the
existing `tee` pipeline.

## Validation

Validation does not launch the full training job. It consists of:

1. Checking the virtual environment dependency graph after TensorBoard install.
2. Importing `tensorboard` and `torch.utils.tensorboard.SummaryWriter`.
3. Running a temporary shadow copy of the launcher with a recording `torchrun`
   stub and confirming its real preflight passes and the recorded command includes
   `--logger tensorboard`; the production launcher itself is not invoked to start
   training.
4. Exercising Speculators' real `setup_metric_logger` path with a temporary run
   name, emitting a scalar at a known step, and closing the handlers.
5. Reading the generated event file with TensorBoard's event accumulator and
   asserting that the expected scalar tag, value, and step are present.
6. Confirming the repository's pre-existing tracked and untracked changes remain
   untouched.

## Failure Handling

Dependency downloads use the verified Aliyun PyPI mirror because direct PyPI is
unreachable or slow from this host. If installation or event validation fails,
the full training job is not started and the original launcher behavior remains
inspectable through Git diff.
