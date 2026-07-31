# GLM-5.2 Hidden-State Collection Command Design

## Goal

Add a manually invoked collection script at
`/mnt/paas/spec_train/collect_glm5.2_hidden_states.sh`. The script collects a
bounded number of hidden-state samples from an already running GLM-5.2 vLLM
service. It does not start, stop, or replace any model service.

## Interface

Run with the default sample limit of 10:

```bash
./collect_glm5.2_hidden_states.sh
```

Override the sample limit with one positive integer positional argument:

```bash
./collect_glm5.2_hidden_states.sh 1000
```

Any other argument form exits with status 2 and prints usage.

## Fixed Configuration

- Endpoint: `http://127.0.0.1:8000/v1`
- Required served model ID: `glm-5.2`
- Preprocessed dataset:
  `/mnt/paas/spec_train/hf_dataset_v2_glm52_8192`
- Output directory: `/mnt/paas/spec_train/hidden_states/glm5.2`
- Python: `/mnt/pass/miniconda3/bin/python3.13`
- Generator:
  `/mnt/paas/spec_train/speculators/scripts/data_generation_offline.py`
- Concurrency: 2
- Request timeout: 600 seconds
- Retries: 1
- Output validation and fail-fast behavior enabled

The script supplies both local source trees through `PYTHONPATH` so it works
without relying on the caller's active directory or editable installation.

## Preflight Checks

Before collection, the script:

1. Validates that the sample limit is a positive integer.
2. Calls the endpoint's `/models` API.
3. Parses the response and requires a model whose exact ID is `glm-5.2`.

An unavailable endpoint or a different served model causes a clear error and
nonzero exit before any generation request is sent. This prevents accidentally
sending GLM token IDs to another model service on port 8000.

## Collection and Resume Behavior

The script invokes `data_generation_offline.py` with `--max-samples` set to the
requested limit. Existing files named `hs_<index>.safetensors` are treated as
completed samples by the generator and skipped. Therefore rerunning the script
is resumable and does not overwrite previously collected samples.

The default invocation will perform no new work when `hs_0.safetensors` through
`hs_9.safetensors` already exist. The script must state this resume behavior in
its startup summary; it must not delete or overwrite files automatically.

## Errors and Output

The script uses Bash strict mode. It prints the endpoint, dataset, output path,
and sample limit before starting. Endpoint failures, model-ID mismatches,
invalid arguments, and generator failures return nonzero status. Successful
completion leaves the GLM service running.

## Verification

Implementation verification covers:

- `bash -n` syntax validation.
- Invalid and zero sample limits fail before network access.
- A dry local mock of `/v1/models` is not added; the real endpoint check remains
  an integration precondition.
- Shell inspection confirms the exact dataset, output, model ID, validation,
  timeout, retry, and fail-fast arguments.
- Running against an unavailable endpoint fails without creating or deleting
  hidden-state files.

