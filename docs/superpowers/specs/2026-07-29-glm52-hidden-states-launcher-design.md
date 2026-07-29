# GLM-5.2 Hidden-State Launcher Design

## Goal

Add a dedicated shell script that starts the local GLM-5.2 verifier through
`scripts/launch_vllm.py`, enabling hidden-state extraction without changing the
existing baseline GLM-5.2 launcher.

## Inputs and Constraints

- Script path: `/mnt/paas/spec_train/start_glm5.2_hidden_states.sh`
- Model path: `/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1`
- Python runtime: `/mnt/pass/miniconda3/bin/python3.13`
- Hidden-state launcher:
  `/mnt/paas/spec_train/speculators/scripts/launch_vllm.py`
- Hidden-state output:
  `/mnt/paas/spec_train/hidden_states/glm5.2`
- Auxiliary target layers: `8`, `23`, `39`, `55`, and `70`
- Final target layer: `78`, appended by `launch_vllm.py`
- GPU topology: tensor parallel size 8 with expert parallelism across GPUs
  `0,1,2,3,4,5,6,7`
- The existing `/mnt/paas/spec_train/start_glm5.2.sh` remains unchanged.

## Selected Approach

Create a separate, purpose-specific launcher next to the existing baseline
launcher. This keeps normal serving and hidden-state collection as explicit
operations, avoids a mode switch that could be enabled accidentally, and keeps
the command easy to audit against `test.sh`.

The script will use `set -euo pipefail`, export the same CUDA and logging
environment as the baseline launcher, create the hidden-state directory, and
then replace itself with the Python launcher process via `exec`. It accepts an
optional `--dry-run` argument and forwards it to `launch_vllm.py` before the
vLLM argument separator.

## Command Construction and Data Flow

The shell script passes model and extraction arguments to `launch_vllm.py`:

- `--hidden-states-path /mnt/paas/spec_train/hidden_states/glm5.2`
- `--target-layer-ids 8 23 39 55 70`

`launch_vllm.py` reads the local GLM configuration, confirms that the model has
78 hidden layers, appends layer 78, and constructs the vLLM
`extract_hidden_states` speculative configuration. It also configures the file
backend as a KV producer whose shared storage path is the requested output
directory.

Arguments after `--` are forwarded to vLLM. They preserve the known-working
GLM-5.2 configuration from `start_glm5.2.sh`:

- Bind `0.0.0.0:8000` and serve the name `glm-5.2`.
- Use a maximum model length of 20480 and BF16 KV cache.
- Use TP=8, expert parallelism, the FlashInfer NVLink one-sided all-to-all
  backend, and the GLM sparse MLA attention backend.
- Retain GLM reasoning and tool-call parsers, automatic tool choice, trusted
  remote code, multiprocessing execution, and automatic MoE backend selection.

The Python launcher automatically disables chunked prefill, as required by its
hidden-state extraction flow. Requests sent to the resulting OpenAI-compatible
endpoint cause the verifier to write captured hidden states to the configured
directory.

## Error Handling

- Strict Bash mode stops on an unset variable or failed setup command.
- Any argument other than a single optional `--dry-run` prints usage and exits
  with status 2 before model launch.
- `mkdir -p` makes output-directory setup idempotent and fails before model
  launch if the directory cannot be created.
- Absolute paths avoid dependence on the caller's current working directory.
- `exec` preserves vLLM's exit status and signal behavior.
- The script does not stop an existing service or delete prior hidden-state
  files; port/GPU conflicts remain visible as vLLM startup errors.

## Verification

Verification will not load the model or occupy GPUs:

1. Run `bash -n` on the new script.
2. Invoke the new script itself with `--dry-run`, exercising its real argument
   order and paths without starting vLLM.
3. Confirm the printed command contains:
   - the GLM-5.2 model path;
   - hidden-state layers `[8, 23, 39, 55, 70, 78]`;
   - the file connector output path;
   - TP=8 and the existing GLM-specific vLLM options;
   - `--no-enable-chunked-prefill`.
4. Confirm the existing baseline launcher still matches its recorded SHA-256
   digest.

Actually starting the eight-GPU model and issuing collection requests is out of
scope for this script-writing task.
