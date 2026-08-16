#!/usr/bin/env bash
# Prepare GLM-5.2 W4A8 mixed-quant metadata for vLLM Ascend.
# This script only creates a runtime model view; it never modifies the source
# model and never starts vLLM.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1}
RUNTIME_ROOT=${RUNTIME_ROOT:-/kos_ulan/spec_train/runtime_models/glm52-w4a8-mg13}

if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_mixed_quant_model.py"
    --model "$MODEL_PATH"
    --runtime-root "$RUNTIME_ROOT"
)

if (( $# == 1 )); then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

[[ -d $MODEL_PATH ]] || {
    echo "ERROR: model directory does not exist: $MODEL_PATH" >&2
    exit 1
}

output=$("${cmd[@]}")
mapfile -t result <<<"$output"
(( ${#result[@]} == 3 )) || {
    echo "ERROR: unexpected processor output: $output" >&2
    exit 1
}

runtime_model_path=${result[0]}
quant_method=${result[1]}
normalize_status=${result[2]}

[[ $quant_method == compressed-tensors ]] || {
    echo "ERROR: expected compressed-tensors, got: ${quant_method:-none}" >&2
    exit 1
}
[[ $normalize_status == unchanged || -f $runtime_model_path/.ready ]] || {
    echo "ERROR: runtime model is incomplete: $runtime_model_path" >&2
    exit 1
}

printf 'RUNTIME_MODEL_PATH=%s\n' "$runtime_model_path"
printf 'QUANT_METHOD=%s\n' "$quant_method"
printf 'NORMALIZE_STATUS=%s\n' "$normalize_status"
printf '\nUse RUNTIME_MODEL_PATH as the vLLM model path and remove --quantization ascend.\n'
