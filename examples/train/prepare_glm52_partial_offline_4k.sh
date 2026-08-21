#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_DATA=${SOURCE_DATA:-/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k}
HIDDEN_STATES=${HIDDEN_STATES:-/mnt/xds/mtp/spec_train/hidden_states/glm52-w4a8-mg13-offline-4k}
OUTPUT_DATA=${OUTPUT_DATA:-/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k-partial-offline}
VALIDATE_SAMPLES=${VALIDATE_SAMPLES:-32}

usage() {
    cat <<EOF
Usage: $(basename "$0") [build|verify|build-and-verify]

Environment overrides:
  SOURCE_DATA       Original Hugging Face dataset ($SOURCE_DATA)
  HIDDEN_STATES     Partial hs_*.safetensors cache ($HIDDEN_STATES)
  OUTPUT_DATA       Filtered dataset to create ($OUTPUT_DATA)
  VALIDATE_SAMPLES  Files to validate; -1 means all, 0 skips ($VALIDATE_SAMPLES)

No hidden-state files are copied. The filtered dataset stores source_index so
training continues to read the original hs_<index>.safetensors files.
EOF
}

build() {
    "$PYTHON_BIN" "$REPO_ROOT/scripts/build_partial_offline_dataset.py" \
        --data "$SOURCE_DATA" \
        --hidden-states "$HIDDEN_STATES" \
        --output "$OUTPUT_DATA" \
        --validate-samples "$VALIDATE_SAMPLES"
}

verify() {
    "$PYTHON_BIN" "$REPO_ROOT/scripts/check_offline_hidden_states.py" \
        --data "$OUTPUT_DATA" \
        --hidden-states "$HIDDEN_STATES" \
        --validate-samples "$VALIDATE_SAMPLES"
}

COMMAND=${1:-build-and-verify}
case "$COMMAND" in
    build)
        build
        ;;
    verify)
        verify
        ;;
    build-and-verify)
        build
        verify
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
