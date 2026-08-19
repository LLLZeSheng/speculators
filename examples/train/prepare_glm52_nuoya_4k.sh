#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4}
NUOYA_SOURCE=${NUOYA_SOURCE:-/mnt/xds/mtp/spec_train/dataset/raw_conversions/average-2k-nuoya}
LONG_SOURCE=${LONG_SOURCE:-/mnt/xds/mtp/spec_train/dataset/raw_conversions/average-8k}
OUTPUT=${OUTPUT:-/mnt/xds/mtp/spec_train/dataset/hf/nuoya-first5-long1-4k}
NUM_PREPROCESSING_WORKERS=${NUM_PREPROCESSING_WORKERS:-8}
SAVE_WORKERS=${SAVE_WORKERS:-8}

# Select exactly the same six JSONL files as the 8K profile, but truncate and
# pack them independently at 4096 tokens. A separate output avoids mixing
# tokenized rows or hidden-state cache files produced with another sequence
# length.
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_glm52_nuoya_32k.py" \
    --model "$MODEL" \
    --source "$NUOYA_SOURCE" \
    --source-file-limit 5 \
    --source "$LONG_SOURCE" \
    --source-file-limit 1 \
    --jsonl-only \
    --output "$OUTPUT" \
    --seq-length 4096 \
    --num-preprocessing-workers "$NUM_PREPROCESSING_WORKERS" \
    --save-workers "$SAVE_WORKERS" \
    "$@"
