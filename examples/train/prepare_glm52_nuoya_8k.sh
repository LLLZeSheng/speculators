#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4}
NUOYA_SOURCE=${NUOYA_SOURCE:-/kos_ulan/lzs/spec_train/dataset/raw_conversions/average-2k-nuoya}
LONG_SOURCE=${LONG_SOURCE:-/kos_ulan/lzs/spec_train/dataset/raw_conversions/average-8k}
OUTPUT=${OUTPUT:-/kos_ulan/lzs/spec_train/dataset/hf/nuoya-first5-long1-8k}
NUM_PREPROCESSING_WORKERS=${NUM_PREPROCESSING_WORKERS:-8}
SAVE_WORKERS=${SAVE_WORKERS:-8}

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_glm52_nuoya_32k.py" \
    --model "$MODEL" \
    --source "$NUOYA_SOURCE" \
    --source-file-limit 5 \
    --source "$LONG_SOURCE" \
    --source-file-limit 1 \
    --jsonl-only \
    --output "$OUTPUT" \
    --seq-length 8192 \
    --num-preprocessing-workers "$NUM_PREPROCESSING_WORKERS" \
    --save-workers "$SAVE_WORKERS" \
    "$@"
