#!/usr/bin/env bash
# Four-node GLM-5.2 online MTP3 training on Ascend 910B3.
# Run one role per node. See docs/user_guide/tutorials/train_mtp_ascend_online.md.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROLE=${ROLE:-}
DRY_RUN=${DRY_RUN:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

VERIFIER_MODEL_PATH=${VERIFIER_MODEL_PATH:-/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1}
MTP_INIT_MODEL_PATH=${MTP_INIT_MODEL_PATH:-/mnt/xds/sfs/GLM-5.2}
DATA_PATH=${DATA_PATH:-/mnt/xds/sfs/datasets/glm52-dspark-train}
HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH:-/mnt/xds/sfs/spec_train/online_hidden_states/glm52-w4a8}
MTP_DRAFT_PATH=${MTP_DRAFT_PATH:-/mnt/xds/sfs/spec_train/initial/glm52-bf16-mtp3}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/xds/sfs/spec_train/checkpoints/glm52-w4a8-mtp3}
LOG_ROOT=${LOG_ROOT:-/mnt/xds/sfs/spec_train/logs/glm52-w4a8-mtp3}

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm52-w4a8-verifier}
VERIFIER_HOST=${VERIFIER_HOST:-}
VERIFIER_BIND_HOST=${VERIFIER_BIND_HOST:-0.0.0.0}
VERIFIER_PORT=${VERIFIER_PORT:-8000}
VERIFIER_TP_SIZE=${VERIFIER_TP_SIZE:-16}
VERIFIER_MAX_MODEL_LEN=${VERIFIER_MAX_MODEL_LEN:-8193}
TARGET_LAYER_ID=${TARGET_LAYER_ID:-78}

NNODES=${NNODES:-3}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NODE_RANK=${NODE_RANK:-}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29500}
RUN_ID=${RUN_ID:-glm52-w4a8-mtp3}

TOTAL_SEQ_LEN=${TOTAL_SEQ_LEN:-4096}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
STEP_WEIGHT_BETA=${STEP_WEIGHT_BETA:-0.6}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-1000}
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-0.98}
NUM_WORKERS=${NUM_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
MAX_STEPS=${MAX_STEPS:-}
RUN_NAME=${RUN_NAME:-glm52-w4a8-ascend-mtp3}

SMOKE_SAMPLES=${SMOKE_SAMPLES:-64}
SMOKE_RUN_ID=${SMOKE_RUN_ID:-}
SMOKE_DATA_PATH=${SMOKE_DATA_PATH:-/mnt/xds/sfs/spec_train/smoke/glm52-mtp3-tokens-64-${SMOKE_RUN_ID:-unset}}
MTP_PREPARE_TIMEOUT=${MTP_PREPARE_TIMEOUT:-3600}

ASCEND_DEVICES=${ASCEND_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_DEVICES}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local name=$1
    [[ -n ${!name:-} ]] || fail "$name must be set for ROLE=$ROLE"
}

print_cmd() {
    printf '[command]'
    printf ' %q' "$@"
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [[ $DRY_RUN != 1 ]]; then
        "$@"
    fi
}

CHILD_PID=
forward_signal() {
    if [[ -n $CHILD_PID ]]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
}
trap forward_signal INT TERM

run_logged() {
    local log_file=$1
    shift
    print_cmd "$@"
    if [[ $DRY_RUN == 1 ]]; then
        return
    fi
    mkdir -p "$(dirname -- "$log_file")"
    "$@" > >(tee -a "$log_file") 2>&1 &
    CHILD_PID=$!
    set +e
    wait "$CHILD_PID"
    local status=$?
    set -e
    CHILD_PID=
    return "$status"
}

validate_role() {
    case "$ROLE" in
        preflight | verifier | trainer) ;;
        smoke) require_value SMOKE_RUN_ID ;;
        *) fail "ROLE must be one of: preflight, verifier, trainer, smoke" ;;
    esac
}

validate_trainer_topology() {
    require_value VERIFIER_HOST
    require_value MASTER_ADDR
    require_value NODE_RANK
    [[ $NODE_RANK =~ ^[0-9]+$ ]] || fail "NODE_RANK must be an integer"
    ((NODE_RANK >= 0 && NODE_RANK < NNODES)) || \
        fail "NODE_RANK must be in [0, $((NNODES - 1))]"
}

validate_context_window() {
    local input_length=$1
    [[ $input_length =~ ^[0-9]+$ && $VERIFIER_MAX_MODEL_LEN =~ ^[0-9]+$ ]] || \
        fail "TOTAL_SEQ_LEN and VERIFIER_MAX_MODEL_LEN must be integers"
    ((input_length < VERIFIER_MAX_MODEL_LEN)) || \
        fail "VERIFIER_MAX_MODEL_LEN must exceed the online input length by at least 1"
}

show_config() {
    cat <<EOF
Resolved Ascend MTP3 configuration:
  ROLE=$ROLE DRY_RUN=$DRY_RUN
  VERIFIER_MODEL_PATH=$VERIFIER_MODEL_PATH
  MTP_INIT_MODEL_PATH=$MTP_INIT_MODEL_PATH
  DATA_PATH=$DATA_PATH
  HIDDEN_STATES_PATH=$HIDDEN_STATES_PATH
  MTP_DRAFT_PATH=$MTP_DRAFT_PATH
  OUTPUT_PATH=$OUTPUT_PATH
  LOG_ROOT=$LOG_ROOT
  VERIFIER_HOST=${VERIFIER_HOST:-<unset>} VERIFIER_PORT=$VERIFIER_PORT
  MASTER_ADDR=${MASTER_ADDR:-<unset>} MASTER_PORT=$MASTER_PORT
  NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=${NODE_RANK:-<unset>}
  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
EOF
}

preflight_args() {
    local required_devices=$NPROC_PER_NODE
    if [[ $ROLE == verifier ]]; then
        required_devices=$VERIFIER_TP_SIZE
    fi
    PREFLIGHT_CMD=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/preflight_ascend_mtp.py"
        --bf16-model "$MTP_INIT_MODEL_PATH"
        --verifier-model "$VERIFIER_MODEL_PATH"
        --data-path "$DATA_PATH"
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --required-devices "$required_devices"
    )
    if [[ $DRY_RUN == 1 ]]; then
        PREFLIGHT_CMD+=(--skip-device-check)
    fi
}

run_preflight() {
    preflight_args
    run_cmd "${PREFLIGHT_CMD[@]}" "$@"
}

run_verifier() {
    run_preflight --require-vllm
    local log_file="$LOG_ROOT/verifier/verifier.log"
    local cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/launch_vllm.py"
        "$VERIFIER_MODEL_PATH"
        --hidden-states-backend file
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --target-layer-ids "$TARGET_LAYER_ID"
        --
        --host "$VERIFIER_BIND_HOST"
        --port "$VERIFIER_PORT"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$VERIFIER_TP_SIZE"
        --max-model-len "$VERIFIER_MAX_MODEL_LEN"
        --trust-remote-code
    )
    if [[ $DRY_RUN == 1 ]]; then
        printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    fi
    run_logged "$log_file" "${cmd[@]}"
}

wait_for_marker() {
    local marker=$1
    local description=$2
    local waited=0
    while [[ ! -f $marker ]]; do
        ((waited < MTP_PREPARE_TIMEOUT)) || \
            fail "timed out waiting for $description: $marker"
        sleep 5
        waited=$((waited + 5))
    done
}

prepare_mtp_draft() {
    local marker="$MTP_DRAFT_PATH/.ready"
    if [[ $DRY_RUN == 1 ]]; then
        print_cmd flock "${MTP_DRAFT_PATH}.lock" "$PYTHON_BIN" -m speculators \
            convert "$MTP_INIT_MODEL_PATH" --algorithm mtp \
            --verifier "$MTP_INIT_MODEL_PATH" --output-path "$MTP_DRAFT_PATH" \
            --algorithm-kwargs '{"num_speculative_steps":3}'
        return
    fi
    if [[ $NODE_RANK == 0 ]]; then
        mkdir -p "$(dirname -- "$MTP_DRAFT_PATH")"
        (
            flock -x 9
            if [[ -f $marker ]]; then
                exit 0
            fi
            [[ ! -e $MTP_DRAFT_PATH ]] || \
                fail "$MTP_DRAFT_PATH exists without .ready; inspect it manually"
            local temporary="${MTP_DRAFT_PATH}.tmp.${HOSTNAME}.$$"
            trap 'rm -rf -- "$temporary"' EXIT
            "$PYTHON_BIN" -m speculators convert "$MTP_INIT_MODEL_PATH" \
                --algorithm mtp \
                --verifier "$MTP_INIT_MODEL_PATH" \
                --output-path "$temporary" \
                --algorithm-kwargs '{"num_speculative_steps":3}'
            mv -- "$temporary" "$MTP_DRAFT_PATH"
            touch "$marker"
            trap - EXIT
        ) 9>"${MTP_DRAFT_PATH}.lock"
    else
        wait_for_marker "$marker" "BF16 native MTP initialization"
    fi
}

prepare_smoke_data() {
    local marker="$SMOKE_DATA_PATH/.ready"
    printf 'Prepare smoke dataset (%s samples): %s\n' \
        "$SMOKE_SAMPLES" "$SMOKE_DATA_PATH"
    local code='from datasets import load_from_disk; import os, sys; src, dst, count = sys.argv[1], sys.argv[2], int(sys.argv[3]); data = load_from_disk(src); tmp = dst + ".tmp." + str(os.getpid()); data.select(range(min(count, len(data)))).save_to_disk(tmp); os.rename(tmp, dst); open(os.path.join(dst, ".ready"), "w").close()'
    if [[ $DRY_RUN == 1 ]]; then
        print_cmd "$PYTHON_BIN" -c "$code" "$DATA_PATH" "$SMOKE_DATA_PATH" \
            "$SMOKE_SAMPLES"
        return
    fi
    if [[ $NODE_RANK == 0 ]]; then
        mkdir -p "$(dirname -- "$SMOKE_DATA_PATH")"
        (
            flock -x 9
            if [[ ! -f $marker ]]; then
                [[ ! -e $SMOKE_DATA_PATH ]] || \
                    fail "$SMOKE_DATA_PATH exists without .ready; inspect it manually"
                "$PYTHON_BIN" -c "$code" "$DATA_PATH" "$SMOKE_DATA_PATH" \
                    "$SMOKE_SAMPLES"
            fi
        ) 9>"${SMOKE_DATA_PATH}.lock"
    else
        wait_for_marker "$marker" "smoke dataset"
    fi
}

run_trainer() {
    local mode=$1
    validate_trainer_topology
    run_preflight --endpoint "http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"
    prepare_mtp_draft

    local effective_data=$DATA_PATH
    local effective_output=$OUTPUT_PATH
    local effective_log_root=$LOG_ROOT
    local effective_seq_len=$TOTAL_SEQ_LEN
    local effective_max_steps=$MAX_STEPS
    local effective_run_name=$RUN_NAME
    if [[ $mode == smoke ]]; then
        prepare_smoke_data
        effective_data=$SMOKE_DATA_PATH
        effective_output="${OUTPUT_PATH}-smoke-${SMOKE_RUN_ID}"
        effective_log_root="${LOG_ROOT}-smoke-${SMOKE_RUN_ID}"
        effective_seq_len=1024
        effective_max_steps=2
        effective_run_name="${RUN_NAME}-smoke-${SMOKE_RUN_ID}"
        if [[ $DRY_RUN != 1 && -e $effective_output ]]; then
            fail "smoke output already exists; choose a fresh SMOKE_RUN_ID: $effective_output"
        fi
    fi
    validate_context_window "$effective_seq_len"

    local cmd=(
        "$TORCHRUN_BIN"
        --nnodes "$NNODES"
        --nproc-per-node "$NPROC_PER_NODE"
        --node-rank "$NODE_RANK"
        --master-addr "$MASTER_ADDR"
        --master-port "$MASTER_PORT"
        "$REPO_ROOT/scripts/train.py"
        --verifier-name-or-path "$MTP_INIT_MODEL_PATH"
        --generation-model-name-or-path "$SERVED_MODEL_NAME"
        --from-pretrained "$MTP_DRAFT_PATH"
        --trust-remote-code
        --data-path "$effective_data"
        --hidden-states-backend file
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --vllm-endpoint "http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"
        --on-missing generate
        --on-generate delete
        --force-generate
        --on-generation-error raise
        --train-data-ratio "$TRAIN_DATA_RATIO"
        --save-path "$effective_output"
        --log-dir "$effective_log_root/metrics"
        --logger tensorboard
        --run-name "$effective_run_name"
        --speculator-type mtp
        --num-speculative-steps 3
        --step-weight-beta "$STEP_WEIGHT_BETA"
        --total-seq-len "$effective_seq_len"
        --hidden-states-dtype bfloat16
        --noise-std 0
        --optimizer adamw
        --lr "$LR"
        --weight-decay "$WEIGHT_DECAY"
        --scheduler-type cosine
        --scheduler-warmup-ratio 0.03
        --epochs "$EPOCHS"
        --checkpoint-steps "$CHECKPOINT_STEPS"
        --num-workers "$NUM_WORKERS"
        --prefetch-factor "$PREFETCH_FACTOR"
        --log-freq 10
        --fsdp-shard
    )
    if [[ -n $effective_max_steps ]]; then
        cmd+=(--max-steps "$effective_max_steps")
    fi
    if [[ $mode == smoke ]]; then
        cmd+=(--no-resume-from-checkpoint)
    fi
    local log_file="$effective_log_root/trainer-node${NODE_RANK}.log"
    run_logged "$log_file" "${cmd[@]}"
}

validate_role
show_config

case "$ROLE" in
    preflight) run_preflight ;;
    verifier) run_verifier ;;
    trainer) run_trainer trainer ;;
    smoke) run_trainer smoke ;;
esac
