#!/usr/bin/env bash
# Eight-node GLM-5.2 online MTP3 training on Ascend 910C / Atlas A3.
# Topology: four 16-NPU verifier nodes plus four 16-NPU trainer nodes.
# Run one role per node. See docs/user_guide/tutorials/train_mtp_ascend_online.md.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

ROLE=${ROLE:-}
DRY_RUN=${DRY_RUN:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

SHARED_ROOT=${SHARED_ROOT:-/kos_ulan}
VERIFIER_MODEL_PATH=${VERIFIER_MODEL_PATH:-/mnt/xds/dev/s00838505/GLM-5.2-w4a8c8}
# The W4A8C8 verifier is inference-only. This path must point to an unquantized
# GLM-5.2 checkpoint containing the native MTP layer.
MTP_INIT_MODEL_PATH=${MTP_INIT_MODEL_PATH:-${SHARED_ROOT}/models/GLM-5.2}
DATA_PATH=${DATA_PATH:-${SHARED_ROOT}/datasets/glm52-dspark-train}
HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH:-${SHARED_ROOT}/spec_train/online_hidden_states/glm52-w4a8c8}
MTP_DRAFT_PATH=${MTP_DRAFT_PATH:-${SHARED_ROOT}/spec_train/initial/glm52-bf16-mtp3}
OUTPUT_PATH=${OUTPUT_PATH:-${SHARED_ROOT}/spec_train/checkpoints/glm52-w4a8c8-mtp3}
LOG_ROOT=${LOG_ROOT:-${SHARED_ROOT}/spec_train/logs/glm52-w4a8c8-mtp3}
VERIFIER_METADATA_PATH=${VERIFIER_METADATA_PATH:-${SHARED_ROOT}/spec_train/metadata/glm52-w4a8c8}
VERIFIER_RUNTIME_ROOT=${VERIFIER_RUNTIME_ROOT:-${SHARED_ROOT}/spec_train/runtime_models/glm52-w4a8c8}
# auto inspects quantization_config.quant_method. compressed-tensors deliberately
# adds no --quantization flag; ascend adds --quantization ascend.
VERIFIER_QUANTIZATION_MODE=${VERIFIER_QUANTIZATION_MODE:-auto}

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm52-w4a8c8-verifier}
VERIFIER_HOST=${VERIFIER_HOST:-}
VERIFIER_ID=${VERIFIER_ID:-0}
VERIFIER_BIND_HOST=${VERIFIER_BIND_HOST:-0.0.0.0}
VERIFIER_PORT=${VERIFIER_PORT:-8077}
VERIFIER_TP_SIZE=${VERIFIER_TP_SIZE:-8}
VERIFIER_DP_SIZE=${VERIFIER_DP_SIZE:-2}
VERIFIER_MAX_MODEL_LEN=${VERIFIER_MAX_MODEL_LEN:-8193}
VERIFIER_MAX_NUM_SEQS=${VERIFIER_MAX_NUM_SEQS:-12}
VERIFIER_MAX_BATCHED_TOKENS=${VERIFIER_MAX_BATCHED_TOKENS:-8192}
VERIFIER_GPU_MEMORY_UTILIZATION=${VERIFIER_GPU_MEMORY_UTILIZATION:-0.92}
TARGET_LAYER_ID=${TARGET_LAYER_ID:-78}

NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NODE_RANK=${NODE_RANK:-}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29500}
LOCAL_IP=${LOCAL_IP:-}
NIC_NAME=${NIC_NAME:-}
RUN_ID=${RUN_ID:-glm52-w4a8c8-mtp3}

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
RUN_NAME=${RUN_NAME:-glm52-w4a8c8-ascend-mtp3}

SMOKE_SAMPLES=${SMOKE_SAMPLES:-64}
SMOKE_RUN_ID=${SMOKE_RUN_ID:-}
SMOKE_DATA_PATH=${SMOKE_DATA_PATH:-${SHARED_ROOT}/spec_train/smoke/glm52-mtp3-tokens-64-${SMOKE_RUN_ID:-unset}}
MTP_PREPARE_TIMEOUT=${MTP_PREPARE_TIMEOUT:-3600}

ASCEND_DEVICES=${ASCEND_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_DEVICES}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_TRANSFER_TIMEOUT=${HCCL_TRANSFER_TIMEOUT:-600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}
export VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}
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

validate_verifier_id() {
    [[ $VERIFIER_ID =~ ^[0-9]+$ ]] || fail "VERIFIER_ID must be an integer"
    ((VERIFIER_ID >= 0 && VERIFIER_ID < 4)) || \
        fail "VERIFIER_ID must be in [0, 3]"
}

validate_verifier_quantization_mode() {
    case "$VERIFIER_QUANTIZATION_MODE" in
        auto | ascend | compressed-tensors) ;;
        *) fail "VERIFIER_QUANTIZATION_MODE must be auto, ascend, or compressed-tensors" ;;
    esac
}

validate_trainer_topology() {
    require_value VERIFIER_HOST
    require_value MASTER_ADDR
    require_value NODE_RANK
    [[ $NODE_RANK =~ ^[0-9]+$ ]] || fail "NODE_RANK must be an integer"
    ((NODE_RANK >= 0 && NODE_RANK < NNODES)) || \
        fail "NODE_RANK must be in [0, $((NNODES - 1))]"
    if [[ -n $LOCAL_IP || -n $NIC_NAME ]]; then
        require_value LOCAL_IP
        require_value NIC_NAME
        export HCCL_IF_IP=$LOCAL_IP
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export TP_SOCKET_IFNAME=$NIC_NAME
        export HCCL_SOCKET_IFNAME=$NIC_NAME
    fi
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
  VERIFIER_METADATA_PATH=$VERIFIER_METADATA_PATH
  VERIFIER_RUNTIME_ROOT=$VERIFIER_RUNTIME_ROOT
  VERIFIER_QUANTIZATION_MODE=$VERIFIER_QUANTIZATION_MODE
  VERIFIER_ID=$VERIFIER_ID VERIFIER_HOST=${VERIFIER_HOST:-<unset>} VERIFIER_PORT=$VERIFIER_PORT
  MASTER_ADDR=${MASTER_ADDR:-<unset>} MASTER_PORT=$MASTER_PORT
  NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=${NODE_RANK:-<unset>}
  LOCAL_IP=${LOCAL_IP:-<auto>} NIC_NAME=${NIC_NAME:-<auto>}
  VERIFIER_DP_SIZE=$VERIFIER_DP_SIZE VERIFIER_TP_SIZE=$VERIFIER_TP_SIZE
  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
EOF
}

VERIFIER_RUNTIME_MODEL_PATH=$VERIFIER_MODEL_PATH
VERIFIER_DETECTED_QUANT_METHOD=
prepare_verifier_runtime_model() {
    local prepare_cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_mixed_quant_model.py"
        --model "$VERIFIER_MODEL_PATH"
        --runtime-root "$VERIFIER_RUNTIME_ROOT/verifier${VERIFIER_ID}"
    )
    if [[ $DRY_RUN == 1 ]]; then
        print_cmd "${prepare_cmd[@]}"
        if [[ $VERIFIER_QUANTIZATION_MODE != auto ]]; then
            VERIFIER_DETECTED_QUANT_METHOD=$VERIFIER_QUANTIZATION_MODE
        fi
        return
    fi

    local output
    output=$("${prepare_cmd[@]}") || fail "failed to prepare verifier quantization metadata"
    local -a result=()
    mapfile -t result <<<"$output"
    ((${#result[@]} == 3)) || fail "unexpected mixed-quant preparation output"
    VERIFIER_RUNTIME_MODEL_PATH=${result[0]}
    VERIFIER_DETECTED_QUANT_METHOD=${result[1]}
    printf '[quantization] method=%s metadata=%s model=%s\n' \
        "${VERIFIER_DETECTED_QUANT_METHOD:-none}" "${result[2]}" \
        "$VERIFIER_RUNTIME_MODEL_PATH"

    if [[ $VERIFIER_QUANTIZATION_MODE != auto && \
          $VERIFIER_DETECTED_QUANT_METHOD != "$VERIFIER_QUANTIZATION_MODE" ]]; then
        fail "requested quantization mode $VERIFIER_QUANTIZATION_MODE does not match model method ${VERIFIER_DETECTED_QUANT_METHOD:-none}"
    fi
}

preflight_args() {
    local required_devices=$NPROC_PER_NODE
    local verifier_preflight_path=$VERIFIER_MODEL_PATH
    if [[ $ROLE == verifier ]]; then
        required_devices=$((VERIFIER_TP_SIZE * VERIFIER_DP_SIZE))
    elif [[ $ROLE == trainer || $ROLE == smoke ]]; then
        verifier_preflight_path=$VERIFIER_METADATA_PATH
    fi
    PREFLIGHT_CMD=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/preflight_ascend_mtp.py"
        --bf16-model "$MTP_INIT_MODEL_PATH"
        --verifier-model "$verifier_preflight_path"
        --data-path "$DATA_PATH"
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --required-devices "$required_devices"
    )
    if [[ $DRY_RUN == 1 ]]; then
        PREFLIGHT_CMD+=(--skip-device-check)
    fi
}

publish_verifier_metadata() {
    local marker="$VERIFIER_METADATA_PATH/.ready"
    if [[ $DRY_RUN == 1 ]]; then
        printf '[metadata] publish verifier config/tokenizer to %s\n' \
            "$VERIFIER_METADATA_PATH"
        return
    fi
    mkdir -p "$(dirname -- "$VERIFIER_METADATA_PATH")"
    (
        flock -x 9
        if [[ -f $marker ]]; then
            local existing
            for existing in "$VERIFIER_METADATA_PATH"/*; do
                [[ -f $existing ]] || continue
                [[ ${existing##*/} == .ready ]] && continue
                [[ -f $VERIFIER_MODEL_PATH/${existing##*/} ]] || \
                    fail "published verifier metadata is stale: ${existing##*/}"
                cmp -s -- "$existing" "$VERIFIER_MODEL_PATH/${existing##*/}" || \
                    fail "published verifier metadata differs: ${existing##*/}"
            done
            exit 0
        fi
        [[ ! -e $VERIFIER_METADATA_PATH ]] || \
            fail "$VERIFIER_METADATA_PATH exists without .ready; inspect it manually"
        local temporary="${VERIFIER_METADATA_PATH}.tmp.${HOSTNAME}.$$"
        trap 'rm -rf -- "$temporary"' EXIT
        mkdir -p "$temporary"
        local filename
        for filename in config.json tokenizer.json tokenizer.model \
            tokenizer_config.json special_tokens_map.json added_tokens.json; do
            if [[ -f $VERIFIER_MODEL_PATH/$filename ]]; then
                cp -- "$VERIFIER_MODEL_PATH/$filename" "$temporary/$filename"
            fi
        done
        [[ -f $temporary/config.json ]] || \
            fail "verifier config.json was not copied from $VERIFIER_MODEL_PATH"
        if [[ ! -f $temporary/tokenizer.json && ! -f $temporary/tokenizer.model ]]; then
            fail "verifier tokenizer assets were not copied from $VERIFIER_MODEL_PATH"
        fi
        touch "$temporary/.ready"
        mv -- "$temporary" "$VERIFIER_METADATA_PATH"
        trap - EXIT
    ) 9>"${VERIFIER_METADATA_PATH}.lock"
}

run_preflight() {
    preflight_args
    run_cmd "${PREFLIGHT_CMD[@]}" "$@"
}

run_verifier() {
    validate_verifier_id
    validate_verifier_quantization_mode
    run_preflight --require-vllm
    prepare_verifier_runtime_model
    publish_verifier_metadata
    local log_file="$LOG_ROOT/verifier${VERIFIER_ID}/verifier.log"
    local effective_quantization_mode=$VERIFIER_QUANTIZATION_MODE
    if [[ $effective_quantization_mode == auto ]]; then
        effective_quantization_mode=$VERIFIER_DETECTED_QUANT_METHOD
    fi
    local quantization_args=()
    if [[ $effective_quantization_mode == ascend ]]; then
        quantization_args+=(--quantization ascend)
    fi
    local cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/launch_vllm.py"
        "$VERIFIER_RUNTIME_MODEL_PATH"
        --hidden-states-backend file
        --hidden-states-path "$HIDDEN_STATES_PATH"
        --target-layer-ids "$TARGET_LAYER_ID"
        --
        --host "$VERIFIER_BIND_HOST"
        --port "$VERIFIER_PORT"
        --served-model-name "$SERVED_MODEL_NAME"
        --safetensors-load-strategy prefetch
        --api-server-count 1
        --data-parallel-size "$VERIFIER_DP_SIZE"
        --tensor-parallel-size "$VERIFIER_TP_SIZE"
        --enable-expert-parallel
        --seed 1024
        --max-num-seqs "$VERIFIER_MAX_NUM_SEQS"
        --max-model-len "$VERIFIER_MAX_MODEL_LEN"
        --max-num-batched-tokens "$VERIFIER_MAX_BATCHED_TOKENS"
        --gpu-memory-utilization "$VERIFIER_GPU_MEMORY_UTILIZATION"
        "${quantization_args[@]}"
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        --additional-config '{"enable_dsa_cp":true,"enable_sparse_li_c8":true,"enable_balance_scheduling":true,"multistream_overlap_shared_expert":true}'
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
    if [[ $DRY_RUN == 1 ]]; then
        printf '[wait] %s: %s\n' "$description" "$marker"
        return
    fi
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
    wait_for_marker "$VERIFIER_METADATA_PATH/.ready" "verifier metadata"
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
