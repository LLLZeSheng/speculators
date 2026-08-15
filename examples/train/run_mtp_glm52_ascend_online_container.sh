#!/usr/bin/env bash
# Host-side wrapper for the eight-node Ascend 910C / Atlas A3 online MTP3 job.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_HOST_PATH=$(cd -- "$SCRIPT_DIR/../.." && pwd)

CONFIG_FILE=${CONFIG_FILE:-${1:-}}
if (( $# > 1 )); then
    printf 'ERROR: usage: %s [CONFIG_FILE]\n' "$0" >&2
    exit 2
fi
if [[ -n $CONFIG_FILE ]]; then
    [[ -f $CONFIG_FILE ]] || {
        printf 'ERROR: configuration file is missing: %s\n' "$CONFIG_FILE" >&2
        exit 2
    }
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

ROLE=${ROLE:-}
TRAINER_MODE=${TRAINER_MODE:-smoke}
IMAGE=${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0rc1-a3}
CONTAINER_REPO_PATH=${CONTAINER_REPO_PATH:-/workspace/speculators}
SHARED_ROOT=${SHARED_ROOT:-/kos_ulan}
VERIFIER_MODEL_PATH=${VERIFIER_MODEL_PATH:-/mnt/xds/dev/s00838505/GLM-5.2-w4a8c8}
MTP_INIT_MODEL_PATH=${MTP_INIT_MODEL_PATH:-${SHARED_ROOT}/models/GLM-5.2}
DATA_PATH=${DATA_PATH:-${SHARED_ROOT}/datasets/glm52-dspark-train}
NIC_NAME=${NIC_NAME:-}
INSTALL_SPECULATORS=${INSTALL_SPECULATORS:-1}
DRY_RUN=${DRY_RUN:-0}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

array_is_declared() {
    declare -p "$1" &>/dev/null
}

resolve_cluster_role() {
    if [[ -n $ROLE ]]; then
        return 0
    fi
    array_is_declared CLUSTER_VERIFIER_IPS || \
        fail "ROLE is unset and CLUSTER_VERIFIER_IPS is not declared"
    array_is_declared CLUSTER_TRAINER_IPS || \
        fail "ROLE is unset and CLUSTER_TRAINER_IPS is not declared"
    ((${#CLUSTER_VERIFIER_IPS[@]} == 4)) || \
        fail "CLUSTER_VERIFIER_IPS must contain exactly four addresses"
    ((${#CLUSTER_TRAINER_IPS[@]} == 4)) || \
        fail "CLUSTER_TRAINER_IPS must contain exactly four addresses"
    case "$TRAINER_MODE" in
        smoke | trainer) ;;
        *) fail "TRAINER_MODE must be smoke or trainer" ;;
    esac

    local candidates=" ${NODE_IP:-$(hostname -I 2>/dev/null || true)} "
    local index address matches=0
    for index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
        address=${CLUSTER_VERIFIER_IPS[$index]}
        if [[ $candidates == *" $address "* ]]; then
            ROLE=verifier
            VERIFIER_ID=$index
            LOCAL_IP=$address
            matches=$((matches + 1))
        fi
    done
    for index in "${!CLUSTER_TRAINER_IPS[@]}"; do
        address=${CLUSTER_TRAINER_IPS[$index]}
        if [[ $candidates == *" $address "* ]]; then
            ROLE=$TRAINER_MODE
            NODE_RANK=$index
            NNODES=4
            MASTER_ADDR=${CLUSTER_TRAINER_IPS[0]}
            VERIFIER_HOST=${CLUSTER_VERIFIER_IPS[$index]}
            LOCAL_IP=$address
            matches=$((matches + 1))
        fi
    done
    ((matches == 1)) || \
        fail "could not uniquely map local addresses${NODE_IP:+ (NODE_IP=$NODE_IP)} to the configured 4+4 cluster"
}

resolve_cluster_role
CONTAINER_NAME=${CONTAINER_NAME:-glm52-mtp3-${ROLE}-${VERIFIER_ID:-${NODE_RANK:-0}}}

case "$ROLE" in
    preflight | verifier | trainer | smoke) ;;
    *) fail "ROLE must be one of: preflight, verifier, trainer, smoke" ;;
esac

if [[ -n $CONFIG_FILE ]]; then
    for value in "$NIC_NAME" "$MTP_INIT_MODEL_PATH" "$DATA_PATH"; do
        [[ -n $value && $value != *FILL_* ]] || \
            fail "configuration still contains an unfilled FILL_* value"
    done
    if [[ $ROLE == smoke ]]; then
        [[ -n ${SMOKE_RUN_ID:-} && $SMOKE_RUN_ID != *FILL_* ]] || \
            fail "SMOKE_RUN_ID must be replaced with a unique value"
    fi
fi

[[ -d $REPO_HOST_PATH ]] || fail "repository is missing: $REPO_HOST_PATH"
if [[ $DRY_RUN != 1 ]]; then
    [[ -d $SHARED_ROOT ]] || fail "shared directory is not mounted: $SHARED_ROOT"
    if [[ $ROLE == verifier || $ROLE == preflight ]]; then
        [[ -d $VERIFIER_MODEL_PATH ]] || \
            fail "verifier model is missing: $VERIFIER_MODEL_PATH"
    fi
fi

docker_args=(
    run --rm
    --name "$CONTAINER_NAME"
    --network host
    --ipc host
    --ulimit memlock=-1
    --cap-add SYS_NICE
    -v "$SHARED_ROOT:$SHARED_ROOT"
    -v "$REPO_HOST_PATH:$CONTAINER_REPO_PATH"
)

for device in /dev/davinci{0..15} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    if [[ $DRY_RUN != 1 && ! -e $device ]]; then
        fail "required Ascend device is missing: $device"
    fi
    docker_args+=(--device "$device")
done

driver_mounts=(
    /usr/local/dcmi
    /usr/local/Ascend/driver/tools/hccn_tool
    /usr/local/bin/npu-smi
    /usr/local/Ascend/driver/lib64
    /usr/local/Ascend/driver/version.info
    /etc/ascend_install.info
)
for path in "${driver_mounts[@]}"; do
    [[ -e $path ]] && docker_args+=(-v "$path:$path:ro")
done

if [[ -d $VERIFIER_MODEL_PATH ]]; then
    docker_args+=(-v "$VERIFIER_MODEL_PATH:$VERIFIER_MODEL_PATH:ro")
fi

forward_vars=(
    ROLE DRY_RUN SHARED_ROOT VERIFIER_MODEL_PATH MTP_INIT_MODEL_PATH DATA_PATH
    HIDDEN_STATES_PATH MTP_DRAFT_PATH OUTPUT_PATH LOG_ROOT VERIFIER_METADATA_PATH
    SERVED_MODEL_NAME VERIFIER_HOST VERIFIER_ID VERIFIER_BIND_HOST VERIFIER_PORT
    VERIFIER_TP_SIZE VERIFIER_DP_SIZE VERIFIER_MAX_MODEL_LEN
    VERIFIER_MAX_NUM_SEQS VERIFIER_MAX_BATCHED_TOKENS
    VERIFIER_GPU_MEMORY_UTILIZATION TARGET_LAYER_ID NNODES NPROC_PER_NODE
    NODE_RANK MASTER_ADDR MASTER_PORT LOCAL_IP NIC_NAME RUN_ID TOTAL_SEQ_LEN
    EPOCHS LR WEIGHT_DECAY STEP_WEIGHT_BETA CHECKPOINT_STEPS TRAIN_DATA_RATIO
    NUM_WORKERS PREFETCH_FACTOR MAX_STEPS RUN_NAME SMOKE_SAMPLES SMOKE_RUN_ID
    SMOKE_DATA_PATH MTP_PREPARE_TIMEOUT ASCEND_DEVICES EXPECTED_TORCH_NPU_VERSION
    EXPECTED_VLLM_VERSION EXPECTED_VLLM_ASCEND_VERSION
)
for name in "${forward_vars[@]}"; do
    [[ -v $name ]] && docker_args+=(-e "$name=${!name}")
done

inner=(bash "$CONTAINER_REPO_PATH/examples/train/mtp_glm52_ascend_online.sh")
if [[ $INSTALL_SPECULATORS == 1 ]]; then
    inner=(
        bash -lc
        "python -m pip install --no-deps -e '$CONTAINER_REPO_PATH' && exec bash '$CONTAINER_REPO_PATH/examples/train/mtp_glm52_ascend_online.sh'"
    )
fi

printf '[container]'
printf ' %q' docker "${docker_args[@]}" "$IMAGE" "${inner[@]}"
printf '\n'
if [[ $DRY_RUN != 1 ]]; then
    exec docker "${docker_args[@]}" "$IMAGE" "${inner[@]}"
fi
