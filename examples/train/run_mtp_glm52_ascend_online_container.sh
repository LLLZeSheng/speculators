#!/usr/bin/env bash
# Host-side wrapper for the 4-verifier + 2/4-trainer Ascend online MTP3 job.

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
    case "$CONFIG_FILE" in
        *.yaml | *.yml)
            rendered_config=$(mktemp)
            trap 'rm -f -- "${rendered_config:-}"' EXIT
            python "$REPO_HOST_PATH/scripts/render_ascend_mtp_cluster_yaml.py" \
                --config "$CONFIG_FILE" >"$rendered_config"
            # shellcheck disable=SC1090
            source "$rendered_config"
            rm -f -- "$rendered_config"
            trap - EXIT
            ;;
        *)
            # Legacy shell configuration remains readable for existing jobs.
            # shellcheck disable=SC1090
            source "$CONFIG_FILE"
            ;;
    esac
fi

ROLE=${ROLE:-}
TRAINER_MODE=${TRAINER_MODE:-smoke}
IMAGE=${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0rc1-a3}
CONTAINER_MODE=${CONTAINER_MODE:-create}
EXISTING_CONTAINER_NAME=${EXISTING_CONTAINER_NAME:-}
CONTAINER_REPO_PATH=${CONTAINER_REPO_PATH:-/workspace/speculators}
CONTAINER_SHM_SIZE=${CONTAINER_SHM_SIZE:-1g}
SHARED_ROOT=${SHARED_ROOT:-/kos_ulan}
VERIFIER_MODEL_PATH=${VERIFIER_MODEL_PATH:-/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v4}
VERIFIER_SOURCE_MODEL_PATH=${VERIFIER_SOURCE_MODEL_PATH:-}
MTP_INIT_MODEL_PATH=${MTP_INIT_MODEL_PATH:-$VERIFIER_MODEL_PATH}
DATA_PATH=${DATA_PATH:-${SHARED_ROOT}/lzs/spec_train/dataset/hf/nuoya-average2k8k-32k}
NIC_NAME=${NIC_NAME:-}
INSTALL_SPECULATORS=${INSTALL_SPECULATORS:-auto}
DRY_RUN=${DRY_RUN:-0}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

array_is_declared() {
    declare -p "$1" &>/dev/null
}

verifier_hosts_for_trainer() {
    local trainer_index=$1
    local trainer_count=${#CLUSTER_TRAINER_IPS[@]}
    local verifier_index
    local -a selected=()
    for verifier_index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
        if ((verifier_index % trainer_count == trainer_index)); then
            selected+=("${CLUSTER_VERIFIER_IPS[$verifier_index]}")
        fi
    done
    local IFS=,
    printf '%s' "${selected[*]}"
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
    local trainer_count=${#CLUSTER_TRAINER_IPS[@]}
    ((trainer_count == 2 || trainer_count == 4)) || \
        fail "CLUSTER_TRAINER_IPS must contain exactly two or four addresses"
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
            NNODES=$trainer_count
            MASTER_ADDR=${CLUSTER_TRAINER_IPS[0]}
            VERIFIER_HOSTS=$(verifier_hosts_for_trainer "$index")
            VERIFIER_HOST=${VERIFIER_HOSTS%%,*}
            LOCAL_IP=$address
            matches=$((matches + 1))
        fi
    done
    ((matches == 1)) || \
        fail "could not uniquely map local addresses${NODE_IP:+ (NODE_IP=$NODE_IP)} to the configured 4-verifier + 2/4-trainer cluster"
}

resolve_cluster_role
CONTAINER_NAME=${CONTAINER_NAME:-${CONTAINER_NAME_PREFIX:-glm52-mtp3}-${ROLE}${VERIFIER_ID:-${NODE_RANK:-0}}}
if [[ $CONTAINER_MODE == existing ]]; then
    [[ -n $EXISTING_CONTAINER_NAME ]] || \
        fail "EXISTING_CONTAINER_NAME is required in existing mode"
    CONTAINER_NAME=$EXISTING_CONTAINER_NAME
fi

resolve_nic_name() {
    if [[ $NIC_NAME != auto ]]; then
        return
    fi
    [[ -n ${LOCAL_IP:-} ]] || \
        fail "NIC_NAME=auto requires LOCAL_IP or NODE_IP"
    command -v ip >/dev/null || fail "NIC_NAME=auto requires the host ip command"
    NIC_NAME=$(
        ip -o -4 addr show | awk -v target="$LOCAL_IP" '
            {
                split($4, address, "/")
                if (address[1] == target) {
                    print $2
                    exit
                }
            }
        '
    )
    [[ -n $NIC_NAME ]] || fail "could not resolve NIC for LOCAL_IP=$LOCAL_IP"
    printf '[network] LOCAL_IP=%s NIC_NAME=%s\n' "$LOCAL_IP" "$NIC_NAME"
}

resolve_nic_name

case "$ROLE" in
    preflight | verifier | trainer | smoke) ;;
    *) fail "ROLE must be one of: preflight, verifier, trainer, smoke" ;;
esac

if [[ $INSTALL_SPECULATORS == auto ]]; then
    case "$ROLE" in
        trainer | smoke)
            INSTALL_SPECULATORS=${INSTALL_SPECULATORS_TRAINER:-0}
            ;;
        verifier | preflight)
            INSTALL_SPECULATORS=${INSTALL_SPECULATORS_VERIFIER:-0}
            ;;
    esac
fi
case "$INSTALL_SPECULATORS" in
    0 | 1) ;;
    *) fail "INSTALL_SPECULATORS must resolve to 0 or 1" ;;
esac
case "$CONTAINER_MODE" in
    create | existing) ;;
    *) fail "CONTAINER_MODE must be create or existing" ;;
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

if ! array_is_declared CONTAINER_MOUNTS; then
    CONTAINER_MOUNTS=(
        "$SHARED_ROOT:$SHARED_ROOT"
        "$REPO_HOST_PATH:$CONTAINER_REPO_PATH"
    )
    [[ ! -e /mnt/xds/sfs ]] || CONTAINER_MOUNTS+=(/mnt/xds/sfs:/mnt/xds/sfs)
    [[ ! -e /root/.cache ]] || CONTAINER_MOUNTS+=(/root/.cache:/root/.cache)
fi

forward_vars=(
    ROLE DRY_RUN SHARED_ROOT VERIFIER_MODEL_PATH VERIFIER_SOURCE_MODEL_PATH MTP_INIT_MODEL_PATH DATA_PATH
    HIDDEN_STATES_PATH MTP_DRAFT_PATH OUTPUT_PATH LOG_ROOT VERIFIER_METADATA_PATH
    VERIFIER_RUNTIME_ROOT VERIFIER_QUANTIZATION_MODE CONTAINER_REPO_PATH
    CONTAINER_NAME INSTALL_SPECULATORS
    SERVED_MODEL_NAME VERIFIER_HOST VERIFIER_HOSTS VERIFIER_ID VERIFIER_BIND_HOST VERIFIER_PORT
    VERIFIER_TP_SIZE VERIFIER_DP_SIZE VERIFIER_MAX_MODEL_LEN
    VERIFIER_MAX_NUM_SEQS VERIFIER_MAX_BATCHED_TOKENS
    VERIFIER_GPU_MEMORY_UTILIZATION TARGET_LAYER_ID NNODES NPROC_PER_NODE
    NODE_RANK MASTER_ADDR MASTER_PORT LOCAL_IP NIC_NAME RUN_ID TOTAL_SEQ_LEN
    EPOCHS LR WEIGHT_DECAY STEP_WEIGHT_BETA CHECKPOINT_STEPS TRAIN_DATA_RATIO
    NUM_WORKERS PREFETCH_FACTOR REQUEST_TIMEOUT MAX_RETRIES MAX_STEPS RUN_NAME TRAINER_DATA_MODE SMOKE_SAMPLES SMOKE_RUN_ID
    SMOKE_DATA_PATH MTP_PREPARE_TIMEOUT ASCEND_DEVICES EXPECTED_TORCH_NPU_VERSION
    EXPECTED_VLLM_VERSION EXPECTED_VLLM_ASCEND_VERSION
)

inner=(bash "$CONTAINER_REPO_PATH/examples/train/run_mtp_glm52_ascend_online_job.sh")

if [[ $CONTAINER_MODE == existing ]]; then
    docker_args=(exec)
    for name in "${forward_vars[@]}"; do
        [[ -v $name ]] && docker_args+=(-e "$name=${!name}")
    done
    docker_args+=("$CONTAINER_NAME")
    printf '[container-existing]'
    printf ' %q' docker "${docker_args[@]}" "${inner[@]}"
    printf '\n'
    if [[ $DRY_RUN != 1 ]]; then
        docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | \
            grep -qx true || fail "existing container is not running: $CONTAINER_NAME"
        exec docker "${docker_args[@]}" "${inner[@]}"
    fi
    exit 0
fi

docker_args=(
    run
    --name "$CONTAINER_NAME"
    --net host
    --shm-size "$CONTAINER_SHM_SIZE"
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
    [[ -e $path ]] && docker_args+=(-v "$path:$path")
done

for mount in "${CONTAINER_MOUNTS[@]}"; do
    host_path=${mount%%:*}
    if [[ $DRY_RUN != 1 && ! -e $host_path ]]; then
        fail "container mount source is missing: $host_path"
    fi
    docker_args+=(-v "$mount")
done

for name in "${forward_vars[@]}"; do
    [[ -v $name ]] && docker_args+=(-e "$name=${!name}")
done

printf '[container-create]'
printf ' %q' docker "${docker_args[@]}" "$IMAGE" "${inner[@]}"
printf '\n'
if [[ $DRY_RUN != 1 ]]; then
    exec docker "${docker_args[@]}" "$IMAGE" "${inner[@]}"
fi
