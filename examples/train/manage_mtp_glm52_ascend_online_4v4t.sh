#!/usr/bin/env bash
# Validate and operate a user-configured 4-verifier + 4-trainer MTP cluster.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CONFIG_RENDERER=$REPO_ROOT/scripts/render_ascend_mtp_cluster_yaml.py

DEFAULT_CONFIG=/kos_ulan/spec_train/config/glm52-mtp3-4v4t.yaml
DEFAULT_REPO=/kos_ulan/spec_train/speculators
DEFAULT_CONTAINER_PREFIX=glm52-mtp3

usage() {
    cat <<'EOF'
Usage:
  manage_mtp_glm52_ascend_online_4v4t.sh COMMAND [--config FILE]

Commands:
  validate-config  Validate the user-maintained YAML without opening SSH.
  preflight        Run non-training validation on all eight nodes.
  start-verifiers  Start four verifier containers in the background.
  wait-verifiers   Wait until all verifier health endpoints return HTTP 200.
  smoke            Start the two-step smoke job on four trainer nodes.
  train            Start the production online-cache training job.
  offline          Resume using only cached hidden states.
  status           Show matching containers and recent host-wrapper logs.
  stop             Stop cluster jobs; restart reused containers in existing mode.

Options:
  --config FILE    User-maintained YAML. Default:
                   /kos_ulan/spec_train/config/glm52-mtp3-4v4t.yaml

Runtime environment overrides:
  SSH_USER=root SSH_PORT=22 SSH_IDENTITY_FILE=/path/to/key
  SSH_STRICT_HOST_KEY_CHECKING=accept-new HEALTH_TIMEOUT=7200
  MANAGER_DRY_RUN=1  # print remote SSH commands without executing them

Copy and edit examples/train/mtp_glm52_ascend_online_4v4t.example.yaml; this
manager never generates or rewrites the YAML. It deliberately does not store
passwords; configure SSH keys first.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

quote() {
    printf '%q' "$1"
}

CONFIG_FILE=$DEFAULT_CONFIG
parse_config_option() {
    while (( $# )); do
        case "$1" in
            --config) CONFIG_FILE=${2:-}; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) fail "unknown option: $1" ;;
        esac
    done
}

load_config() {
    [[ -f $CONFIG_FILE ]] || fail "configuration not found: $CONFIG_FILE"
    case "$CONFIG_FILE" in
        *.yaml | *.yml)
            local rendered_config
            rendered_config=$(mktemp)
            "$CONFIG_RENDERER" --config "$CONFIG_FILE" >"$rendered_config" || {
                rm -f -- "$rendered_config"
                fail "invalid YAML configuration: $CONFIG_FILE"
            }
            # shellcheck disable=SC1090
            source "$rendered_config"
            rm -f -- "$rendered_config"
            ;;
        *)
            # Legacy shell configuration remains readable for existing jobs.
            # shellcheck disable=SC1090
            source "$CONFIG_FILE"
            ;;
    esac
    declare -p CLUSTER_VERIFIER_IPS >/dev/null 2>&1 || \
        fail "CLUSTER_VERIFIER_IPS is missing"
    declare -p CLUSTER_TRAINER_IPS >/dev/null 2>&1 || \
        fail "CLUSTER_TRAINER_IPS is missing"
    ((${#CLUSTER_VERIFIER_IPS[@]} == 4)) || fail "expected four verifier IPs"
    ((${#CLUSTER_TRAINER_IPS[@]} == 4)) || fail "expected four trainer IPs"
    CONTAINER_NAME_PREFIX=${CONTAINER_NAME_PREFIX:-$DEFAULT_CONTAINER_PREFIX}
    REMOTE_REPO_PATH=${REMOTE_REPO_PATH:-$DEFAULT_REPO}
    SHARED_ROOT=${SHARED_ROOT:-/kos_ulan}
    ORCHESTRATOR_LOG_ROOT=${ORCHESTRATOR_LOG_ROOT:-$SHARED_ROOT/spec_train/logs/orchestrator}
    CONTAINER_MODE=${CONTAINER_MODE:-create}
    EXISTING_CONTAINER_NAME=${EXISTING_CONTAINER_NAME:-}
}

container_for() {
    local role=$1 index=$2
    if [[ $CONTAINER_MODE == existing ]]; then
        printf '%s' "$EXISTING_CONTAINER_NAME"
    else
        printf '%s-%s%s' "$CONTAINER_NAME_PREFIX" "$role" "$index"
    fi
}

ssh_args() {
    SSH_ARGS=(
        -p "${SSH_PORT:-22}"
        -o BatchMode=yes
        -o ConnectTimeout=10
        -o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
    )
    if [[ -n ${SSH_IDENTITY_FILE:-} ]]; then
        SSH_ARGS+=(-i "$SSH_IDENTITY_FILE")
    fi
}

remote() {
    local host=$1
    shift
    ssh_args
    if [[ ${MANAGER_DRY_RUN:-0} == 1 ]]; then
        printf '[ssh]'
        printf ' %q' ssh "${SSH_ARGS[@]}" "${SSH_USER:-root}@${host}" "$@"
        printf '\n'
        return
    fi
    ssh "${SSH_ARGS[@]}" "${SSH_USER:-root}@${host}" "$@"
}

build_remote_command() {
    local destination=$1
    shift
    printf -v "$destination" '%q ' "$@"
}

start_node() {
    local host=$1 role=$2 index=$3 mode=${4:-}
    local container
    container=$(container_for "$role" "$index")
    local verifier_host=${CLUSTER_VERIFIER_IPS[$index]:-}
    local host_log="$ORCHESTRATOR_LOG_ROOT/$CONTAINER_NAME_PREFIX-$role$index.host.log"
    local -a command=(
        env
        "NODE_IP=$host"
        "LOCAL_IP=$host"
        NIC_NAME=auto
        "ROLE=$role"
        "CONTAINER_NAME=$container"
    )
    case "$role" in
        verifier) command+=("VERIFIER_ID=$index") ;;
        smoke | trainer)
            command+=(
                "NODE_RANK=$index"
                NNODES=4
                "MASTER_ADDR=${CLUSTER_TRAINER_IPS[0]}"
                "VERIFIER_HOST=$verifier_host"
            )
            [[ -n $mode ]] && command+=("TRAINER_DATA_MODE=$mode")
            ;;
        preflight) ;;
        *) fail "unsupported role: $role" ;;
    esac
    command+=(
        bash "$REMOTE_REPO_PATH/examples/train/run_mtp_glm52_ascend_online_container.sh"
        "$CONFIG_FILE"
    )
    local serialized
    build_remote_command serialized "${command[@]}"
    local shell_command
    shell_command="mkdir -p $(quote "$ORCHESTRATOR_LOG_ROOT"); "
    shell_command+="cd $(quote "$REMOTE_REPO_PATH"); "
    shell_command+="nohup $serialized >$(quote "$host_log") 2>&1 </dev/null "
    shell_command+="& echo STARTED_PID=\$!"
    printf '[start] host=%s container=%s log=%s\n' "$host" "$container" "$host_log"
    remote "$host" "$shell_command"
}

run_preflight_node() {
    local host=$1 index=$2
    local container
    container=$(container_for preflight "$index")
    local -a command=(
        env "NODE_IP=$host" "LOCAL_IP=$host" NIC_NAME=auto ROLE=preflight
        "CONTAINER_NAME=$container"
        bash "$REMOTE_REPO_PATH/examples/train/run_mtp_glm52_ascend_online_container.sh"
        "$CONFIG_FILE"
    )
    local serialized
    build_remote_command serialized "${command[@]}"
    printf '[preflight] host=%s\n' "$host"
    remote "$host" "cd $(quote "$REMOTE_REPO_PATH"); $serialized"
}

start_group() {
    local role=$1 mode=${2:-}
    local -n addresses=$3
    local index
    for index in "${!addresses[@]}"; do
        start_node "${addresses[$index]}" "$role" "$index" "$mode"
    done
}

wait_verifiers() {
    local timeout=${HEALTH_TIMEOUT:-7200}
    local started=$SECONDS
    local -a ready=(0 0 0 0)
    while :; do
        local index remaining=0 code
        for index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
            if [[ ${ready[$index]} == 1 ]]; then
                continue
            fi
            code=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' \
                "http://${CLUSTER_VERIFIER_IPS[$index]}:${VERIFIER_PORT:-8077}/health" \
                2>/dev/null || true)
            if [[ $code == 200 ]]; then
                ready[$index]=1
                printf '[healthy] verifier%d=%s\n' \
                    "$index" "${CLUSTER_VERIFIER_IPS[$index]}"
            else
                remaining=$((remaining + 1))
            fi
        done
        ((remaining > 0)) || return 0
        ((SECONDS - started < timeout)) || \
            fail "timed out waiting for $remaining verifier(s)"
        sleep 15
    done
}

show_role_status() {
    local role=$1
    local -n addresses=$2
    local index host container
    for index in "${!addresses[@]}"; do
        host=${addresses[$index]}
        container=$(container_for "$role" "$index")
        printf '\n[status] host=%s container=%s\n' "$host" "$container"
        local status_command
        status_command="docker ps -a --filter name=^/$(quote "$container")\$ "
        status_command+="--format '{{.Names}} {{.Status}}'; "
        status_command+="tail -n 8 "
        status_command+="$(quote "$ORCHESTRATOR_LOG_ROOT/$CONTAINER_NAME_PREFIX-$role$index.host.log") "
        status_command+="2>/dev/null || true"
        remote "$host" "$status_command" || true
    done
}

show_status() {
    show_role_status verifier CLUSTER_VERIFIER_IPS
    show_role_status trainer CLUSTER_TRAINER_IPS
    show_role_status smoke CLUSTER_TRAINER_IPS
}

stop_role() {
    local role=$1
    local -n addresses=$2
    local index host container
    for index in "${!addresses[@]}"; do
        host=${addresses[$index]}
        container=$(container_for "$role" "$index")
        printf '[stop] host=%s container=%s\n' "$host" "$container"
        if [[ $CONTAINER_MODE == existing ]]; then
            local pid_file="$LOG_ROOT/runtime_pids/$container.$role$index.pid"
            local stop_command
            stop_command="docker exec $(quote "$container") bash -lc "
            stop_command+=$(quote "if [[ -f $pid_file ]]; then pid=\$(cat $pid_file); if [[ \$pid =~ ^[0-9]+$ ]] && kill -0 \"\$pid\" 2>/dev/null; then kill -TERM \"\$pid\"; for ((i=0; i<180; i++)); do kill -0 \"\$pid\" 2>/dev/null || break; sleep 1; done; if kill -0 \"\$pid\" 2>/dev/null; then echo 'ERROR: job did not stop within 180 seconds: pid='\"\$pid\" >&2; exit 1; fi; fi; rm -f -- $pid_file; fi")
            if ! remote "$host" "$stop_command"; then
                printf 'WARNING: graceful stop did not complete: host=%s container=%s\n' \
                    "$host" "$container" >&2
            fi
        else
            remote "$host" \
                "docker stop -t 30 $(quote "$container") >/dev/null 2>&1 || true" || true
        fi
    done
}

restart_existing_containers() {
    [[ $CONTAINER_MODE == existing ]] || return 0

    local role addresses_name index host container
    for role in verifier trainer; do
        if [[ $role == verifier ]]; then
            addresses_name=CLUSTER_VERIFIER_IPS
        else
            addresses_name=CLUSTER_TRAINER_IPS
        fi
        local -n addresses=$addresses_name
        for index in "${!addresses[@]}"; do
            host=${addresses[$index]}
            container=$(container_for "$role" "$index")
            printf '[restart] host=%s container=%s\n' "$host" "$container"
            remote "$host" \
                "docker restart -t 30 $(quote "$container") >/dev/null"
        done
        unset -n addresses
    done
}

stop_cluster() {
    stop_role verifier CLUSTER_VERIFIER_IPS
    stop_role trainer CLUSTER_TRAINER_IPS
    stop_role smoke CLUSTER_TRAINER_IPS
    restart_existing_containers
}

COMMAND=${1:-}
[[ -n $COMMAND ]] || { usage; exit 2; }
shift

parse_config_option "$@"
load_config

case "$COMMAND" in
    validate-config)
        printf 'CONFIG_STATUS=valid\nCONFIG_FILE=%s\n' "$CONFIG_FILE"
        printf 'VERIFIERS=%s\nTRAINERS=%s\n' \
            "${#CLUSTER_VERIFIER_IPS[@]}" "${#CLUSTER_TRAINER_IPS[@]}"
        printf 'CONTAINER_MODE=%s\n' "$CONTAINER_MODE"
        ;;
    preflight)
        index=0
        for host in "${CLUSTER_VERIFIER_IPS[@]}" "${CLUSTER_TRAINER_IPS[@]}"; do
            run_preflight_node "$host" "$index"
            index=$((index + 1))
        done
        ;;
    start-verifiers) start_group verifier '' CLUSTER_VERIFIER_IPS ;;
    wait-verifiers) wait_verifiers ;;
    smoke) start_group smoke online-cache CLUSTER_TRAINER_IPS ;;
    train) start_group trainer online-cache CLUSTER_TRAINER_IPS ;;
    offline) start_group trainer offline CLUSTER_TRAINER_IPS ;;
    status) show_status ;;
    stop) stop_cluster ;;
    -h | --help | help) usage ;;
    *) fail "unknown command: $COMMAND" ;;
esac
