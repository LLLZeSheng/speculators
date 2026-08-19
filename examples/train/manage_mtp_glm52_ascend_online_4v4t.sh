#!/usr/bin/env bash
# Validate and operate a user-configured N-verifier + 0/2/4-trainer MTP cluster.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CONFIG_RENDERER=$REPO_ROOT/scripts/render_ascend_mtp_cluster_yaml.py
DASHBOARD_SCRIPT=$REPO_ROOT/scripts/monitor_ascend_mtp_cluster.py

DEFAULT_CONFIG=/mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
DEFAULT_REPO=/mnt/xds/mtp/spec_train/speculators
DEFAULT_CONTAINER_PREFIX=glm52-mtp3

usage() {
    cat <<'EOF'
Usage:
  manage_mtp_glm52_ascend_online_4v4t.sh COMMAND [--config FILE] [--index N]

Commands:
  validate-config  Validate the user-maintained YAML without opening SSH.
  preflight        Run non-training validation on every configured node.
  start-verifiers  Ensure every configured verifier is running.
  start-verifier   Ensure one configured verifier is running (requires --index N).
  wait-verifiers   Wait until all verifier health endpoints return HTTP 200.
  collect-offline  Start one resumable collector per configured verifier.
  offline-status   Show collector jobs, logs, and current cache file count.
  verify-offline   Require a complete cache and publish .offline-ready.json.
  smoke            Start the two-step smoke job on all trainer nodes.
  train            Start the production online-cache training job.
  offline          Resume using only cached hidden states.
  status           Show matching containers and recent host-wrapper logs.
  dashboard        Start the control-node web dashboard.
  dashboard-status Show dashboard process, URL, and log location.
  stop-dashboard   Stop only the control-node web dashboard.
  restart-verifiers
                   Stop verifier jobs and restart only reused verifier containers.
  restart-trainers Stop trainer/smoke jobs and restart only reused trainer containers.
  stop-collectors  Stop only offline collection jobs; keep verifiers running.
  stop             Stop cluster jobs; restart reused containers in existing mode.

Options:
  --config FILE    User-maintained YAML. Default:
                   /mnt/xds/mtp/spec_train/config/glm52-mtp3-4v4t.yaml
  --index N        Verifier index for start-verifier.

Runtime environment overrides:
  SSH_USER=root SSH_PORT=22 SSH_IDENTITY_FILE=/path/to/key
  SSH_STRICT_HOST_KEY_CHECKING=accept-new HEALTH_TIMEOUT=7200
  STOP_GRACE_SECONDS=15  # existing-mode job grace before container restart
  DASHBOARD_ADVERTISE_HOST=control.node.ip  # URL shown to users
  DASHBOARD_PYTHON=python3
  MANAGER_DRY_RUN=1  # print remote SSH commands without executing them

Copy and edit a checked-in online-training or offline-collection YAML; this
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
TARGET_INDEX=
parse_config_option() {
    while (( $# )); do
        case "$1" in
            --config) CONFIG_FILE=${2:-}; shift 2 ;;
            --index) TARGET_INDEX=${2:-}; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) fail "unknown option: $1" ;;
        esac
    done
}

validate_verifier_index() {
    [[ $TARGET_INDEX =~ ^[0-9]+$ ]] || fail "--index must be a non-negative integer"
    ((TARGET_INDEX < ${#CLUSTER_VERIFIER_IPS[@]})) || \
        fail "--index must be smaller than ${#CLUSTER_VERIFIER_IPS[@]}"
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
    ((${#CLUSTER_VERIFIER_IPS[@]} >= 1)) || fail "expected at least one verifier IP"
    local trainer_count=${#CLUSTER_TRAINER_IPS[@]}
    ((trainer_count == 0 || trainer_count == 2 || trainer_count == 4)) || \
        fail "expected zero, two, or four trainer IPs"
    CONTAINER_NAME_PREFIX=${CONTAINER_NAME_PREFIX:-$DEFAULT_CONTAINER_PREFIX}
    REMOTE_REPO_PATH=${REMOTE_REPO_PATH:-$DEFAULT_REPO}
    SHARED_ROOT=${SHARED_ROOT:-/mnt/xds/mtp}
    ORCHESTRATOR_LOG_ROOT=${ORCHESTRATOR_LOG_ROOT:-$SHARED_ROOT/spec_train/logs/orchestrator}
    CONTAINER_MODE=${CONTAINER_MODE:-create}
    EXISTING_CONTAINER_NAME=${EXISTING_CONTAINER_NAME:-}
    DASHBOARD_HOST=${DASHBOARD_HOST:-0.0.0.0}
    DASHBOARD_PORT=${DASHBOARD_PORT:-6007}
    DASHBOARD_AUTO_START=${DASHBOARD_AUTO_START:-1}
    DASHBOARD_LOG_FILE=${DASHBOARD_LOG_FILE:-$ORCHESTRATOR_LOG_ROOT/mtp-dashboard.log}
    DASHBOARD_PID_FILE=${DASHBOARD_PID_FILE:-$ORCHESTRATOR_LOG_ROOT/mtp-dashboard.pid}
    OFFLINE_COLLECTION_CONCURRENCY=${OFFLINE_COLLECTION_CONCURRENCY:-1}
    OFFLINE_COLLECTION_MAX_SAMPLES=${OFFLINE_COLLECTION_MAX_SAMPLES:-0}
    OFFLINE_VALIDATION_SAMPLES=${OFFLINE_VALIDATION_SAMPLES:-8}
}

require_trainers() {
    ((${#CLUSTER_TRAINER_IPS[@]} == 2 || ${#CLUSTER_TRAINER_IPS[@]} == 4)) || \
        fail "this command requires exactly two or four trainer_ips; use a training YAML"
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

remote_test() {
    local host=$1
    shift
    if [[ ${MANAGER_DRY_RUN:-0} == 1 ]]; then
        return 1
    fi
    ssh_args
    ssh "${SSH_ARGS[@]}" "${SSH_USER:-root}@${host}" "$@" \
        >/dev/null 2>&1
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
    local verifier_hosts verifier_host
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
        collector)
            command+=(
                "VERIFIER_ID=$index"
                "VERIFIER_HOST=$host"
                "OFFLINE_COLLECTION_RANK=$index"
                "OFFLINE_COLLECTION_WORLD_SIZE=${#CLUSTER_VERIFIER_IPS[@]}"
            )
            ;;
        smoke | trainer)
            verifier_hosts=$(verifier_hosts_for_trainer "$index")
            verifier_host=${verifier_hosts%%,*}
            command+=(
                "NODE_RANK=$index"
                "NNODES=${#CLUSTER_TRAINER_IPS[@]}"
                "MASTER_ADDR=${CLUSTER_TRAINER_IPS[0]}"
                "VERIFIER_HOST=$verifier_host"
                "VERIFIER_HOSTS=$verifier_hosts"
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

verifier_health_code() {
    local host=$1
    curl --noproxy '*' -sS -m 2 -o /dev/null -w '%{http_code}' \
        "http://${host}:${VERIFIER_PORT}/health" 2>/dev/null || true
}

verifier_job_is_running() {
    local index=$1
    local host=${CLUSTER_VERIFIER_IPS[$index]}
    local container
    container=$(container_for verifier "$index")
    if [[ $CONTAINER_MODE == existing ]]; then
        local pid_file="$LOG_ROOT/runtime_pids/$container.verifier$index.pid"
        local check_command
        check_command="docker exec $(quote "$container") bash -lc "
        check_command+=$(quote "test -s $pid_file && pid=\$(cat $pid_file) && [[ \$pid =~ ^[0-9]+$ ]] && kill -0 \"\$pid\" 2>/dev/null")
        remote_test "$host" "$check_command"
    else
        remote_test "$host" \
            "docker inspect -f '{{.State.Running}}' $(quote "$container") 2>/dev/null | grep -qx true"
    fi
}

verifier_container_exists() {
    local index=$1
    local host=${CLUSTER_VERIFIER_IPS[$index]}
    local container
    container=$(container_for verifier "$index")
    remote_test "$host" \
        "docker inspect $(quote "$container") >/dev/null 2>&1"
}

verifier_container_is_running() {
    local index=$1
    local host=${CLUSTER_VERIFIER_IPS[$index]}
    local container
    container=$(container_for verifier "$index")
    remote_test "$host" \
        "docker inspect -f '{{.State.Running}}' $(quote "$container") 2>/dev/null | grep -qx true"
}

ensure_verifier() {
    local index=$1
    local host=${CLUSTER_VERIFIER_IPS[$index]}
    local container code
    container=$(container_for verifier "$index")
    code=$(verifier_health_code "$host")
    if [[ $code == 200 ]]; then
        printf '[skip] verifier=%s host=%s container=%s reason=healthy\n' \
            "$index" "$host" "$container"
        return
    fi
    if verifier_job_is_running "$index"; then
        printf '[skip] verifier=%s host=%s container=%s reason=process-running health=%s\n' \
            "$index" "$host" "$container" "${code:-unreachable}"
        printf 'WARNING: verifier %s is running but not healthy; wait or inspect its log before restarting it.\n' \
            "$index" >&2
        return
    fi
    if verifier_container_exists "$index" && \
        ! verifier_container_is_running "$index"; then
        printf '[start-container] verifier=%s host=%s container=%s\n' \
            "$index" "$host" "$container"
        remote "$host" "docker start $(quote "$container") >/dev/null"
        if [[ $CONTAINER_MODE == existing ]]; then
            start_node "$host" verifier "$index"
        fi
        return
    fi
    start_node "$host" verifier "$index"
}

ensure_verifiers() {
    local index
    for index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
        ensure_verifier "$index"
    done
}

start_collectors() {
    local index code
    if [[ ${MANAGER_DRY_RUN:-0} != 1 ]]; then
        for index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
            code=$(verifier_health_code "${CLUSTER_VERIFIER_IPS[$index]}")
            [[ $code == 200 ]] || \
                fail "verifier $index is not healthy (HTTP ${code:-000}); run start-verifiers and wait-verifiers first"
        done
    fi
    start_group collector "" CLUSTER_VERIFIER_IPS
}

wait_verifiers() {
    local timeout=${HEALTH_TIMEOUT:-7200}
    local started=$SECONDS
    local -a ready=()
    local index
    for index in "${!CLUSTER_VERIFIER_IPS[@]}"; do
        ready[$index]=0
    done
    while :; do
        local remaining=0 code
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

dashboard_advertise_host() {
    if [[ -n ${DASHBOARD_ADVERTISE_HOST:-} ]]; then
        printf '%s' "$DASHBOARD_ADVERTISE_HOST"
    elif [[ $DASHBOARD_HOST != 0.0.0.0 && $DASHBOARD_HOST != :: ]]; then
        printf '%s' "$DASHBOARD_HOST"
    else
        local -a addresses=()
        read -r -a addresses <<<"$(hostname -I 2>/dev/null || true)"
        printf '%s' "${addresses[0]:-127.0.0.1}"
    fi
}

dashboard_url() {
    printf 'http://%s:%s' "$(dashboard_advertise_host)" "$DASHBOARD_PORT"
}

dashboard_running_pid() {
    [[ -f $DASHBOARD_PID_FILE ]] || return 1
    local owner pid
    read -r owner pid <"$DASHBOARD_PID_FILE" || return 1
    [[ $owner == "$(hostname)" && $pid =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    local command_line
    command_line=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    [[ $command_line == *"$DASHBOARD_SCRIPT"* ]] || return 1
    printf '%s' "$pid"
}

start_dashboard() {
    local force=${1:-0}
    if [[ $force != 1 && $DASHBOARD_AUTO_START != 1 ]]; then
        printf '[dashboard] automatic startup disabled\n'
        return
    fi
    [[ -f $DASHBOARD_SCRIPT ]] || fail "dashboard script not found: $DASHBOARD_SCRIPT"
    local url
    url=$(dashboard_url)
    if [[ ${MANAGER_DRY_RUN:-0} == 1 ]]; then
        printf '[dashboard-dry-run] %q %q --config %q --host %q --port %q\n' \
            "${DASHBOARD_PYTHON:-python3}" "$DASHBOARD_SCRIPT" "$CONFIG_FILE" \
            "$DASHBOARD_HOST" "$DASHBOARD_PORT"
        printf 'DASHBOARD_URL=%s\n' "$url"
        return
    fi
    local pid
    if pid=$(dashboard_running_pid); then
        printf '[dashboard] already running pid=%s\n' "$pid"
        printf 'DASHBOARD_URL=%s\nDASHBOARD_LOG=%s\n' "$url" "$DASHBOARD_LOG_FILE"
        return
    fi
    mkdir -p -- "$ORCHESTRATOR_LOG_ROOT"
    rm -f -- "$DASHBOARD_PID_FILE"
    nohup "${DASHBOARD_PYTHON:-python3}" "$DASHBOARD_SCRIPT" \
        --config "$CONFIG_FILE" --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT" \
        >"$DASHBOARD_LOG_FILE" 2>&1 </dev/null &
    pid=$!
    printf '%s %s\n' "$(hostname)" "$pid" >"$DASHBOARD_PID_FILE"
    local health_host=127.0.0.1
    if [[ $DASHBOARD_HOST != 0.0.0.0 && $DASHBOARD_HOST != :: ]]; then
        health_host=$DASHBOARD_HOST
    fi
    local attempt code=000
    for attempt in {1..25}; do
        code=$(curl --noproxy '*' -sS -m 1 -o /dev/null -w '%{http_code}' \
            "http://$health_host:$DASHBOARD_PORT/health" 2>/dev/null || true)
        [[ $code == 200 ]] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    if [[ $code != 200 ]]; then
        printf 'WARNING: dashboard did not become healthy; inspect %s\n' \
            "$DASHBOARD_LOG_FILE" >&2
    fi
    printf '[dashboard] pid=%s\nDASHBOARD_URL=%s\nDASHBOARD_LOG=%s\n' \
        "$pid" "$url" "$DASHBOARD_LOG_FILE"
}

dashboard_status() {
    local pid
    if pid=$(dashboard_running_pid); then
        printf 'DASHBOARD_STATUS=running\nDASHBOARD_PID=%s\n' "$pid"
    else
        printf 'DASHBOARD_STATUS=stopped\n'
    fi
    printf 'DASHBOARD_URL=%s\nDASHBOARD_LOG=%s\n' \
        "$(dashboard_url)" "$DASHBOARD_LOG_FILE"
}

stop_dashboard() {
    local pid
    if ! pid=$(dashboard_running_pid); then
        rm -f -- "$DASHBOARD_PID_FILE"
        printf '[dashboard] already stopped\n'
        return
    fi
    printf '[dashboard-stop] pid=%s\n' "$pid"
    if [[ ${MANAGER_DRY_RUN:-0} != 1 ]]; then
        kill -TERM "$pid"
        local attempt
        for attempt in {1..25}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -0 "$pid" 2>/dev/null && \
            printf 'WARNING: dashboard pid %s is still running\n' "$pid" >&2
        rm -f -- "$DASHBOARD_PID_FILE"
    fi
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
    show_role_status collector CLUSTER_VERIFIER_IPS
    show_role_status trainer CLUSTER_TRAINER_IPS
    show_role_status smoke CLUSTER_TRAINER_IPS
    printf '\n[dashboard]\n'
    dashboard_status
}

show_offline_status() {
    show_role_status collector CLUSTER_VERIFIER_IPS
    local host=${CLUSTER_VERIFIER_IPS[0]}
    local container
    container=$(container_for verifier 0)
    printf '\n[offline-cache] host=%s path=%s\n' "$host" "$HIDDEN_STATES_PATH"
    local command
    command="find $HIDDEN_STATES_PATH -maxdepth 1 -name 'hs_*.safetensors' "
    command+="-printf x 2>/dev/null | wc -c; "
    command+="du -sh $HIDDEN_STATES_PATH 2>/dev/null || true"
    remote "$host" \
        "docker exec $(quote "$container") bash -lc $(quote "$command")" || true
}

verify_offline_cache() {
    local host=${CLUSTER_VERIFIER_IPS[0]}
    local container
    container=$(container_for verifier 0)
    local command="export PYTHONPATH=$(quote "$CONTAINER_REPO_PATH/src"):\${PYTHONPATH:-}; "
    command+="python $(quote "$CONTAINER_REPO_PATH/scripts/check_offline_hidden_states.py") "
    command+="--data $(quote "$DATA_PATH") --hidden-states $(quote "$HIDDEN_STATES_PATH") "
    # Training consumes the complete dataset even when collection was capped
    # for a pilot, so readiness must never accept only a sampled prefix.
    command+="--max-samples 0 "
    command+="--validate-samples $(quote "${OFFLINE_VALIDATION_SAMPLES:-8}") --write-ready-marker"
    remote "$host" \
        "docker exec $(quote "$container") bash -lc $(quote "$command")"
}

stop_role() {
    local role=$1
    local -n addresses=$2
    local index host container
    local stop_grace=${STOP_GRACE_SECONDS:-15}
    [[ $stop_grace =~ ^[0-9]+$ ]] || \
        fail "STOP_GRACE_SECONDS must be a non-negative integer"
    for index in "${!addresses[@]}"; do
        host=${addresses[$index]}
        container=$(container_for "$role" "$index")
        printf '[stop] host=%s container=%s\n' "$host" "$container"
        if [[ $CONTAINER_MODE == existing ]]; then
            local pid_file="$LOG_ROOT/runtime_pids/$container.$role$index.pid"
            local stop_command
            stop_command="docker exec $(quote "$container") bash -lc "
            stop_command+=$(quote "if [[ -f $pid_file ]]; then pid=\$(cat $pid_file); if [[ \$pid =~ ^[0-9]+$ ]] && kill -0 \"\$pid\" 2>/dev/null; then kill -TERM \"\$pid\"; for ((i=0; i<$stop_grace; i++)); do kill -0 \"\$pid\" 2>/dev/null || break; sleep 1; done; if kill -0 \"\$pid\" 2>/dev/null; then echo 'WARNING: job still running after $stop_grace seconds; container restart will clean it: pid='\"\$pid\" >&2; fi; fi; rm -f -- $pid_file; fi")
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

restart_existing_role() {
    local role=$1
    local -n addresses=$2
    [[ $CONTAINER_MODE == existing ]] || \
        fail "restart-$role is supported only when container_mode is existing"

    local index host container
    for index in "${!addresses[@]}"; do
        host=${addresses[$index]}
        container=$(container_for "$role" "$index")
        printf '[restart] host=%s container=%s\n' "$host" "$container"
        remote "$host" \
            "docker restart -t 30 $(quote "$container") >/dev/null"
    done
}

restart_verifiers() {
    stop_role collector CLUSTER_VERIFIER_IPS
    stop_role verifier CLUSTER_VERIFIER_IPS
    restart_existing_role verifier CLUSTER_VERIFIER_IPS
}

restart_trainers() {
    stop_role trainer CLUSTER_TRAINER_IPS
    stop_role smoke CLUSTER_TRAINER_IPS
    restart_existing_role trainer CLUSTER_TRAINER_IPS
}

stop_cluster() {
    stop_role collector CLUSTER_VERIFIER_IPS
    stop_role verifier CLUSTER_VERIFIER_IPS
    stop_role trainer CLUSTER_TRAINER_IPS
    stop_role smoke CLUSTER_TRAINER_IPS
    if [[ $CONTAINER_MODE == existing ]]; then
        restart_existing_role verifier CLUSTER_VERIFIER_IPS
        restart_existing_role trainer CLUSTER_TRAINER_IPS
    fi
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
    start-verifiers)
        ensure_verifiers
        start_dashboard
        ;;
    start-verifier)
        validate_verifier_index
        ensure_verifier "$TARGET_INDEX"
        start_dashboard
        ;;
    wait-verifiers)
        start_dashboard
        wait_verifiers
        ;;
    collect-offline)
        start_collectors
        start_dashboard
        ;;
    offline-status) show_offline_status ;;
    verify-offline) verify_offline_cache ;;
    smoke)
        require_trainers
        start_group smoke online-cache CLUSTER_TRAINER_IPS
        start_dashboard
        ;;
    train)
        require_trainers
        start_group trainer online-cache CLUSTER_TRAINER_IPS
        start_dashboard
        ;;
    offline)
        require_trainers
        if [[ ${MANAGER_DRY_RUN:-0} != 1 ]]; then
            verify_offline_cache
        else
            printf '[offline] dry-run skips cache validation\n'
        fi
        start_group trainer offline CLUSTER_TRAINER_IPS
        start_dashboard
        ;;
    status) show_status ;;
    dashboard) start_dashboard 1 ;;
    dashboard-status) dashboard_status ;;
    stop-dashboard) stop_dashboard ;;
    restart-verifiers) restart_verifiers ;;
    restart-trainers) require_trainers; restart_trainers ;;
    stop-collectors) stop_role collector CLUSTER_VERIFIER_IPS ;;
    stop) stop_cluster ;;
    -h | --help | help) usage ;;
    *) fail "unknown command: $COMMAND" ;;
esac
