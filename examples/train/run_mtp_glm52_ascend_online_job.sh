#!/usr/bin/env bash
# Container-side job supervisor used by both docker run and docker exec modes.

set -euo pipefail

CONTAINER_REPO_PATH=${CONTAINER_REPO_PATH:-/workspace/speculators}
LOG_ROOT=${LOG_ROOT:-/kos_ulan/spec_train/logs/glm52-w4a8c8-mtp3}
CONTAINER_NAME=${CONTAINER_NAME:?CONTAINER_NAME is required}
ROLE=${ROLE:?ROLE is required}
INSTALL_SPECULATORS=${INSTALL_SPECULATORS:-0}

job_slot=${VERIFIER_ID:-${NODE_RANK:-0}}
pid_root=$LOG_ROOT/runtime_pids
pid_file=$pid_root/$CONTAINER_NAME.$ROLE$job_slot.pid
mkdir -p "$pid_root"

child_pid=
cleanup() {
    rm -f -- "$pid_file"
}
forward_signal() {
    if [[ -n $child_pid ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap forward_signal INT TERM

if [[ -f $pid_file ]]; then
    previous_pid=$(<"$pid_file")
    if [[ $previous_pid =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        printf 'ERROR: an MTP job is already running: pid=%s file=%s\n' \
            "$previous_pid" "$pid_file" >&2
        exit 2
    fi
    rm -f -- "$pid_file"
fi

if [[ $INSTALL_SPECULATORS == 1 ]]; then
    python -m pip install --no-build-isolation --no-deps \
        -e "$CONTAINER_REPO_PATH/hs_connectors" \
        -e "$CONTAINER_REPO_PATH"
fi

bash "$CONTAINER_REPO_PATH/examples/train/mtp_glm52_ascend_online.sh" &
child_pid=$!
printf '%s\n' "$child_pid" >"$pid_file"
set +e
wait "$child_pid"
status=$?
set -e
child_pid=
exit "$status"
