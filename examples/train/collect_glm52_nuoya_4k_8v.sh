#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANAGER=${MANAGER:-$SCRIPT_DIR/manage_mtp_glm52_ascend_online_4v4t.sh}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-$SCRIPT_DIR/prepare_glm52_nuoya_4k.sh}
CONFIG=${CONFIG:-/mnt/xds/mtp/spec_train/config/glm52-mtp3-offline-8v-4k.yaml}

usage() {
    cat <<EOF
Usage: CONFIG=/path/to/4k-8v.yaml $(basename "$0") COMMAND

Commands:
  prepare              Build the six-file Nuoya 4K Hugging Face dataset.
  collect              Validate config, ensure all eight verifiers are healthy,
                       and start/resume eight non-overlapping collectors.
  prepare-and-collect  Run prepare followed by collect.
  status               Show collector processes, recent logs, and cache count.
  verify               Validate complete cache and publish .offline-ready.json.
  stop                 Stop collectors without stopping healthy verifiers.

The config should be copied from:
  $SCRIPT_DIR/mtp_glm52_ascend_offline_collect_8v_4k.example.yaml
EOF
}

require_config() {
    [[ -f $CONFIG ]] || {
        printf 'ERROR: config does not exist: %s\n' "$CONFIG" >&2
        printf 'Copy and edit: %s\n' \
            "$SCRIPT_DIR/mtp_glm52_ascend_offline_collect_8v_4k.example.yaml" >&2
        exit 2
    }
}

collect() {
    require_config
    bash "$MANAGER" validate-config --config "$CONFIG"
    bash "$MANAGER" start-verifiers --config "$CONFIG"
    bash "$MANAGER" wait-verifiers --config "$CONFIG"
    bash "$MANAGER" collect-offline --config "$CONFIG"
    bash "$MANAGER" offline-status --config "$CONFIG"
}

COMMAND=${1:-}
case "$COMMAND" in
    prepare)
        bash "$PREPARE_SCRIPT"
        ;;
    collect)
        collect
        ;;
    prepare-and-collect)
        bash "$PREPARE_SCRIPT"
        collect
        ;;
    status)
        require_config
        bash "$MANAGER" offline-status --config "$CONFIG"
        ;;
    verify)
        require_config
        bash "$MANAGER" verify-offline --config "$CONFIG"
        ;;
    stop)
        require_config
        bash "$MANAGER" stop-collectors --config "$CONFIG"
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
