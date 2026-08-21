#!/usr/bin/env bash
# Start/resume four-node MTP3 training from the completed subset of a 4K cache.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANAGER=${MANAGER:-$SCRIPT_DIR/manage_mtp_glm52_ascend_online_4v4t.sh}
CONFIG=${CONFIG:-$SCRIPT_DIR/mtp_glm52_ascend_partial_offline_4v4t_4k.yaml}
PARTIAL_DATA_SCRIPT=${PARTIAL_DATA_SCRIPT:-$SCRIPT_DIR/prepare_glm52_partial_offline_4k.sh}

usage() {
    cat <<EOF
Usage: $(basename "$0") [train|check|validate-data|dry-run|status]

Commands:
  train          Validate the cache and start/resume all four trainers (default).
  check          Validate the YAML and filtered dataset without starting training.
  validate-data  Validate the filtered dataset/cache mapping only.
  dry-run        Print the remote trainer launch commands without executing them.
  status         Show containers and recent trainer host-wrapper logs.

Environment overrides:
  CONFIG=$CONFIG
  VALIDATE_SAMPLES=32   # use -1 to open and validate every selected cache file

This entry point never starts verifiers or collectors. The Manager's offline
command uses verifier0's existing container only to validate the shared cache.
EOF
}

require_files() {
    [[ -f $CONFIG ]] || {
        printf 'ERROR: training config is missing: %s\n' "$CONFIG" >&2
        exit 2
    }
    [[ -x $MANAGER || -f $MANAGER ]] || {
        printf 'ERROR: manager is missing: %s\n' "$MANAGER" >&2
        exit 2
    }
}

validate_data() {
    printf '[partial-offline] phase=data_validation status=started samples=%s\n' \
        "${VALIDATE_SAMPLES:-32}"
    VALIDATE_SAMPLES=${VALIDATE_SAMPLES:-32} \
        bash "$PARTIAL_DATA_SCRIPT" verify
    printf '[partial-offline] phase=data_validation status=completed\n'
}

train() {
    bash "$MANAGER" validate-config --config "$CONFIG"
    # `offline` checks every source_index has a final cache file and then starts
    # torchrun with --on-missing raise. No HTTP endpoint is contacted.
    bash "$MANAGER" offline --config "$CONFIG"
}

require_files
COMMAND=${1:-train}
case "$COMMAND" in
    train)
        train
        ;;
    check)
        bash "$MANAGER" validate-config --config "$CONFIG"
        validate_data
        ;;
    validate-data)
        validate_data
        ;;
    dry-run)
        MANAGER_DRY_RUN=1 bash "$MANAGER" offline --config "$CONFIG"
        ;;
    status)
        bash "$MANAGER" status --config "$CONFIG"
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
