#!/usr/bin/env bash
# Run an isolated two-step online smoke without building a persistent HS cache.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
MANAGER=${MANAGER:-$SCRIPT_DIR/manage_mtp_glm52_ascend_online_4v4t.sh}
SOURCE_CONFIG=${SOURCE_CONFIG:-/mnt/xds/mtp/spec_train/config/mtp_glm52_production_4v4t_4k.yaml}
SMOKE_CONFIG=${SMOKE_CONFIG:-/mnt/xds/mtp/spec_train/config/mtp_glm52_quick_online_smoke.yaml}
SMOKE_SEQ_LEN=${SMOKE_SEQ_LEN:-1024}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-256}
SMOKE_RUN_ID=${SMOKE_RUN_ID:-quick-${SMOKE_SEQ_LEN}-$(date +%Y%m%d-%H%M%S)}

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND

Commands:
  render  Generate and validate the isolated online-smoke YAML.
  run     Stop offline collectors, restart verifiers with the smoke YAML,
          and start a two-step online smoke on all configured trainers.
  status  Show verifier and smoke logs using the generated YAML.
  stop    Stop smoke trainer jobs by restarting reused trainer containers.

Environment:
  SOURCE_CONFIG=$SOURCE_CONFIG
  SMOKE_CONFIG=$SMOKE_CONFIG
  SMOKE_SEQ_LEN=$SMOKE_SEQ_LEN       # use 4096 after the 1024-token test
  SMOKE_SAMPLES=$SMOKE_SAMPLES
  SMOKE_RUN_ID=$SMOKE_RUN_ID         # must be unique for every run

Smoke generation still uses a shared request-scoped file because verifier and
trainer run on different hosts. The trainer reads and deletes every generated
file (--on-generate delete); the production offline cache is never touched.
EOF
}

require_source() {
    [[ -f $SOURCE_CONFIG ]] || {
        printf 'ERROR: source production config does not exist: %s\n' \
            "$SOURCE_CONFIG" >&2
        exit 2
    }
}

render() {
    require_source
    python "$REPO_ROOT/scripts/make_ascend_mtp_online_smoke_config.py" \
        --source "$SOURCE_CONFIG" \
        --output "$SMOKE_CONFIG" \
        --run-id "$SMOKE_RUN_ID" \
        --smoke-seq-len "$SMOKE_SEQ_LEN" \
        --smoke-samples "$SMOKE_SAMPLES"
    bash "$MANAGER" validate-config --config "$SMOKE_CONFIG"
}

run() {
    render
    printf '[quick-smoke] stopping resumable offline collectors from %s\n' \
        "$SOURCE_CONFIG"
    bash "$MANAGER" stop-collectors --config "$SOURCE_CONFIG"
    printf '[quick-smoke] restarting verifiers with shared ephemeral output\n'
    bash "$MANAGER" restart-verifiers --config "$SMOKE_CONFIG"
    bash "$MANAGER" start-verifiers --config "$SMOKE_CONFIG"
    bash "$MANAGER" wait-verifiers --config "$SMOKE_CONFIG"
    printf '[quick-smoke] starting online %s-token, two-step training smoke\n' \
        "$SMOKE_SEQ_LEN"
    bash "$MANAGER" smoke --config "$SMOKE_CONFIG"
    bash "$MANAGER" status --config "$SMOKE_CONFIG"
    printf '\nQUICK_SMOKE_CONFIG=%s\n' "$SMOKE_CONFIG"
    printf 'Monitor: bash %q status\n' "$0"
}

COMMAND=${1:-}
case "$COMMAND" in
    render) render ;;
    run) run ;;
    status)
        [[ -f $SMOKE_CONFIG ]] || {
            printf 'ERROR: generated smoke config is missing: %s\n' \
                "$SMOKE_CONFIG" >&2
            exit 2
        }
        bash "$MANAGER" status --config "$SMOKE_CONFIG"
        ;;
    stop)
        [[ -f $SMOKE_CONFIG ]] || {
            printf 'ERROR: generated smoke config is missing: %s\n' \
                "$SMOKE_CONFIG" >&2
            exit 2
        }
        bash "$MANAGER" restart-trainers --config "$SMOKE_CONFIG"
        ;;
    -h | --help | help) usage ;;
    *) usage >&2; exit 2 ;;
esac
