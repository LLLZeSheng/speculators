#!/usr/bin/env bash
# Create a non-invasive Ascend ModelSlim runtime view of GLM-5.2-W4A8-MG13.
# Only config.json is copied and changed; model weights remain absolute symlinks.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_MODEL=${SOURCE_MODEL:-/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1}
RUNTIME_MODEL=${RUNTIME_MODEL:-/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

if (( $# == 1 )); then
    printf 'SOURCE_MODEL=%s\n' "$SOURCE_MODEL"
    printf 'RUNTIME_MODEL=%s\n' "$RUNTIME_MODEL"
    printf 'ACTION=create ModelSlim runtime view and set quant_method=ascend\n'
    exit 0
fi

[[ -d $SOURCE_MODEL ]] || fail "source model does not exist: $SOURCE_MODEL"
[[ -f $SOURCE_MODEL/config.json ]] || fail "missing source config.json"
[[ -f $SOURCE_MODEL/quant_model_description.json ]] || \
    fail "missing quant_model_description.json; this is not a complete ModelSlim model"
[[ $SOURCE_MODEL != "$RUNTIME_MODEL" ]] || fail "source and runtime paths must differ"

if [[ -f $RUNTIME_MODEL/.ready ]]; then
    detected=$(
        "$PYTHON_BIN" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["quantization_config"]["quant_method"])' \
            "$RUNTIME_MODEL/config.json"
    )
    [[ $detected == ascend ]] || fail "existing runtime model has quant_method=$detected"
    printf 'RUNTIME_MODEL_PATH=%s\n' "$RUNTIME_MODEL"
    printf 'QUANT_METHOD=ascend\n'
    printf 'PREPARE_STATUS=already-ready\n'
    exit 0
fi

[[ ! -e $RUNTIME_MODEL ]] || \
    fail "runtime path exists but is incomplete (no .ready): $RUNTIME_MODEL"

runtime_parent=$(dirname -- "$RUNTIME_MODEL")
runtime_name=$(basename -- "$RUNTIME_MODEL")
mkdir -p -- "$runtime_parent"
temporary=$(mktemp -d --tmpdir="$runtime_parent" ".${runtime_name}.tmp.XXXXXXXX")

cleanup() {
    if [[ -d ${temporary:-} ]]; then
        rm -rf -- "$temporary"
    fi
}
trap cleanup EXIT

shopt -s dotglob nullglob
for source_item in "$SOURCE_MODEL"/*; do
    name=$(basename -- "$source_item")
    case "$name" in
        config.json | .ready | runtime_model_manifest.json)
            continue
            ;;
    esac
    ln -s -- "$source_item" "$temporary/$name"
done
shopt -u dotglob nullglob

cp -- "$SOURCE_MODEL/config.json" "$temporary/config.json"

"$PYTHON_BIN" -c '
import json
import sys

path = sys.argv[1]
source = sys.argv[2]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
quant = config.get("quantization_config")
if not isinstance(quant, dict):
    raise SystemExit("config.json has no quantization_config object")
original = quant.get("quant_method")
quant["quant_method"] = "ascend"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
manifest = {
    "source_model": source,
    "original_quant_method": original,
    "runtime_quant_method": "ascend",
    "weight_format": "Ascend ModelSlim W4A8",
    "note": "config-only runtime view; weights are symlinks",
}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
' "$temporary/config.json" "$SOURCE_MODEL" "$temporary/runtime_model_manifest.json"

touch -- "$temporary/.ready"
mv -- "$temporary" "$RUNTIME_MODEL"
trap - EXIT

printf 'RUNTIME_MODEL_PATH=%s\n' "$RUNTIME_MODEL"
printf 'QUANT_METHOD=ascend\n'
printf 'PREPARE_STATUS=created\n'
printf '\nStart vLLM with this path and: --quantization ascend\n'
