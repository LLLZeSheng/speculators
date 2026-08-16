#!/usr/bin/env bash
# Create a non-invasive Ascend ModelSlim runtime view of GLM-5.2-W4A8-MG13.
# Only config.json is copied and changed; model weights remain absolute symlinks.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE_MODEL=${SOURCE_MODEL:-/mnt/xds/sfs/GLM-5.2-W4A8-MG13/v1}
RUNTIME_MODEL=${RUNTIME_MODEL:-/mnt/xds/sfs/l00936201/glm52-w4a8-mg13/v1-ascend-modelslim-v3}

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
    printf 'ACTION=create ModelSlim runtime view and defer quantization to --quantization ascend\n'
    exit 0
fi

[[ -d $SOURCE_MODEL ]] || fail "source model does not exist: $SOURCE_MODEL"
[[ -f $SOURCE_MODEL/config.json ]] || fail "missing source config.json"
[[ -f $SOURCE_MODEL/quant_model_description.json ]] || \
    fail "missing quant_model_description.json; this is not a complete ModelSlim model"
[[ $SOURCE_MODEL != "$RUNTIME_MODEL" ]] || fail "source and runtime paths must differ"

if [[ -f $RUNTIME_MODEL/.ready ]]; then
    has_embedded_quant=$(
        "$PYTHON_BIN" -c \
            'import json,sys; print("yes" if "quantization_config" in json.load(open(sys.argv[1])) else "no")' \
            "$RUNTIME_MODEL/config.json"
    )
    [[ $has_embedded_quant == no ]] || \
        fail "existing runtime model still embeds quantization_config"
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
        config.json | quant_model_description.json | .ready | runtime_model_manifest.json)
            continue
            ;;
    esac
    ln -s -- "$source_item" "$temporary/$name"
done
shopt -u dotglob nullglob

cp -- "$SOURCE_MODEL/config.json" "$temporary/config.json"
cp -- "$SOURCE_MODEL/quant_model_description.json" \
    "$temporary/quant_model_description.json"

"$PYTHON_BIN" -c '
import json
import sys

path = sys.argv[1]
source = sys.argv[2]
description_path = sys.argv[3]
manifest_path = sys.argv[4]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
quant = config.get("quantization_config")
original = quant.get("quant_method") if isinstance(quant, dict) else None
# AscendModelSlimConfig.from_config() treats any embedded quantization_config
# as the actual per-layer quant_description.  This source model embeds a
# compressed-tensors schema there, while its real ModelSlim layer map lives in
# quant_model_description.json.  Remove the embedded block so
# --quantization ascend creates an empty ModelSlim config and then loads the
# dedicated description file via maybe_update_config().
config.pop("quantization_config", None)

# ModelSlim currently omits unquantized DeepSeek-style MTP layers from
# quant_model_description.json. vLLM still queries those layer prefixes while
# constructing the drafter, so register every on-disk MTP weight as FLOAT.
# Refuse the rewrite if an MTP quantization auxiliary tensor exists: that would
# indicate genuinely quantized MTP weights and must not be mislabeled FLOAT.
import glob
import re

with open(description_path, encoding="utf-8") as handle:
    description = json.load(handle)
index_candidates = sorted(glob.glob(source + "/*.safetensors.index.json"))
if not index_candidates:
    raise SystemExit("no safetensors index found in source model")
with open(index_candidates[0], encoding="utf-8") as handle:
    weight_map = json.load(handle).get("weight_map", {})

num_hidden_layers = int(config["num_hidden_layers"])
num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
mtp_end = num_hidden_layers + num_mtp_layers
layer_pattern = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")
mtp_keys = []
for name in weight_map:
    match = layer_pattern.search(name)
    if match and num_hidden_layers <= int(match.group(1)) < mtp_end:
        mtp_keys.append(name)

quant_aux_suffixes = (
    ".weight_scale",
    ".weight_offset",
    ".scale_bias",
    ".input_scale",
    ".input_offset",
)
unexpected_quantized = [
    name for name in mtp_keys if name.endswith(quant_aux_suffixes)
]
if unexpected_quantized:
    raise SystemExit(
        "MTP layers contain quantization auxiliaries and cannot be marked FLOAT: "
        + unexpected_quantized[0]
    )

mtp_float_entries = 0
for name in mtp_keys:
    if name.endswith(".weight") and name not in description:
        description[name] = "FLOAT"
        mtp_float_entries += 1
with open(description_path, "w", encoding="utf-8") as handle:
    json.dump(description, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
manifest = {
    "source_model": source,
    "original_quant_method": original,
    "runtime_quant_method": "ascend (selected by CLI)",
    "weight_format": "Ascend ModelSlim W4A8",
    "mtp_float_entries_added": mtp_float_entries,
    "note": "config-only runtime view; weights are symlinks",
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
' "$temporary/config.json" "$SOURCE_MODEL" \
    "$temporary/quant_model_description.json" \
    "$temporary/runtime_model_manifest.json"

touch -- "$temporary/.ready"
mv -- "$temporary" "$RUNTIME_MODEL"
trap - EXIT

printf 'RUNTIME_MODEL_PATH=%s\n' "$RUNTIME_MODEL"
printf 'QUANT_METHOD=ascend\n'
printf 'PREPARE_STATUS=created\n'
printf '\nStart vLLM with this path and: --quantization ascend\n'
