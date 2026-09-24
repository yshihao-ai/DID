#!/usr/bin/env bash

set -Eeuo pipefail

# All paths can be overridden through environment variables when needed.
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-/datanfs5/xxx/code/PyTorch-BigGraph/test1/did_bar_level}"
ACC_ROOT="${ACC_ROOT:-/datanfs5/xxx/code/PyTorch-BigGraph/test1/baseline0911/acc}"
POP909_ANNOTATION_ROOT="${POP909_ANNOTATION_ROOT:-${PROJECT_ROOT}/dataset/MIDI_dataset/Pop909}"
OUTPUT_DIR="${OUTPUT_DIR:-${ACC_ROOT}/eval_32bars}"

EVAL_SCRIPT="${PROJECT_ROOT}/smg_metric/eval.py"
METHODS=(
  "D3PIA"
  "FGG"
  "polyffusion"
  "WholeSongGen"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] File not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Directory not found: $path" >&2
    exit 1
  fi
}

require_file "$EVAL_SCRIPT"
require_dir "$ACC_ROOT"
require_dir "$POP909_ANNOTATION_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "Evaluation script : $EVAL_SCRIPT"
echo "Baseline root     : $ACC_ROOT"
echo "POP909 annotations: $POP909_ANNOTATION_ROOT"
echo "JSON output       : $OUTPUT_DIR"
echo

for method in "${METHODS[@]}"; do
  pred_dir="${ACC_ROOT}/${method}/predict"
  ref_dir="${ACC_ROOT}/${method}/answer"
  output_json="${OUTPUT_DIR}/${method}_32bars.json"

  require_dir "$pred_dir"
  require_dir "$ref_dir"

  echo "============================================================"
  echo "Evaluating ${method} (first 32 bars)"
  echo "Prediction: $pred_dir"
  echo "Reference : $ref_dir"
  echo "Output    : $output_json"

  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    "$pred_dir" \
    "$ref_dir" \
    "$output_json" \
    --max-bars 32 \
    --pop909-annotation-root "$POP909_ANNOTATION_ROOT"
done

echo
echo "==================== 32-bar results ===================="

"$PYTHON_BIN" - "$OUTPUT_DIR" "${METHODS[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path


output_dir = Path(sys.argv[1])
methods = sys.argv[2:]


def metric_percent(data, section, name):
    value = data["summary"][section][name]
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value * 100:.2f}"


print(f"{'Method':<18} {'OOK (%) ↓':>12} {'CA (%) ↑':>12} {'CS (%) ↑':>12}")
print("-" * 57)
for method in methods:
    json_path = output_dir / f"{method}_32bars.json"
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    ook = metric_percent(data, "A_single_file_quality", "ook")
    ca = metric_percent(data, "D_chord_level_pairwise", "ca")
    cs = metric_percent(data, "D_chord_level_pairwise", "cs")
    print(f"{method:<18} {ook:>12} {ca:>12} {cs:>12}")

print(f"\nJSON files: {output_dir}")
PY
