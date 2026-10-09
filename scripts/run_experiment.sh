#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${1:-configs/mnist_cnn.yaml}
RUN_NAME=$(grep -m1 '^run_name:' "$CONFIG" | awk '{print $2}')

if [ -z "$RUN_NAME" ]; then
  echo "could not read run_name from $CONFIG" >&2
  exit 1
fi

QUANTLAB="quantlab"
if [ -x ".venv/bin/quantlab" ]; then
  QUANTLAB=".venv/bin/quantlab"
fi

"$QUANTLAB" train --config "$CONFIG"
"$QUANTLAB" quantize --config "$CONFIG"
"$QUANTLAB" benchmark --config "$CONFIG"
"$QUANTLAB" compare --results "benchmarks/${RUN_NAME}.json" --csv "benchmarks/${RUN_NAME}_comparison.csv"
