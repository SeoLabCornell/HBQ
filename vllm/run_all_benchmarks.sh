#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-run_logs}"

formats=(
  "mxfp-w4a8kv8:config/mxfp_w4a8kv8.yaml"
  "nvfp-w4a8kv4:config/nvfp_w4a8kv4.yaml"
  "hbq-a-kv:config/hbq_a_new_kv.yaml"
  "hbq-e-kv:config/hbq_e_new_kv.yaml"
)

benchmarks=(
  "gsm8k:config/llama-8b-distill_gsm8k.yaml"
  "math500:config/llama-8b-distill_math500.yaml"
)

mkdir -p "$LOG_ROOT"

total=$((${#formats[@]} * ${#benchmarks[@]}))
run_number=0
failed=()

for format_entry in "${formats[@]}"; do
  format_name="${format_entry%%:*}"
  quant_config="${format_entry#*:}"

  for benchmark_entry in "${benchmarks[@]}"; do
    benchmark_name="${benchmark_entry%%:*}"
    model_config="${benchmark_entry#*:}"
    run_number=$((run_number + 1))

    log_dir="$LOG_ROOT/$format_name"
    log_file="$log_dir/$benchmark_name.log"
    mkdir -p "$log_dir"

    echo "[$run_number/$total] format=$format_name benchmark=$benchmark_name"
    echo "Log: $log_file"

    "$PYTHON_BIN" main.py \
      --config_dir "$model_config" \
      --quant_config "$quant_config" \
      "$@" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    if ((status == 0)); then
      echo "[$run_number/$total] PASS: $format_name / $benchmark_name"
    else
      failed+=("$format_name/$benchmark_name (exit $status)")
      echo "[$run_number/$total] FAIL: $format_name / $benchmark_name (exit $status)"
    fi
  done
done

echo
if ((${#failed[@]} == 0)); then
  echo "All $total runs completed successfully. Logs: $LOG_ROOT"
  exit 0
fi

echo "${#failed[@]} of $total runs failed:"
printf '  - %s\n' "${failed[@]}"
exit 1
