#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
MODEL_CONFIG="${1:-configs/models/paper_llama2_7b_chat.yaml}"
SUSPECT_MODEL_CONFIG="${2:-${MODEL_CONFIG}}"
OUT_DIR="${3:-results/smoke}"
RUNTIME_CSV="${OUT_DIR}/tables/smoke_runtime.csv"

mkdir -p "${OUT_DIR}/fingerprints" "${OUT_DIR}/runs" "${OUT_DIR}/tables" "${OUT_DIR}/logs"
printf 'stage,method,start_time,end_time,seconds,exit_code,command\n' > "${RUNTIME_CSV}"

run_timed() {
  local stage="$1"
  local method="$2"
  shift 2

  local logfile="${OUT_DIR}/logs/${method}_${stage}.log"
  local start_time
  local end_time
  local start_seconds
  local end_seconds
  local elapsed
  local exit_code
  local command_string

  command_string="$*"
  command_string="${command_string//\"/\"\"}"
  start_time="$(date -Is)"
  start_seconds="$(date +%s)"

  set +e
  "$@" > "${logfile}" 2>&1
  exit_code=$?
  set -e

  end_seconds="$(date +%s)"
  end_time="$(date -Is)"
  elapsed=$((end_seconds - start_seconds))
  printf '%s,%s,%s,%s,%s,%s,"%s"\n' \
    "${stage}" "${method}" "${start_time}" "${end_time}" "${elapsed}" "${exit_code}" "${command_string}" \
    >> "${RUNTIME_CSV}"

  if [ "${exit_code}" -ne 0 ]; then
    echo "Smoke ${stage} for ${method} failed. See ${logfile}" >&2
    exit "${exit_code}"
  fi
}

run_timed construct trap \
  "${PYTHON_BIN}" -m llmfp.cli construct \
  --method trap \
  --method-config configs/methods/smoke_trap.yaml \
  --model-config "${MODEL_CONFIG}" \
  --out "${OUT_DIR}/fingerprints/trap.jsonl"

run_timed verify trap \
  "${PYTHON_BIN}" -m llmfp.cli verify \
  --method trap \
  --method-config configs/methods/smoke_trap.yaml \
  --suspect-model-config "${SUSPECT_MODEL_CONFIG}" \
  --fingerprints "${OUT_DIR}/fingerprints/trap.jsonl" \
  --out "${OUT_DIR}/runs/trap_verify.jsonl"

run_timed construct proflingo \
  "${PYTHON_BIN}" -m llmfp.cli construct \
  --method proflingo \
  --method-config configs/methods/smoke_proflingo.yaml \
  --model-config "${MODEL_CONFIG}" \
  --out "${OUT_DIR}/fingerprints/proflingo.jsonl"

run_timed verify proflingo \
  "${PYTHON_BIN}" -m llmfp.cli verify \
  --method proflingo \
  --method-config configs/methods/smoke_proflingo.yaml \
  --suspect-model-config "${SUSPECT_MODEL_CONFIG}" \
  --fingerprints "${OUT_DIR}/fingerprints/proflingo.jsonl" \
  --out "${OUT_DIR}/runs/proflingo_verify.jsonl"

run_timed construct sraf \
  "${PYTHON_BIN}" -m llmfp.cli construct \
  --method sraf \
  --method-config configs/methods/smoke_sraf.yaml \
  --model-config "${MODEL_CONFIG}" \
  --out "${OUT_DIR}/fingerprints/sraf.jsonl"

run_timed verify sraf \
  "${PYTHON_BIN}" -m llmfp.cli verify \
  --method sraf \
  --method-config configs/methods/smoke_sraf.yaml \
  --suspect-model-config "${SUSPECT_MODEL_CONFIG}" \
  --fingerprints "${OUT_DIR}/fingerprints/sraf.jsonl" \
  --out "${OUT_DIR}/runs/sraf_verify.jsonl"

run_timed construct llmprint \
  "${PYTHON_BIN}" -m llmfp.cli construct \
  --method llmprint \
  --method-config configs/methods/smoke_llmprint.yaml \
  --model-config "${MODEL_CONFIG}" \
  --out "${OUT_DIR}/fingerprints/llmprint.jsonl"

run_timed verify llmprint \
  "${PYTHON_BIN}" -m llmfp.cli verify \
  --method llmprint \
  --method-config configs/methods/smoke_llmprint.yaml \
  --suspect-model-config "${SUSPECT_MODEL_CONFIG}" \
  --fingerprints "${OUT_DIR}/fingerprints/llmprint.jsonl" \
  --out "${OUT_DIR}/runs/llmprint_verify.jsonl"

run_timed summarize all \
  "${PYTHON_BIN}" -m llmfp.cli summarize \
  --results-dir "${OUT_DIR}/runs" \
  --out "${OUT_DIR}/tables/summary.csv"

echo "Smoke test finished."
echo "Runtime CSV: ${RUNTIME_CSV}"
echo "Summary CSV: ${OUT_DIR}/tables/summary.csv"
