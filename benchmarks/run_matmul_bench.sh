#!/usr/bin/env bash
# Build Release (no sanitizers) and save matmul benchmark results for comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build-bench"
RESULTS_DIR="${ROOT}/benchmarks/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
JSON_OUT="${RESULTS_DIR}/matmul_${STAMP}.json"
TXT_OUT="${RESULTS_DIR}/matmul_${STAMP}.txt"
LATEST_JSON="${RESULTS_DIR}/matmul_latest.json"
LATEST_TXT="${RESULTS_DIR}/matmul_latest.txt"

mkdir -p "${RESULTS_DIR}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLEARN_BUILD_BENCHMARKS=ON

cmake --build "${BUILD_DIR}" --target benchmark_matmul --config Release -j

BENCH="${BUILD_DIR}/benchmark_matmul"
if [[ ! -x "${BENCH}" && -x "${BUILD_DIR}/Release/benchmark_matmul" ]]; then
  BENCH="${BUILD_DIR}/Release/benchmark_matmul"
fi

"${BENCH}" \
  --benchmark_out="${JSON_OUT}" \
  --benchmark_out_format=json \
  --benchmark_display_aggregates_only=false \
  | tee "${TXT_OUT}"

cp "${JSON_OUT}" "${LATEST_JSON}"
cp "${TXT_OUT}" "${LATEST_TXT}"

echo
echo "Saved:"
echo "  ${JSON_OUT}"
echo "  ${TXT_OUT}"
echo "  ${LATEST_JSON} (copy)"
echo "  ${LATEST_TXT} (copy)"
