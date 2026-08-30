#!/usr/bin/env bash
# Build Release (no sanitizers) and save matmul benchmark results for comparison.
#
# Usage:
#   ./benchmarks/run_matmul_bench.sh          # all benchmarks
#   ./benchmarks/run_matmul_bench.sh compare  # unblocked + blocked (block=128)
#   ./benchmarks/run_matmul_bench.sh sgemm    # unblocked vs single-thread cblas_sgemm
#   ./benchmarks/run_matmul_bench.sh mt       # multithreaded_matmul at 1024 (1/2/4/8 threads)
#   ./benchmarks/run_matmul_bench.sh register # register_optimized_matmul at all sizes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build-bench"
RESULTS_DIR="${ROOT}/benchmarks/results"
MODE="${1:-all}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JSON_OUT="${RESULTS_DIR}/matmul_${STAMP}.json"
TXT_OUT="${RESULTS_DIR}/matmul_${STAMP}.txt"
LATEST_JSON="${RESULTS_DIR}/matmul_latest.json"
LATEST_TXT="${RESULTS_DIR}/matmul_latest.txt"

BENCHMARK_FILTER=""
BENCH_ENV=()
if [[ "${MODE}" == "compare" ]]; then
  BENCHMARK_FILTER='--benchmark_filter=BM_MatrixMatmulIntoSquare|BM_BlockedMatmulSquare'
  JSON_OUT="${RESULTS_DIR}/matmul_blocked_compare_${STAMP}.json"
  TXT_OUT="${RESULTS_DIR}/matmul_blocked_compare_${STAMP}.txt"
  LATEST_JSON="${RESULTS_DIR}/matmul_blocked_compare_latest.json"
  LATEST_TXT="${RESULTS_DIR}/matmul_blocked_compare_latest.txt"
elif [[ "${MODE}" == "sgemm" ]]; then
  BENCHMARK_FILTER='--benchmark_filter=BM_MatrixMatmulIntoSquare|BM_CblasSgemmSquare'
  JSON_OUT="${RESULTS_DIR}/matmul_sgemm_compare_${STAMP}.json"
  TXT_OUT="${RESULTS_DIR}/matmul_sgemm_compare_${STAMP}.txt"
  LATEST_JSON="${RESULTS_DIR}/matmul_sgemm_compare_latest.json"
  LATEST_TXT="${RESULTS_DIR}/matmul_sgemm_compare_latest.txt"
  BENCH_ENV=(env VECLIB_MAXIMUM_THREADS=1)
elif [[ "${MODE}" == "mt" ]]; then
  BENCHMARK_FILTER='--benchmark_filter=BM_MultithreadedMatmulSquare'
  JSON_OUT="${RESULTS_DIR}/matmul_mt_compare_${STAMP}.json"
  TXT_OUT="${RESULTS_DIR}/matmul_mt_compare_${STAMP}.txt"
  LATEST_JSON="${RESULTS_DIR}/matmul_mt_compare_latest.json"
  LATEST_TXT="${RESULTS_DIR}/matmul_mt_compare_latest.txt"
elif [[ "${MODE}" == "register" ]]; then
  BENCHMARK_FILTER='--benchmark_filter=BM_RegisterOptimizedMatmulSquare'
  JSON_OUT="${RESULTS_DIR}/matmul_register_compare_${STAMP}.json"
  TXT_OUT="${RESULTS_DIR}/matmul_register_compare_${STAMP}.txt"
  LATEST_JSON="${RESULTS_DIR}/matmul_register_compare_latest.json"
  LATEST_TXT="${RESULTS_DIR}/matmul_register_compare_latest.txt"
fi

mkdir -p "${RESULTS_DIR}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLEARN_BUILD_BENCHMARKS=ON

cmake --build "${BUILD_DIR}" --target benchmark_matmul --config Release -j

BENCH="${BUILD_DIR}/benchmark_matmul"
if [[ ! -x "${BENCH}" && -x "${BUILD_DIR}/Release/benchmark_matmul" ]]; then
  BENCH="${BUILD_DIR}/Release/benchmark_matmul"
fi

# macOS bash 3.2 + set -u: empty "${arr[@]}" is an unbound-variable error.
if ((${#BENCH_ENV[@]})); then
  "${BENCH_ENV[@]}" "${BENCH}" \
    ${BENCHMARK_FILTER} \
    --benchmark_out="${JSON_OUT}" \
    --benchmark_out_format=json \
    --benchmark_display_aggregates_only=false \
    | tee "${TXT_OUT}"
else
  "${BENCH}" \
    ${BENCHMARK_FILTER} \
    --benchmark_out="${JSON_OUT}" \
    --benchmark_out_format=json \
    --benchmark_display_aggregates_only=false \
    | tee "${TXT_OUT}"
fi

cp "${JSON_OUT}" "${LATEST_JSON}"
cp "${TXT_OUT}" "${LATEST_TXT}"

echo
echo "Saved:"
echo "  ${JSON_OUT}"
echo "  ${TXT_OUT}"
echo "  ${LATEST_JSON} (copy)"
echo "  ${LATEST_TXT} (copy)"
