// Square Tensor::matmul microbenchmarks (Google Benchmark).
//
// Build (Release, no sanitizers):
// cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLEARN_BUILD_BENCHMARKS=ON && cmake --build build-bench --target benchmark_matmul -j && ./build-bench/benchmark_matmul
//
// Or use: ./benchmarks/run_matmul_bench.sh

#include "my_tensor.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>

namespace {

learn::Tensor<float> make_square(std::size_t n, float scale, float stride) {
    learn::Vector<std::size_t> shape;
    shape.push_back(n);
    shape.push_back(n);
    learn::Tensor<float> t(shape);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<float>(i) * scale + stride;
    }
    return t;
}

void BM_MatmulSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    // Construct and initialize outside the timed loop.
    const learn::Tensor<float> a = make_square(n, 0.001f, 0.1f);
    const learn::Tensor<float> b = make_square(n, 0.002f, -0.05f);

    for (auto _ : state) {
        learn::Tensor<float> result = a.matmul(b);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    // Rough FLOP count for n x n @ n x n: 2 * n^3
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * n * n);
    state.counters["n"] = static_cast<double>(n);
}

}  // namespace

BENCHMARK(BM_MatmulSquare)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
