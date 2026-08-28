// Square matmul microbenchmarks (Google Benchmark).
//
// Build (Release, no sanitizers):
// cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLEARN_BUILD_BENCHMARKS=ON && cmake --build build-bench --target benchmark_matmul -j && ./build-bench/benchmark_matmul
//
// Compare blocked vs baseline at 256–2048:
//   ./build-bench/benchmark_matmul --benchmark_filter='BM_.*Square/(256|512|1024|2048)'
//
// Blocked matmul only:
//   ./build-bench/benchmark_matmul --benchmark_filter=BM_BlockedMatmulSquare
//
// Or use: ./benchmarks/run_matmul_bench.sh

#include "my_matrix.hpp"
#include "my_tensor.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>

namespace {

constexpr int kCompareSizes[] = {256, 512, 1024, 2048};
constexpr int kBlockSizes[] = {8, 16, 32, 64, 128, 256};

learn::Tensor<float> make_square_tensor(std::size_t n, float scale, float stride) {
    learn::Vector<std::size_t> shape;
    shape.push_back(n);
    shape.push_back(n);
    learn::Tensor<float> t(shape);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<float>(i) * scale + stride;
    }
    return t;
}

learn::Matrix<float> make_square_matrix(std::size_t n, float scale, float stride) {
    learn::Matrix<float> m(n, n);
    for (std::size_t i = 0; i < m.size(); ++i) {
        m.data()[i] = static_cast<float>(i) * scale + stride;
    }
    return m;
}

void set_matmul_counters(benchmark::State& state, std::size_t n) {
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * n * n);
    state.counters["n"] = static_cast<double>(n);
}

void BM_MatmulSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    const learn::Tensor<float> a = make_square_tensor(n, 0.001f, 0.1f);
    const learn::Tensor<float> b = make_square_tensor(n, 0.002f, -0.05f);

    for (auto _ : state) {
        learn::Tensor<float> result = a.matmul(b);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, n);
}

void BM_MatrixMatmulSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    const learn::Matrix<float> a = make_square_matrix(n, 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(n, 0.002f, -0.05f);

    for (auto _ : state) {
        learn::Matrix<float> result = a.matmul(b);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, n);
}

void BM_BlockedMatmulSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t block_size = static_cast<std::size_t>(state.range(1));

    const learn::Matrix<float> a = make_square_matrix(n, 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(n, 0.002f, -0.05f);

    for (auto _ : state) {
        learn::Matrix<float> result = a.blocked_matmul(b, block_size);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, n);
    state.counters["block"] = static_cast<double>(block_size);
}

void AllMatmulSizeArgs(benchmark::internal::Benchmark* b) {
    for (int n : {32, 64, 128, 256, 512, 1024, 2048}) {
        b->Arg(n);
    }
}

void CompareSizeArgs(benchmark::internal::Benchmark* b) {
    for (int n : kCompareSizes) {
        b->Arg(n);
    }
}

void BlockedArgs(benchmark::internal::Benchmark* b) {
    for (int n : kCompareSizes) {
        for (int block : kBlockSizes) {
            b->Args({n, block});
        }
    }
}

}  // namespace

BENCHMARK(BM_MatmulSquare)
    ->Apply(AllMatmulSizeArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMatmulSquare)
    ->Apply(CompareSizeArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_BlockedMatmulSquare)
    ->Apply(BlockedArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
