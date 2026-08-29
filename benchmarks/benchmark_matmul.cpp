// Square matmul microbenchmarks (Google Benchmark).
//
// Build (Release, no sanitizers):
// cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLEARN_BUILD_BENCHMARKS=ON && cmake --build build-bench --target benchmark_matmul -j && ./build-bench/benchmark_matmul
//
// Compare unblocked kernel vs single-thread cblas_sgemm (macOS):
//   VECLIB_MAXIMUM_THREADS=1 ./build-bench/benchmark_matmul --benchmark_filter='BM_MatrixMatmulIntoSquare|BM_CblasSgemmSquare'
//
// Multithreaded matmul at 1024 with 1/2/4/8 threads:
//   ./build-bench/benchmark_matmul --benchmark_filter=BM_MultithreadedMatmulSquare
//
// Compare blocked vs baseline:
//   ./build-bench/benchmark_matmul --benchmark_filter='BM_MatrixMatmulIntoSquare|BM_BlockedMatmulSquare'
//
// Or use:
//   ./benchmarks/run_matmul_bench.sh compare
//   ./benchmarks/run_matmul_bench.sh sgemm
//   ./benchmarks/run_matmul_bench.sh mt

#include "my_matrix.hpp"
#include "my_tensor.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace {

constexpr int kAllSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
constexpr int kBlockedBlockSize = 128;
constexpr int kMultithreadedSize = 1024;
constexpr int kThreadCounts[] = {1, 2, 4, 8};

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

void BM_MatrixMatmulIntoSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    const learn::Matrix<float> a = make_square_matrix(n, 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(n, 0.002f, -0.05f);
    learn::Matrix<float> c(n, n);

    for (auto _ : state) {
        a.matmul_into(c, b);
        benchmark::DoNotOptimize(c.data().data());
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, n);
}

#if defined(__APPLE__)
void BM_CblasSgemmSquare(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));

    const learn::Matrix<float> a = make_square_matrix(static_cast<std::size_t>(n), 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(static_cast<std::size_t>(n), 0.002f, -0.05f);
    learn::Matrix<float> c(static_cast<std::size_t>(n), static_cast<std::size_t>(n));

    for (auto _ : state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n,
                    1.0f,
                    a.data().data(), n,
                    b.data().data(), n,
                    0.0f,
                    c.data().data(), n);
        benchmark::DoNotOptimize(c.data().data());
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, static_cast<std::size_t>(n));
}
#endif

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

void BM_MultithreadedMatmulSquare(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t num_threads = static_cast<std::size_t>(state.range(1));

    const learn::Matrix<float> a = make_square_matrix(n, 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(n, 0.002f, -0.05f);

    for (auto _ : state) {
        learn::Matrix<float> result = a.multithreaded_matmul(b, num_threads);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, n);
    state.counters["threads"] = static_cast<double>(num_threads);
}

void AllMatmulSizeArgs(benchmark::internal::Benchmark* b) {
    for (int n : kAllSizes) {
        b->Arg(n);
    }
}

void BlockedArgs(benchmark::internal::Benchmark* b) {
    for (int n : kAllSizes) {
        b->Args({n, kBlockedBlockSize});
    }
}

void MultithreadedArgs(benchmark::internal::Benchmark* b) {
    for (int threads : kThreadCounts) {
        b->Args({kMultithreadedSize, threads});
    }
}

bool sgemm_matches_matmul(std::size_t n) {
#if !defined(__APPLE__)
    (void)n;
    return true;
#else
    const learn::Matrix<float> a = make_square_matrix(n, 0.001f, 0.1f);
    const learn::Matrix<float> b = make_square_matrix(n, 0.002f, -0.05f);
    learn::Matrix<float> mine(n, n);
    learn::Matrix<float> blas(n, n);

    a.matmul_into(mine, b);
    const int ni = static_cast<int>(n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ni, ni, ni,
                1.0f,
                a.data().data(), ni,
                b.data().data(), ni,
                0.0f,
                blas.data().data(), ni);

    for (std::size_t i = 0; i < n * n; ++i) {
        if (mine.data()[i] != blas.data()[i]) {
            return false;
        }
    }
    return true;
#endif
}

}  // namespace

BENCHMARK(BM_MatmulSquare)
    ->Apply(AllMatmulSizeArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMatmulIntoSquare)
    ->Apply(AllMatmulSizeArgs)
    ->Unit(benchmark::kMillisecond);

#if defined(__APPLE__)
BENCHMARK(BM_CblasSgemmSquare)
    ->Apply(AllMatmulSizeArgs)
    ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK(BM_BlockedMatmulSquare)
    ->Apply(BlockedArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MultithreadedMatmulSquare)
    ->Apply(MultithreadedArgs)
    ->Unit(benchmark::kMillisecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

int main(int argc, char** argv) {
#if defined(__APPLE__)
    if (const char* threads = std::getenv("VECLIB_MAXIMUM_THREADS")) {
        std::fprintf(stderr,
                     "Note: VECLIB_MAXIMUM_THREADS=%s (set to 1 for single-thread cblas_sgemm)\n",
                     threads);
    } else {
        std::fprintf(stderr,
                     "Note: VECLIB_MAXIMUM_THREADS is unset; cblas_sgemm may use multiple threads. "
                     "Use VECLIB_MAXIMUM_THREADS=1 for a fair kernel comparison.\n");
    }

    for (std::size_t n : {32, 128, 512}) {
        if (!sgemm_matches_matmul(n)) {
            std::fprintf(stderr, "Sanity check failed: matmul_into != cblas_sgemm at n=%zu\n", n);
            return 1;
        }
    }
#endif

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
