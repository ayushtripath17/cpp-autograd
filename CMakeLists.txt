# full suite test command : 
# cmake -S . -B build-sanitize && cmake --build build-sanitize && ctest --test-dir build-sanitize --output-on-failure

# full benchmark command : 
# cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLEARN_BUILD_BENCHMARKS=ON && cmake --build build-bench --target benchmark_matmul -j && ./build-bench/benchmark_matmul

# profile command : 
# cmake --build build-bench --target benchmark_matmul -j

cmake_minimum_required(VERSION 3.16)
project(cpp_exercises CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(add_learn_test target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_learn_test(test_vector tests/test_vector.cpp)
add_learn_test(test_matrix tests/test_matrix.cpp)
add_learn_test(test_tensor tests/test_tensor.cpp)
add_learn_test(test_autograd tests/test_autograd.cpp)
add_learn_test(test_xor tests/test_xor.cpp)
add_learn_test(test_linear tests/test_linear.cpp)

enable_testing()

add_test(NAME vector COMMAND test_vector)
add_test(NAME matrix COMMAND test_matrix)
add_test(NAME tensor COMMAND test_tensor)
add_test(NAME autograd COMMAND test_autograd)
add_test(NAME xor COMMAND test_xor)
add_test(NAME linear COMMAND test_linear)

# ---------------------------------------------------------------------------
# Benchmarks (optional, Release-oriented — do not mix with sanitizers)
#
#   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLEARN_BUILD_BENCHMARKS=ON
#   cmake --build build-bench --target benchmark_matmul
#   ./benchmarks/run_matmul_bench.sh
# ---------------------------------------------------------------------------
option(LEARN_BUILD_BENCHMARKS "Fetch Google Benchmark and build benchmark targets" OFF)

if(LEARN_BUILD_BENCHMARKS)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(benchmark_matmul benchmarks/benchmark_matmul.cpp)
  target_include_directories(benchmark_matmul PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(benchmark_matmul PRIVATE benchmark::benchmark)
  if(APPLE)
    target_link_libraries(benchmark_matmul PRIVATE "-framework Accelerate")
  endif()
  if(NOT MSVC)
    target_compile_options(benchmark_matmul PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
