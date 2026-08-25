# full suite test command : 
# cmake -S . -B build-sanitize
# cmake --build build-sanitize
# ctest --test-dir build-sanitize --output-on-failure


cmake_minimum_required(VERSION 3.16)
project(cpp_exercises CXX)

set(CMAKE_CXX_STANDARD 17)
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
