cmake_minimum_required(VERSION 3.16)
project(tumbleweed_nnue CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tumbleweed_nnue src/tumbleweed_nnue.cpp)
target_include_directories(tumbleweed_nnue PUBLIC include)

add_executable(check_vectors tests/check_vectors.cpp)
target_link_libraries(check_vectors PRIVATE tumbleweed_nnue)
