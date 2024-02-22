cmake_minimum_required(VERSION 3.22.1)

set(NAME geo-rt-index)
project(${NAME} CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 2070 Super

include_directories(include)
add_subdirectory(src)

add_executable(${NAME} ${SOURCES})

