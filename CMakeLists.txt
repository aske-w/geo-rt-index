cmake_minimum_required(VERSION 3.22.1)

set(NAME geo-rt-index)
project(${NAME} CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
add_subdirectory(src)

add_executable(${NAME} ${SOURCES})

