cmake_minimum_required(VERSION 3.22.1)

set(NAME geo-rt-index)
project(${NAME} CXX CUDA)
find_package(OptiX REQUIRED)

option(USE_DEBUG "Enable debug statements. Always enabled for debug builds." OFF)
if (USE_DEBUG OR (${CMAKE_BUILD_TYPE} STREQUAL "Debug"))
	add_definitions(-DDEBUG)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 2070 Super

include_directories(include)
include_directories(${OptiX_INCLUDE}) # Set during find_package(OptiX REQUIRED)
add_subdirectory(src)

add_executable(${NAME} ${SOURCES})

