cmake_minimum_required(VERSION 3.22.1)

set(NAME geo-rt-index)
project(${NAME} LANGUAGES C CXX CUDA) # C is needed for embedding the ptx file
find_package(OptiX REQUIRED)
find_package(CUDA REQUIRED)
#find_package(OpenMP REQUIRED)

option(USE_DEBUG "Enable debug statements. Always enabled for debug builds." OFF)
if (USE_DEBUG OR ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
	add_compile_definitions(DEBUG)
endif()

option(INDEX_TYPE "Index type" 1)
add_compile_definitions(INDEX_TYPE)

option(DEBUG_PRINT "Enable debug print statements" 0)
if((DEFINED DEBUG_PRINT) AND (DEBUG_PRINT STREQUAL "ON"))
	add_compile_definitions(DEBUG_PRINT=1)
else ()
	add_compile_definitions(DEBUG_PRINT=0)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 2070 Super

include_directories(include)
include_directories(${OptiX_INCLUDE}) # Set during find_package(OptiX REQUIRED)
add_subdirectory(src)

# from https://gitlab.rlp.net/juhenneb/rtindex/-/tree/main
macro(cuda_compile_and_embed output_var cuda_file)
	set(c_var_name ${output_var})
	cuda_compile_ptx(ptx_files ${cuda_file})
	list(GET ptx_files 0 ptx_file)
	set(embedded_file ${ptx_file}_embedded.c)
	#  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
	add_custom_command(
		OUTPUT ${embedded_file}
		COMMAND ${BIN2C} -c -p 0 -t char -n ${c_var_name} ${ptx_file} > ${embedded_file}
		DEPENDS ${ptx_file}
		COMMENT "compiling (and embedding ptx from) ${cuda_file}"
	)
	set(${output_var} ${embedded_file})
endmacro()

cuda_compile_and_embed(embedded_ptx_code src/device_code.cu)

add_executable(${NAME} ${embedded_ptx_code} ${SOURCES})

add_executable(kd_tree_CPU ${KD_TREE_CPU_SOURCES})
target_compile_options(kd_tree_CPU PRIVATE "-O3")
target_link_options(kd_tree_CPU PRIVATE "-fopenmp")

add_executable(kd_tree_GPU ${KD_TREE_GPU_SOURCES})
set(CUDA_SAMPLES_COMMON_DIR CACHE STRING "https://github.com/NVIDIA/cuda-samples/tree/master/Common directory on your machine")
set_target_properties(kd_tree_GPU PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_options(kd_tree_GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_75>)
target_compile_options(kd_tree_GPU PRIVATE "-O3" "-I${CUDA_SAMPLES_COMMON_DIR}")
target_link_options(kd_tree_GPU PRIVATE "-fopenmp")

#add_executable(quadtree ${QUADTREE_SOURCES})
#set_property(TARGET quadtree PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#add_executable(qt_cpu ${QT_SOURCES} src/quadtree/quadtree_cpu.cpp)
#add_executable(qt_gpu ${QT_SOURCES} src/quadtree/quadtree_gpu.cu)

target_link_libraries(${NAME}
	${CUDA_CUDA_LIBRARY}
	${CUDA_LIBRARIES}
)


