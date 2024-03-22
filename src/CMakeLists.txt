add_subdirectory(playground)
add_subdirectory(helpers)
add_subdirectory(factories)
add_subdirectory(quadtree)
add_subdirectory(kd_tree)

set(SOURCES ${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/main.cu
	${CMAKE_CURRENT_SOURCE_DIR}/device_code.cu
	PARENT_SCOPE
)

set(QUADTREE_SOURCES ${QUADTREE_SOURCES} PARENT_SCOPE)
set(KD_TREE_CPU_SOURCES ${KD_TREE_CPU_SOURCES} PARENT_SCOPE)
set(KD_TREE_GPU_SOURCES ${KD_TREE_GPU_SOURCES} PARENT_SCOPE)