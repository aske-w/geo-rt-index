add_subdirectory(playground)
add_subdirectory(helpers)
add_subdirectory(factories)
add_subdirectory(quadtree)

set(SOURCES ${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/main.cu
	${CMAKE_CURRENT_SOURCE_DIR}/device_code.cu
	PARENT_SCOPE
)

set(QUADTREE_SOURCES ${QUADTREE_SOURCES} PARENT_SCOPE)