add_subdirectory(playground)
add_subdirectory(helpers)
add_subdirectory(factories)

set(SOURCES ${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/device_code.cu
	${CMAKE_CURRENT_SOURCE_DIR}/main.cu
	PARENT_SCOPE
)