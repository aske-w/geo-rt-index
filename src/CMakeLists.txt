add_subdirectory(playground)
add_subdirectory(helpers)
add_subdirectory(factories)

set(SOURCES ${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/main.cu
	${CMAKE_CURRENT_SOURCE_DIR}/device_code.cu
	${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
	PARENT_SCOPE
)

set(ORACLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/query_oracle.cpp PARENT_SCOPE)