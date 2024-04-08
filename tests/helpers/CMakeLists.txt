add_subdirectory(spatial_helpers)

set(TEST_SOURCES
	${TEST_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/input_generator.cpp
	PARENT_SCOPE
)