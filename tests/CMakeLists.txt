find_package(Catch2 3 REQUIRED)

set(TEST_SOURCES
	${TEST_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

include_directories(generators)
add_subdirectory(helpers)

add_executable(${NAME}_tests ${TEST_SOURCES})
target_link_libraries(${NAME}_tests PRIVATE
	Catch2::Catch2WithMain
	${CUDA_CUDA_LIBRARY}
	${CUDA_LIBRARIES}
)
include(CTest)
include(Catch)
catch_discover_tests(${NAME}_tests)

target_link_libraries(${NAME}_tests PUBLIC ${CMAKE_PROJECT_NAME}_lib)