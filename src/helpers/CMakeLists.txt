set(SOURCES ${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/optix_wrapper.cu
	${CMAKE_CURRENT_SOURCE_DIR}/optix_pipeline.cu
	#${CMAKE_CURRENT_SOURCE_DIR}/input_generator.cu
	${CMAKE_CURRENT_SOURCE_DIR}/argparser.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/data_loader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/point_sort.cpp

	PARENT_SCOPE
)