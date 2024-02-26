#include "helpers/optix_pipeline.hpp"
#include "helpers/optix_wrapper.hpp"
#include "launch_parameters.hpp"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "types.hpp"

using std::vector;

std::unique_ptr<OptixTraversableHandle> foo(optix_wrapper& optix, const cuda_buffer& triangles_d) {
	static const constexpr uint32_t flags[] = {
	    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
	};
	std::unique_ptr<OptixTraversableHandle> handle(new OptixTraversableHandle{0});
	OptixBuildInput bi{};
	memset(&bi, 0, sizeof(OptixBuildInput));
	bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	bi.triangleArray.numVertices         = triangle::vertex_count();
	bi.triangleArray.vertexBuffers       = (CUdeviceptr*) &triangles_d.raw_ptr;
	bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	bi.triangleArray.vertexStrideInBytes = sizeof(float3);
	bi.triangleArray.numIndexTriplets    = 0;
	bi.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
	bi.triangleArray.preTransform        = 0;
	bi.triangleArray.flags               = flags;
	bi.triangleArray.numSbtRecords       = 1;

	OptixAccelBuildOptions bo;
	memset(&bo, 0, sizeof(OptixAccelBuildOptions));
	bo.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	bo.operation = OPTIX_BUILD_OPERATION_BUILD;
	bo.motionOptions.numKeys = 1;

	OptixAccelBufferSizes structure_buffer_sizes;
	memset(&structure_buffer_sizes, 0, sizeof(OptixAccelBufferSizes));
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optix.optix_context, &bo, &bi,
	                                         1, // num_build_inputs
	                                         &structure_buffer_sizes))
	auto uncompacted_size = structure_buffer_sizes.outputSizeInBytes;

	cuda_buffer compacted_size_buffer;
	compacted_size_buffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = compacted_size_buffer.cu_ptr();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================
	cuda_buffer uncompacted_structure_buffer;
	uncompacted_structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
	cuda_buffer temp_buffer;
	temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optix.optix_context, optix.stream, &bo, &bi, 1, temp_buffer.cu_ptr(),
	                            temp_buffer.size_in_bytes, uncompacted_structure_buffer.cu_ptr(),
	                            uncompacted_structure_buffer.size_in_bytes, &*handle, &emit_desc, 1))
	cudaDeviceSynchronize();
	CUERR
	temp_buffer.free();
	return std::move(handle);
}

int main() {
    const constexpr bool debug = true;
    optix_wrapper optix(debug);
    optix_pipeline pipeline(&optix);

    vector<triangle> triangles {
        {
                {1, 1, 1},
                {2, 1, 1},
                {1, 2, 1},
        }
    };

    cuda_buffer triangles_d, as, result_d;
    triangles_d.alloc_and_upload(triangles);
	result_d.alloc(triangles.size() * sizeof(uint32_t));
    cudaDeviceSynchronize(); CUERR

	auto handle = foo(optix, triangles_d);
    launch_parameters launch_params;
	launch_params.traversable = *handle;
	launch_params.triangles_d = triangles_d.ptr<triangle>();
	launch_params.result_d = result_d.ptr<uint32_t>();

	cuda_buffer launch_params_d;
	launch_params_d.alloc(sizeof(launch_params));
	launch_params_d.upload(&launch_params, 1);
	cudaDeviceSynchronize(); CUERR

	OPTIX_CHECK(optixLaunch(
	    pipeline.pipeline,
	    optix.stream,
	    launch_params_d.cu_ptr(),
	    launch_params_d.size_in_bytes,
	    &pipeline.sbt,
	    1,
	    1,
	    1
	))


	cudaDeviceSynchronize(); CUERR

	uint32_t res = 15;
	result_d.download(&res, 1);

	std::cout << std::to_string(res) << '\n';

	return 0;
}