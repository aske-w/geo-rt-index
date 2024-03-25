//#include "configuration.hpp"
#include "factories/aabb_factory.hpp"
#include "factories/curve_factory.hpp"
#include "factories/factory.hpp"
#include "factories/point_factory.hpp"
#include "factories/pta_factory.hpp"
#include "factories/triangle_input_factory.hpp"
#include "helpers/optix_pipeline.hpp"
#include "helpers/optix_wrapper.hpp"
#include "launch_parameters.hpp"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "types.hpp"
#include <vector>
#include <random>
#include "helpers/input_generator.hpp"
#include <chrono>

//#include "device_code.cu"


using std::unique_ptr;
using std::unique_ptr;

using namespace geo_rt_index;
using helpers::optix_pipeline;
using helpers::optix_wrapper;
using helpers::cuda_buffer;
using factories::Factory;
using factories::PointToAABBFactory;

OptixTraversableHandle foo(optix_wrapper& optix, Factory<OptixBuildInput>& inputFactory) {
	OptixTraversableHandle handle{0};
	unique_ptr<OptixBuildInput> bi = inputFactory.Build();

	OptixAccelBuildOptions bo {};
	memset(&bo, 0, sizeof(OptixAccelBuildOptions));
	bo.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	bo.operation = OPTIX_BUILD_OPERATION_BUILD;
	bo.motionOptions.numKeys = 1;

	OptixAccelBufferSizes structure_buffer_sizes;
	memset(&structure_buffer_sizes, 0, sizeof(OptixAccelBufferSizes));
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optix.optix_context, &bo, &*bi,
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

	OPTIX_CHECK(optixAccelBuild(optix.optix_context, optix.stream, &bo, &*bi, 1, temp_buffer.cu_ptr(),
	                            temp_buffer.size_in_bytes, uncompacted_structure_buffer.cu_ptr(),
	                            uncompacted_structure_buffer.size_in_bytes, &handle, &emit_desc, 1))
	cudaDeviceSynchronize();
	CUERR
	temp_buffer.free();
	return handle;
}

int main() {
    const constexpr bool debug = true;
    optix_wrapper optix(debug);
    optix_pipeline pipeline(&optix);
    cudaDeviceSynchronize(); CUERR


    cuda_buffer /*curve_points_d,*/ as;
	const uint32_t num_points = 1 << 22; // 262.144
	const uint32_t num_in_range = 1 << 11;
	const auto query = OptixAabb{455, 333, 2, 1000, 444, 4};
	const auto space = OptixAabb{0, 0, 0, 200000, 2000, 0};
	auto points_p = InputGenerator::Generate(query, space, num_points, num_in_range);
	auto points = *points_p;

#if INDEX_TYPE == 1
	PointToAABBFactory f{points};
	f.SetQuery(query);

#else
//	TriangleFactory f{};
	AabbFactory f{};
//	PointFactory f{};
#endif

	unique_ptr<cuda_buffer> result_d = std::make_unique<cuda_buffer>();
	auto result = std::make_unique<bool*>(new bool[num_points]);
	memset(*result, 0, num_points);
	result_d->alloc(sizeof(bool) * num_points);
	result_d->upload(*result, num_points);
	uint32_t device_hit_count = 0;
	cuda_buffer hit_count_d;
	hit_count_d.alloc(sizeof(uint32_t));
	hit_count_d.upload(&device_hit_count, 1);

	auto handle = foo(optix, f);
	LaunchParameters launch_params
	{
		.traversable = handle,
#if INDEX_TYPE == 1
		.points = f.GetPointsDevicePointer(),
		.num_points = points.size(),
#endif
		.result_d = result_d->ptr<bool>(),
		.hit_count = hit_count_d.ptr<uint32_t>()
	};

	printf("launch parms num_points %u\n", launch_params.num_points);

	cuda_buffer launch_params_d;
	launch_params_d.alloc(sizeof(launch_params));
	launch_params_d.upload(&launch_params, 1);
	cudaDeviceSynchronize(); CUERR


	auto begin = std::chrono::steady_clock::now();
	OPTIX_CHECK(optixLaunch(
	    pipeline.pipeline,
	    optix.stream,
	    launch_params_d.cu_ptr(),
	    launch_params_d.size_in_bytes,
	    &pipeline.sbt,
#if SINGLE_THREAD
	    1
#else
	    num_points,
#endif
	    1,
	    1
	))

	cudaDeviceSynchronize(); CUERR
	auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
	printf("%ld.%04ld s.\n", total_time_ms / 1000, total_time_ms % 1000);
//	bool res[num_points];
	result_d->download(*result, num_points);
	hit_count_d.download(&device_hit_count, 1);
	uint32_t hit_count = 0;
	for(uint32_t i = 0; i < num_points; i++)
	{
		if ((*result)[i])
		{
			hit_count++;
//			std::cout << std::to_string(i) << '\n';
		}
	}
	std::cout << std::to_string(hit_count) << '\n';
	std::cout << std::to_string(device_hit_count) << '\n';
	assert(hit_count == num_in_range);
	assert(device_hit_count == num_in_range);


	return 0;
}