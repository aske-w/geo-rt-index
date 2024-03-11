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

using std::unique_ptr;
using std::unique_ptr;

using namespace geo_rt_index;
using helpers::optix_pipeline;
using helpers::optix_wrapper;

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

    cuda_buffer /*curve_points_d,*/ as, result_d;
    cudaDeviceSynchronize(); CUERR
#if INDEX_TYPE == 1
	std::random_device rd;
	std::mt19937_64 gen {rd()};
	std::uniform_real_distribution<float> rng {0, 25};
	std::vector<Point> points;
	for (int i = 0; i < 2'000'000; i++)
	{
		points.push_back(Point(rng(gen),rng(gen)));
	}
	PointToAABBFactory f{points};

	f.SetQuery({9, -1, 1, 21, 1, 2});

#else
//	TriangleFactory f{};
	AabbFactory f{};
//	PointFactory f{};
#endif
	auto handle = foo(optix, f);
	result_d.alloc(sizeof(uint32_t) * 2);
	LaunchParameters launch_params
	{
		.traversable = handle,
#if INDEX_TYPE == 1
		.points = f.GetPointsDevicePointer(),
		.num_points = points.size(),
#endif
		.result_d = result_d.ptr<uint32_t>(),
	};

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

//	uint32_t res[2];
//	result_d.download(res, 2);
//	for (auto re : res)
//	{
//		std::cout << std::to_string(re) << '\n';
//	}

	return 0;
}