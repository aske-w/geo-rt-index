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
#include "device_code.cu"


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

    cuda_buffer /*curve_points_d,*/ as, result_d;
    cudaDeviceSynchronize(); CUERR
#if INDEX_TYPE == 1
	std::random_device rd;
	std::mt19937_64 gen {rd()};
	std::uniform_real_distribution<float> rng {0, 25};
	std::vector<Point> points;
//	uint32_t num_in_range = 500;
//	uint32_t num_points= 11;
//	for (int i = 0; i < num_in_range; i++)
//	{
//		auto x = rng(gen);
//		while(x < 9 || 21 < x)
//			x = rng(gen);
//		auto y = rng(gen);
//		while(y < -1 || 1 < y)
//			y = rng(gen);
//
//		points.push_back(Point(x, y));
//	}
//	for (int i = 0; i < num_points - num_in_range; i++)
//	{
//		auto x = rng(gen);
//		while(x > 9 && 21 > x)
//			x = rng(gen);
//		auto y = rng(gen);
//		while(y > -1 && 1 > y)
//			y = rng(gen);
//		points.push_back(Point(x, y));
//	}
	points.emplace_back(21.f, 1.f);
	points.emplace_back(21.f, -1.f);
	points.emplace_back(9.f, 1.f);
	points.emplace_back(9.f, -1.f);
//	points.emplace_back(10.f, 0.1f);
//	points.emplace_back(10.f, 0.2f);
//	points.emplace_back(11, 0);
//	points.emplace_back(12.f, 0.1f);
//	points.emplace_back(13.f, 0.2f);
//
//	points.emplace_back(310.f, 0.1f);
//	points.emplace_back(310.f, 0.2f);
//	points.emplace_back(311, 0);
//	points.emplace_back(312.f, 0.1f);
//	points.emplace_back(313.f, 0.2f);

	uint32_t num_points = points.size();
	PointToAABBFactory f{points};
	f.SetQuery({9, -1, 1, 21, 1, 2});

#else
//	TriangleFactory f{};
	AabbFactory f{};
//	PointFactory f{};
#endif
	auto handle = foo(optix, f);
	result_d.alloc(sizeof(bool) * num_points);
	cudaMemset(result_d.raw_ptr, 0, num_points);
	LaunchParameters launch_params
	{
		.traversable = handle,
#if INDEX_TYPE == 1
		.points = f.GetPointsDevicePointer(),
		.num_points = points.size(),
#endif
		.result_d = result_d.ptr<bool>(),
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

	bool res[num_points];
	result_d.download(res, num_points);
	uint32_t hit_count = 0;
	for(uint32_t i = 0; i < num_points; i++)
	{
		auto result = res[i];
		if (result)
		{
			hit_count++;
//			std::cout << std::to_string(i) << '\n';
		}
	}
	std::cout << std::to_string(hit_count) << '\n';


	return 0;
}