//#include "configuration.hpp"

#include "factories/aabb_factory.hpp"
#include "factories/curve_factory.hpp"
#include "factories/factory.hpp"
#include "factories/point_factory.hpp"
#include "factories/pta_factory.hpp"
#include "factories/triangle_input_factory.hpp"
#include "helpers/argparser.hpp"
#include "helpers/data_loader.hpp"
#include "helpers/input_generator.hpp"
#include "helpers/optix_pipeline.hpp"
#include "helpers/optix_wrapper.hpp"
#include "helpers/pretty_printers.hpp"
#include "helpers/time.hpp"
#include "launch_parameters.hpp"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"
#include "types.hpp"
#include "helpers/spatial_helpers.cuh"

#include <numeric>
#include <vector>

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

int main(const int argc, const char** argv) {
	geo_rt_index::helpers::ArgParser parser{argc, argv};
	auto args = parser.Parse();

    const constexpr bool debug = false;
    optix_wrapper optix(debug);
    optix_pipeline pipeline(&optix);
    cudaDeviceSynchronize(); CUERR

	std::vector<Point> points;
	MEASURE_TIME("Loading points",
	 	points = DataLoader::Load(args.files);
	);
	const auto num_points = points.size();

	std::vector<OptixAabb> z_adjusted;
	z_adjusted.reserve(args.queries.size());
	for(auto query : args.queries)
	{
		z_adjusted.push_back(query.ToOptixAabb(3,4));
	}

	D_PRINT("z: %zu\n", z_adjusted.size());
	PointToAABBFactory f{points, z_adjusted};

	unique_ptr<cuda_buffer> result_d = std::make_unique<cuda_buffer>();
	const auto num_queries = args.queries.size();
	const auto result_size =num_queries * num_points;
	auto result = std::make_unique<bool*>(new bool[result_size]);
	memset(*result, 0, result_size);
	result_d->alloc(sizeof(bool) * result_size);
	result_d->upload(*result, result_size);

//	auto device_hit_count = std::make_unique<uint32_t*>(new uint32_t[num_queries]);
//	memset(*device_hit_count, 0, num_queries * sizeof(uint32_t));
//	std::vector<uint32_t> device_hit_count(args.queries.size(), 0);
//	D_PRINT("%zu\n", device_hit_count.size());
//	cuda_buffer hit_count_d;
//	hit_count_d.alloc(sizeof(uint32_t) * num_queries);
//	hit_count_d.upload(*device_hit_count, num_queries);

	auto handle = foo(optix, f);
	LaunchParameters launch_params
	{
		.traversable = handle,
		.points = f.GetPointsDevicePointer(),
		.num_points = points.size(),
		.result_d = result_d->ptr<bool>(),
//		.hit_count_d = hit_count_d.ptr<uint32_t*>(),
		.queries = f.GetQueriesDevicePointer()
	};

	cuda_buffer launch_params_d;
	launch_params_d.alloc(sizeof(launch_params));
	launch_params_d.upload(&launch_params, 1);
	cudaDeviceSynchronize(); CUERR

	MEASURE_TIME("Query execution",
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
		cudaDeviceSynchronize();
	);
	CUERR

	MEASURE_TIME("result_d->download", result_d->download(*result, num_points * num_queries););

#ifdef VERIFICATION_MODE
	MEASURE_TIME("Result check",
	D_PRINT("Checking device results\n");
	uint32_t hit_count = 0;
	for(uint32_t i = 0; i < num_queries; i++)
	{
		const auto& aabb_query = args.queries.at(i);
		const auto& optixAabb_query = z_adjusted.at(i);
		for(uint32_t j = 0; j < num_points; j++)
		{
			const auto point = points.at(j);
			auto aabb_result = helpers::SpatialHelpers::Contains(aabb_query, point);
			auto optixAabb_result = helpers::SpatialHelpers::Contains(optixAabb_query, point);
			auto device_result = (*result)[(num_points * i) + j];
			if(aabb_result != device_result)
			{
				throw std::runtime_error("aabb_result != device_result");
			}
			if(aabb_result != optixAabb_result)
			{
				throw std::runtime_error("aabb_result != optixAabb_result");
			}
			if(aabb_result && optixAabb_result && device_result)
			{
				hit_count++;
			}
			else if(aabb_result | optixAabb_result | device_result)
			{
				throw std::runtime_error("Unknown error");
			}
		}
	}
	std::cout << "Hit count: " << std::to_string(hit_count) << '\n';
	);


#endif




	return 0;
}