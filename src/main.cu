#include "factories/factory.hpp"
#include "factories/pta_factory.hpp"
#include "helpers/argparser.hpp"
#include "helpers/data_loader.hpp"
#include "helpers/optix_pipeline.hpp"
#include "helpers/optix_wrapper.hpp"
#include "helpers/time.hpp"
#include "launch_parameters.hpp"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include "helpers/spatial_helpers.cuh"
#include "types/point.hpp"

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
using helpers::AabbLayering;
using geo_rt_index::helpers::Args;

OptixTraversableHandle BuildAccelerationStructure(const optix_wrapper& optix,
    Factory<OptixBuildInput>& input_factory)
{
	OptixTraversableHandle handle{0};
	auto bi = input_factory.Build();

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

void Run(const std::vector<Point>& points)
{
	const auto& args = geo_rt_index::helpers::Args::GetInstance();
	const auto& queries = args.GetQueries();
	const constexpr bool debug = false;
	optix_wrapper optix(debug);
	optix_pipeline pipeline(&optix);
	cudaDeviceSynchronize(); CUERR

	const auto num_points = points.size();

	std::vector<OptixAabb> z_adjusted;
	const auto num_queries = queries.size();
	uint8_t offset = 0;
	switch (args.GetLayering())
	{
	case AabbLayering::None:
		offset = 0;
		break;
	case AabbLayering::Stacked:
		offset = 1;
		break;
	case AabbLayering::StackedSpaced:
		offset = 2;
		break;
	default:
		offset = 0;
		break;
	}
	MEASURE_TIME("Converting to OptixAabb",
		z_adjusted.reserve(queries.size());
		for(size_t i = 0; i < num_queries; i++)
		{
		 	auto current_offset = i * offset;
		 	z_adjusted.push_back(queries.at(i).ToOptixAabb(current_offset, 1 + current_offset));
		}
	);

	PointToAABBFactory f{points, z_adjusted};

	cuda_buffer result_d;
	bool* result;
	MEASURE_TIME("Alloc+upload result buffer",
	             const auto result_size =num_queries * num_points;
	             result = new bool[result_size];
	             memset(result, 0, result_size);
	             result_d.alloc(sizeof(bool) * result_size);
	             result_d.upload(result, result_size);
	             cudaDeviceSynchronize(); CUERR
	);

	OptixTraversableHandle handle;
	MEASURE_TIME("Building AS", handle = BuildAccelerationStructure(optix, f));
	LaunchParameters launch_params
	{
		.traversable = handle,
		.points = f.GetPointsDevicePointer(),
		.num_points = num_points,
		.max_z = num_queries * offset + 4,
		.result_d = result_d.ptr<bool>(),
		.rays_per_thread = args.GetRaysPerThread(),
		.queries = f.GetQueriesDevicePointer()
	};

	cuda_buffer launch_params_d;
	MEASURE_TIME("Alloc+upload launch params",
		launch_params_d.alloc(sizeof(launch_params));
		launch_params_d.upload(&launch_params, 1);
		cudaDeviceSynchronize(); CUERR
	);
	D_PRINT("num_points == %zu\n", num_points);
	D_PRINT("args.GetRaysPerThread() == %u\n", args.GetRaysPerThread());
	D_PRINT("num_points / args.GetRaysPerThread() == %zu\n", num_points / args.GetRaysPerThread());
	const auto num_threads = num_points / args.GetRaysPerThread();
	D_PRINT("num_threads: %zu\n", num_threads);
	MEASURE_TIME("Query execution",
	             OPTIX_CHECK(optixLaunch(
	                 pipeline.pipeline,
	                 optix.stream,
	                 launch_params_d.cu_ptr(),
	                 launch_params_d.size_in_bytes,
	                 &pipeline.sbt,
	                 num_threads,
	                 1,
	                 1
	                 ))
	                 cudaDeviceSynchronize(); CUERR
	);

	MEASURE_TIME("result_d.download", result_d.download(result, num_points * num_queries););

#ifdef VERIFICATION_MODE
	MEASURE_TIME("Result check",
	             D_PRINT("Checking device results\n");
	             uint32_t hit_count = 0;
	             for(uint32_t i = 0; i < num_queries; i++)
	             {
		             D_PRINT("Query %d\n",i);
		             const auto& aabb_query = queries.at(i);
		             const auto& optixAabb_query = z_adjusted.at(i);
		             for(uint32_t j = 0; j < num_points; j++)
		             {
			             const auto point = points.at(j);
			             auto aabb_result = helpers::SpatialHelpers::Contains(aabb_query, point);
			             auto optixAabb_result = helpers::SpatialHelpers::Contains(optixAabb_query, point);
			             auto device_result = result[(num_points * i) + j];
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
	             D_PRINT("Hit count: %u\n", hit_count);
	);


#endif
	delete[] result;
}

int main(const int argc, const char** argv) {
	geo_rt_index::helpers::PrintCSVHeader();
	MEASURE_TIME("Parsing args", Args::Parse(argc, argv));

	std::vector<Point> points;
	MEASURE_TIME("Loading points",
	 	points = DataLoader::Load(Args::GetInstance().GetFiles(), Args::GetInstance().GetModifier());
	);
	MEASURE_TIME("Warmup", Run(points));
	for(size_t i = 0; i < Args::GetInstance().GetRepetitions(); i++)
	{
		MEASURE_TIME("Run", Run(points));
	}
	return 0;
}