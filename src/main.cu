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
#include "types/point_sorting.hpp"
#include <numeric>
#include <vector>
#include <future>
#include "helpers/pretty_printers.hpp"
#include "helpers/global_state.hpp"
#include "helpers/point_sort.hpp"
#include "kernel.cuh"

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
using geo_rt_index::types::PointSorting;

OptixTraversableHandle BuildAccelerationStructure(const optix_wrapper& optix,
    Factory<OptixBuildInput>& input_factory, const bool compact)
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
	cudaDeviceSynchronize(); CUERR;

	cuda_buffer compacted_size_buffer;
	compacted_size_buffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emit_desc;
	emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = compacted_size_buffer.cu_ptr();
	D_PRINT("compacted_size_buffer.cu_ptr() %lld\n", compacted_size_buffer.cu_ptr());


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
	cudaDeviceSynchronize(); CUERR;
	
	if(false)
	{
		const uint64_t size = 0;
		MEASURE_TIME("AS compaction",
		compacted_size_buffer.download(&size, 1);
		cuda_buffer compacted_buffer;
		compacted_buffer.alloc(size);
		D_PRINT("handle %lld\n", handle);
		OPTIX_CHECK(optixAccelCompact(optix.optix_context, optix.stream, handle, compacted_buffer.cu_ptr(),
			size, &handle));
		);
		D_PRINT("handle %lld\n", handle);
		geo_rt_index::helpers::PrintCSV("Uncompacted size", static_cast<uint32_t>(uncompacted_size));
		geo_rt_index::helpers::PrintCSV("Compacted size", static_cast<uint32_t>(size));
		uncompacted_structure_buffer.free();
	}
	cudaDeviceSynchronize(); CUERR;
	CUERR
	temp_buffer.free();
	compacted_size_buffer.free();
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
		if (args.GetLayering() == AabbLayering::None)
		{
			const float const_offset = args.GetAabbZValue();
		    for(size_t i = 0; i < num_queries; i++)
		    {
			    z_adjusted.push_back(queries.at(i).ToOptixAabb(const_offset, 1 + const_offset));
		    }
		}
	    else
		{
			for(size_t i = 0; i < num_queries; i++)
			{
		 		const float current_offset = i * offset;
		 		z_adjusted.push_back(queries.at(i).ToOptixAabb(current_offset, 1 + current_offset));
		    }
		}
	);

	PointToAABBFactory f{points, z_adjusted};

	cuda_buffer result_d;
	bool* result;
	const auto result_size = num_queries * num_points;
	MEASURE_TIME("Alloc result buffer",
	             result = new bool[result_size];
	);
	MEASURE_TIME("Memset result buffer",
	             memset(result, 0, result_size);
	);
	MEASURE_TIME("upload result buffer",
	             result_d.alloc(sizeof(bool) * result_size);
	             result_d.upload(result, result_size);
	             cudaDeviceSynchronize(); CUERR
	);
	cudaDeviceSynchronize();

	uint32_t intersect_count = 0;
	cudaDeviceSynchronize();
	cuda_buffer intersect_count_d;
	intersect_count_d.alloc(sizeof(uint32_t));
	intersect_count_d.upload(&intersect_count, 1);
	cudaDeviceSynchronize();

	uint32_t false_positive_count = 0;
	cudaDeviceSynchronize();
	cuda_buffer false_positive_count_d;
	false_positive_count_d.alloc(sizeof(uint32_t));
	false_positive_count_d.upload(&false_positive_count, 1);
	OptixTraversableHandle handle;
	MEASURE_TIME("Building AS", handle = BuildAccelerationStructure(optix, f, args.GetCompaction()));
	LaunchParameters launch_params
	{
		.traversable = handle,
		.points = f.GetPointsDevicePointer(),
		.num_points = num_points,
		.max_z = num_queries * offset + 4,
		.result_d = result_d.ptr<bool>(),
		.rays_per_thread = args.GetRaysPerThread(),
		.false_positive_count = false_positive_count_d.ptr<uint32_t>(),
		.queries = f.GetQueriesDevicePointer(),
		.ray_length = args.GetRayLength(),
		.intersect_count = intersect_count_d.ptr<uint32_t>()
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
	             if(args.IsKernelOnly())
	             {

		             constexpr const uint32_t threads_per_block{1024};
		             constexpr const uint32_t block_x = 65536;
		             const auto block_y = static_cast<uint32_t>(num_points) / ((block_x) * threads_per_block);
		             dim3 grid{block_x, block_y, static_cast<uint32_t>(num_queries)}; // x * y * z blocks
		             KernelFilter<<<grid, threads_per_block>>>(
		                 launch_params_d.ptr<LaunchParameters>());
	             }
	             else
	             {
					 OPTIX_CHECK(optixLaunch(
						 pipeline.pipeline,
						 optix.stream,
						 launch_params_d.cu_ptr(),
						 launch_params_d.size_in_bytes,
						 &pipeline.sbt,
						 num_threads,
						 1,
						 1
						 ));
	             }
				 cudaDeviceSynchronize(); CUERR
	);

	false_positive_count_d.download(&false_positive_count, 1);
	geo_rt_index::helpers::PrintCSV("Errors", false_positive_count);
	intersect_count_d.download(&intersect_count, 1);
	geo_rt_index::helpers::PrintCSV("Total intersections", intersect_count);

	MEASURE_TIME("result_d.download", result_d.download(result, num_points * num_queries););

#ifdef VERIFICATION_MODE
	 MEASURE_TIME("Result check",
		auto query_checker = [&](int i) 
		{
			uint32_t hit_count = 0;
			D_PRINT("Query %d\n",i);
			const auto& aabb_query = queries.at(i);
			const auto& optixAabb_query = z_adjusted.at(i);
			for(uint32_t j = 0; j < num_points; j++)
			{
				const auto point = points.at(j);
				const auto aabb_result = helpers::SpatialHelpers::Contains(aabb_query, point);
				const auto optixAabb_result = helpers::SpatialHelpers::Contains(optixAabb_query, point);
				const auto device_result = result[(num_points * i) + j];
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
		    return hit_count;
		};

		std::vector<std::future<uint32_t>> futures;
		futures.reserve(num_queries);

	    D_PRINT("Checking device results\n");
		for(uint32_t i = 0; i < num_queries; i++)
		{
			auto query_check_handle = std::async(std::launch::async, query_checker, i);
		    futures.push_back(std::move(query_check_handle));
		}
	    uint32_t hit_sum = 0;
	    for(auto&& future : futures)
	    {
		    hit_sum += future.get();
	    }
		D_PRINT("Hit count: %u\n", hit_sum);
	 );

#endif
	delete[] result;
}

int main(const int argc, const char** argv) {
	geo_rt_index::helpers::PrintCSVHeader();
	MEASURE_TIME("Parsing args", Args::Parse(argc, argv));

	const auto& arg_instance = Args::GetInstance();

	std::vector<Point> points;
	MEASURE_TIME("Loading points",
	 	points = DataLoader::Load(arg_instance.GetFiles(), arg_instance.GetModifier());
	);
	MEASURE_TIME("Sort points",
	if (arg_instance.GetPointSorting() != PointSorting::None)
	{
		geo_rt_index::helpers::PointSort::Sort(points, arg_instance.GetPointSorting());
//		std::cout << points.at(0) << '\n';
//		std::cout << points.at(1) << '\n';
//		std::cout << points.at(2) << '\n';
//		std::cout << points.at(3) << '\n';
//		std::cout << points.at(4) << '\n';
	}
	);
	geo_rt_index::helpers::GlobalState::SetIsWarmup(true);
	MEASURE_TIME("Warmup", Run(points));
	geo_rt_index::helpers::GlobalState::SetIsWarmup(false);
	nvtxRangePushA("Benchmark loop");
	for(size_t i = 0; i < Args::GetInstance().GetRepetitions(); i++)
	{
		MEASURE_TIME("Run", Run(points));
	}
	nvtxRangePop();
	return 0;
}