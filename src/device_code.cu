#include <cuda_runtime.h>
#include "types.hpp"
#include "launch_parameters.hpp"
#include "helpers/optix_helpers.cuh"
#include "../include/types.hpp"
#include "../include/launch_parameters.hpp"
#include "../include/helpers/optix_helpers.cuh"
#include <optix.h>

#include <limits>
#include <cstdint>
#include <cuda_runtime.h>
#include "helpers/spatial_helpers.cuh"
#include "../include/helpers/spatial_helpers.cuh"

using namespace geo_rt_index;

extern "C" __constant__ LaunchParameters params;


extern "C" __global__ void __closesthit__test() {
	D_PRINT("__closesthit__test\n");
    // do nothing
}


extern "C" __global__ void __miss__test() {
//	D_PRINT("__miss__test\n");
    // do nothing
}


// this function is called for every potential ray-aabb intersection
extern "C" __global__ void __intersection__test() {
//	const uint32_t primitive_id = optixGetPrimitiveIndex();
	const uint32_t point_id = optixGetPayload_0();
//	D_PRINT("__intersection__test %u\n", point_id);
	auto contained = helpers::SpatialHelpers::Contains(params.query_aabb, params.points[point_id]);
	if(!contained)
	{
		D_PRINT("False positive hit for %d\n", point_id);
		return;
	}
	atomicAdd(params.hit_count, 1);
	D_PRINT("True positive hit for %d\n", point_id);
//	D_PRINT("Is frontface hit: %x ", optixIsFrontFaceHit());
//	D_PRINT("Is backface hit: %x ", optixIsBackFaceHit());
//	D_PRINT("result_count %u\n", params.result_count);
//	D_PRINT("result_d %llX\n", params.result_d);
//	D_PRINT("access %u\n", params.result_d[x]);
	params.result_d[point_id] = true;
//	D_PRINT("Hit %u\n", optixGetPayload_0());
//	D_PRINT("write");
	optixSetPayload_1(optixGetPayload_1() + 1);
//	set_payload_32(primitive_id);
}


// this function is called for every reported (i.e. confirmed) ray-primitive intersection
extern "C" __global__ void __anyhit__test() {
	const uint32_t primitive_id = optixGetPrimitiveIndex();
//	D_PRINT("__anyhit_test %d\n", primitive_id);
	optixIgnoreIntersection();
}

// this is the entry point
extern "C" __global__ void __raygen__test() {
//	D_PRINT("running from thread ID %d\ntx:%d,ty:%d,tz:%d,bx:%d,by:%d,bdx:%d,bdy:%d\n", threadIdx.x + blockDim.x * blockIdx.x, threadIdx.x, threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	constexpr const uint32_t ray_flags = 0;
#if INDEX_TYPE == 1 // IndexType::PTA
	const auto limit = params.num_points;
	const auto points = params.points;
	const auto t_max = 1e16f;
	uint32_t count = 0;
	uint32_t i = threadIdx.x;
//	for (uint32_t i = 0; i < limit; i++)
//	{
		const Point point = points[i];
		if(0 <= point.x && point.x <= 1 &&
		        0 <= point.y && point.y <= 1)
		{
//			D_PRINT("Point: (%f,%f)\n", point.x, point.y);
		}
		const float3 origin {point.x, point.y, 0};
//		D_PRINT("Origin: (%f,%f,0)\n", point.x, point.y);
		const float3 direction {point.x, point.y, t_max};
//		D_PRINT("Direction: (%f,%f,5)\n", point.x, point.y);
		optixTrace(params.traversable, origin, direction, 0, t_max, 0.0f, OptixVisibilityMask(255), ray_flags, 0, 0,
		           0, i, count);
//		D_PRINT("__raygen_test hit %d\n", i, i0);
//	}
//	D_PRINT("count: %d\n", count);
#else
//	D_PRINT("__raygen_test\n");
//	const constexpr float t_max= 100;
//	const float3 origin {0.5,1.5,0.5};
//	const float3 direction {t_max,1.5,0.5};
//	uint32_t i0 = 0;
//	optixTrace(params.traversable, origin, direction, 0, t_max, 0.0f, OptixVisibilityMask(255), ray_flags, 0, 0,
//			   0, i0);
//	D_PRINT("__raygen_test:%d\n",i0);
//	for(uint i = 0; i < 1000; i++)
//	{
//	}
//	for (float i = -1; i < 1.0f; i += 0.1f)
//		for (float j = -1; j < 1.0f; j += 0.1f)
//			for (float k = -1; k < 1.0f; k += 0.1f)
//			{
//				float3 direction {
//				    i,j,k
//				};
//				optixTrace(params.traversable, origin, direction, -10, 10, 100.0f, OptixVisibilityMask(255), ray_flags, 0, 0,
//				           0, i0);
////
//////				params.result_d[count++] = i0;
//			}
#endif
}

