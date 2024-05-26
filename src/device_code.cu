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
#include <numeric>
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
	atomicAdd(params.intersect_count, 1);
	const uint32_t primitive_id = optixGetPrimitiveIndex();
	const uint32_t point_id = optixGetPayload_0();
//	D_PRINT("__intersection__test prim: %u point: %u\n", primitive_id, point_id);
	auto contained = helpers::SpatialHelpers::Contains(params.queries[primitive_id], params.points[point_id]);
//	auto aabb = params.queries[primitive_id];
//	auto p =params.points[point_id];
//	D_PRINT("%f, %f, %f, %f contains (%f, %f)? %d\n", aabb.minX, aabb.minY, aabb.maxX, aabb.maxY, p.x, p.y, contained);
//	D_PRINT("Point: %f, %f\n", p.x, p.y);
	if(!contained)
	{
		atomicAdd(params.false_positive_count, 1);
//		D_PRINT("False positive hit for %u %u\n", primitive_id, point_id);
		return;
	}
//	D_PRINT("True positive hit for %d\n", point_id);
//	D_PRINT("Is frontface hit: %x ", optixIsFrontFaceHit());
//	D_PRINT("Is backface hit: %x ", optixIsBackFaceHit());
//	D_PRINT("result_count %u\n", params.result_count);
//	D_PRINT("result_d %llX\n", params.result_d);
//	D_PRINT("access %u\n", params.result_d[x]);
	params.result_d[(primitive_id * params.num_points) + point_id] = true;
//	D_PRINT("Hit %u\n", optixGetPayload_0());
//	D_PRINT("write");
//	set_payload_32(primitive_id);
}


// this function is called for every reported (i.e. confirmed) ray-primitive intersection
extern "C" __global__ void __anyhit__test() {
//	const uint32_t primitive_id = optixGetPrimitiveIndex();
////	D_PRINT("__anyhit_test %d\n", primitive_id);
//	optixIgnoreIntersection();
}

// this is the entry point
extern "C" __global__ void __raygen__test() {
//	D_PRINT("running from thread ID %d\ntx:%d,ty:%d,tz:%d,bx:%d,by:%d,bdx:%d,bdy:%d\n", threadIdx.x + blockDim.x * blockIdx.x, threadIdx.x, threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	constexpr const uint32_t ray_flags = 0;
	const auto points = params.points;
	const auto t_max = params.ray_length;
//	D_PRINT("ray length %f\n",t_max);
	const uint32_t t_idx = threadIdx.x * params.rays_per_thread; // 0 | 4 etc
	const auto limit = t_idx + params.rays_per_thread;
	for (uint32_t i = t_idx; i < limit; i++) // 0,1,2,3 | 4,5,6,7 etc
	{
		const Point point = points[i];
		const float3 origin {point.x, point.y, 0};
//		D_PRINT("Origin: (%f,%f,0)\n", point.x, point.y);
		const float3 direction {point.x, point.y, t_max};
//		D_PRINT("Direction: (%f,%f,5)\n", point.x, point.y);
		optixTrace(params.traversable, origin, direction, 0, t_max, 0, OptixVisibilityMask(255), ray_flags, 0, 0,
		           0, i);
//		D_PRINT("__raygen_test hit %d\n", i, i0);
	}
//	D_PRINT("count: %d\n", count);
}

